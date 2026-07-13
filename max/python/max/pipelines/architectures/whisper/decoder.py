# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Whisper text decoder (Graph API).

Pre-LN transformer decoder with causal self-attention, encoder-decoder cross
attention, and a tied LM head (``proj_out`` is tied to ``embed_tokens`` in every
Whisper checkpoint, so we reuse ``embed_tokens.weight`` for the output matmul).

v1 simplifications (correctness-first; perf deferred to the on-hardware loop):
  * No KV cache — the decode loop recomputes the full prefix each step.
  * Cross-attention projects K/V from ``encoder_states`` inside the graph each
    call (rather than a separate precompute graph). Precomputing cross-K/V is a
    named perf follow-up.

Two build flavors (chosen at graph-build time, no runtime branch):
  * ``return_alignment=False`` -> full logits ``[batch, seq_len, vocab]``.
  * ``return_alignment=True``  -> full logits + cross-attention probs for the
    configured alignment heads ``[n_align_heads, seq_len, 1500]`` (for DTW).
"""

from __future__ import annotations

import math

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, ops
from max.nn.embedding import Embedding
from max.nn.layer import Module
from max.nn.layer.layer_list import LayerList
from max.nn.linear import Linear
from max.nn.norm import LayerNorm
from transformers import AutoConfig

from .encoder import MLP

# Additive causal-mask fill: large negative so softmax -> 0, but small enough to
# stay finite when added to scaled scores in float32 (avoids inf/NaN).
NEG_INF = -1.0e9


class WhisperDecoderSelfAttention(Module):
    """Causal multi-head self-attention (``k_proj`` has no bias, per checkpoint)."""

    def __init__(
        self, d_model: int, n_heads: int, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = Linear(d_model, d_model, dtype, device, has_bias=True)
        self.wk = Linear(d_model, d_model, dtype, device, has_bias=False)
        self.wv = Linear(d_model, d_model, dtype, device, has_bias=True)
        self.wo = Linear(d_model, d_model, dtype, device, has_bias=True)

    def __call__(self, x: TensorValue, mask: TensorValue) -> TensorValue:
        batch, seq_len = x.shape[0], x.shape[1]
        shape = [batch, seq_len, self.n_heads, self.head_dim]
        xq = ops.reshape(self.wq(x), shape).transpose(1, 2)  # [b, H, T, hd]
        xk = ops.reshape(self.wk(x), shape).transpose(1, 2)
        xv = ops.reshape(self.wv(x), shape).transpose(1, 2)
        scale = math.sqrt(1.0 / self.head_dim)
        scores = xq @ ops.transpose(xk, 2, 3)  # [b, H, T, T]
        # mask is additive [b, 1, T, T]; broadcasts over heads.
        probs = ops.softmax(scores * scale + mask)
        out = (probs @ xv).transpose(1, 2).reshape([batch, seq_len, -1])
        return self.wo(out)


class WhisperCrossAttention(Module):
    """Encoder-decoder cross attention; also returns its softmax probs.

    K/V are projected from ``encoder_states`` here (v1). The returned ``probs``
    ``[b, H, T, S]`` feed the word-timestamp DTW for the alignment heads.
    """

    def __init__(
        self, d_model: int, n_heads: int, dtype: DType, device: DeviceRef
    ) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.wq = Linear(d_model, d_model, dtype, device, has_bias=True)
        self.wk = Linear(d_model, d_model, dtype, device, has_bias=False)
        self.wv = Linear(d_model, d_model, dtype, device, has_bias=True)
        self.wo = Linear(d_model, d_model, dtype, device, has_bias=True)

    def __call__(
        self, x: TensorValue, encoder_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        batch, seq_len = x.shape[0], x.shape[1]
        src_len = encoder_states.shape[1]
        q_shape = [batch, seq_len, self.n_heads, self.head_dim]
        kv_shape = [batch, src_len, self.n_heads, self.head_dim]
        xq = ops.reshape(self.wq(x), q_shape).transpose(1, 2)  # [b, H, T, hd]
        xk = ops.reshape(self.wk(encoder_states), kv_shape).transpose(
            1, 2
        )  # [b, H, S, hd]
        xv = ops.reshape(self.wv(encoder_states), kv_shape).transpose(1, 2)
        scale = math.sqrt(1.0 / self.head_dim)
        scores = xq @ ops.transpose(xk, 2, 3)  # [b, H, T, S]
        probs = ops.softmax(scores * scale)
        out = (probs @ xv).transpose(1, 2).reshape([batch, seq_len, -1])
        return self.wo(out), probs


class WhisperDecoderLayer(Module):
    """Pre-LN decoder block: self-attn -> cross-attn -> MLP, each residual."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_dim: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.attention = WhisperDecoderSelfAttention(
            d_model, n_heads, dtype, device
        )
        self.cross_attention = WhisperCrossAttention(
            d_model, n_heads, dtype, device
        )
        self.mlp = MLP(d_model, ffn_dim, dtype, device)
        self.attention_norm = LayerNorm(
            d_model, devices=[device], dtype=dtype, eps=1e-5
        )
        self.cross_attention_norm = LayerNorm(
            d_model, devices=[device], dtype=dtype, eps=1e-5
        )
        self.mlp_norm = LayerNorm(
            d_model, devices=[device], dtype=dtype, eps=1e-5
        )

    def __call__(
        self, x: TensorValue, mask: TensorValue, encoder_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        h = x + self.attention(self.attention_norm(x), mask)
        cross_out, cross_probs = self.cross_attention(
            self.cross_attention_norm(h), encoder_states
        )
        h = h + cross_out
        h = h + self.mlp(self.mlp_norm(h))
        return h, cross_probs


class WhisperDecoder(Module):
    """Whisper text decoder with a tied LM head.

    Args:
        return_alignment: if True, also return stacked cross-attention probs for
            the ``alignment_heads`` (list of ``[layer, head]`` pairs).
    """

    def __init__(
        self,
        huggingface_config: AutoConfig,
        dtype: DType,
        device: DeviceRef,
        *,
        return_alignment: bool = False,
        alignment_heads: list[tuple[int, int]] | None = None,
    ) -> None:
        super().__init__()
        d_model = huggingface_config.d_model
        n_heads = huggingface_config.decoder_attention_heads
        ffn_dim = huggingface_config.decoder_ffn_dim
        self.return_alignment = return_alignment
        self.alignment_heads = list(alignment_heads or [])

        self.embed_tokens = Embedding(
            vocab_size=huggingface_config.vocab_size,
            hidden_dim=d_model,
            dtype=dtype,
            device=device,
        )
        self.embed_positions = Embedding(
            vocab_size=huggingface_config.max_target_positions,
            hidden_dim=d_model,
            dtype=dtype,
            device=device,
        )
        self.layers = LayerList(
            [
                WhisperDecoderLayer(d_model, n_heads, ffn_dim, dtype, device)
                for _ in range(huggingface_config.decoder_layers)
            ]
        )
        # Final decoder LayerNorm (HF `model.decoder.layer_norm`).
        self.norm = LayerNorm(d_model, devices=[device], dtype=dtype, eps=1e-5)

    def __call__(
        self,
        tokens: TensorValue,
        positions: TensorValue,
        mask: TensorValue,
        encoder_states: TensorValue,
    ) -> tuple[TensorValue, ...]:
        h = self.embed_tokens(tokens) + self.embed_positions(positions)

        cross_probs_all: list[TensorValue] = []
        for layer in self.layers:
            h, cross_probs = layer(h, mask, encoder_states)
            cross_probs_all.append(cross_probs)

        h = self.norm(h)
        # Tied LM head: reuse the token-embedding matrix [vocab, d_model].
        lm_weight = ops.transpose(self.embed_tokens.weight, 0, 1)  # [d, vocab]
        logits = ops.cast(h @ lm_weight, DType.float32)  # [b, T, vocab]

        if not self.return_alignment:
            return (logits,)

        # Gather the configured alignment heads: cross_probs_all[l] is
        # [b, H, T, S]; take batch 0, head h -> [T, S]; stack -> [n_align, T, S].
        rows = [
            cross_probs_all[layer][0, head]
            for (layer, head) in self.alignment_heads
        ]
        align = ops.cast(ops.stack(rows, axis=0), DType.float32)
        return logits, align
