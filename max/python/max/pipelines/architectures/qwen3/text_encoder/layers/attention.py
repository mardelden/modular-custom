# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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

"""Encoder-only attention without KV cache."""

from __future__ import annotations

import math

from max.dtype import DType
from max.graph import DeviceRef, Dim, TensorValue, ops
from max.nn.kernels import masked_flash_attention_gpu
from max.nn.layer import Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.quant_config import QuantConfig
from max.nn.rotary_embedding import RotaryEmbedding


class EncoderAttention(Module):
    """Encoder-only attention without KV cache (Qwen3: interleaved RoPE via rope.forward)."""

    def __init__(
        self,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        scale: float,
        dtype: DType,
        device: DeviceRef,
        rms_norm_eps: float = 1e-6,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.n_heads = num_attention_heads
        self.n_kv_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.scale = scale if scale is not None else math.sqrt(1.0 / head_dim)

        q_dim = head_dim * num_attention_heads
        kv_dim = head_dim * num_key_value_heads

        # `dtype` is the compute dtype (bf16); `quant_config` (when set) makes
        # the projection *weights* fp4/fp8. q_norm/k_norm stay bf16.
        self.q_proj = Linear(
            hidden_size, q_dim, dtype, device, has_bias=False,
            quant_config=quant_config,
        )
        self.k_proj = Linear(
            hidden_size, kv_dim, dtype, device, has_bias=False,
            quant_config=quant_config,
        )
        self.v_proj = Linear(
            hidden_size, kv_dim, dtype, device, has_bias=False,
            quant_config=quant_config,
        )
        self.o_proj = Linear(
            q_dim, hidden_size, dtype, device, has_bias=False,
            quant_config=quant_config,
        )

        self.q_norm = RMSNorm(
            head_dim,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            head_dim,
            dtype=dtype,
            eps=rms_norm_eps,
            multiply_before_cast=False,
        )

    def _repeat_kv(self, x: TensorValue, n_rep: int) -> TensorValue:
        """Repeat KV heads for GQA (Grouped Query Attention).

        Args:
            x: Input tensor with shape [batch, seq_len, n_kv_heads, head_dim]
            n_rep: Number of times to repeat each head

        Returns:
            Tensor with shape [batch, seq_len, n_kv_heads * n_rep, head_dim]
        """
        if n_rep == 1:
            return x

        batch = x.shape[0]
        seq_len = x.shape[1]
        n_kv_heads = x.shape[2]
        head_dim = x.shape[3]

        # [B, S, H_kv, D] -> [B, S, H_kv, 1, D] -> [B, S, H_kv, n_rep, D]
        #   -> [B, S, H, D]. Use concat instead of tile: tile has no GPU
        # implementation and forces a CPU round-trip for every layer.
        x = ops.unsqueeze(x, 3)
        x = ops.concat([x] * n_rep, axis=3)
        return ops.reshape(x, (batch, seq_len, n_kv_heads * n_rep, head_dim))

    def __call__(
        self,
        x: TensorValue,
        rope: RotaryEmbedding,
        attention_bias: TensorValue,
    ) -> TensorValue:
        """Forward pass computing self-attention over a batch of sequences.

        Args:
            x: Input tensor with shape [batch, seq_len, hidden_dim]
            rope: RotaryEmbedding module
            attention_bias: Additive mask, shape [batch, 1, seq_len, seq_len]
        Returns:
            Output tensor with shape [batch, seq_len, hidden_dim]
        """
        batch = x.shape[0]
        seq_len = x.shape[1]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = ops.reshape(q, (batch, seq_len, self.n_heads, self.head_dim))
        k = ops.reshape(k, (batch, seq_len, self.n_kv_heads, self.head_dim))
        v = ops.reshape(v, (batch, seq_len, self.n_kv_heads, self.head_dim))

        # Qwen3: norm over head_dim (per-head), then RoPE.
        q = self.q_norm(q)
        k = self.k_norm(k)

        # RotaryEmbedding.forward expects 4D [B, S, H, D] (already batched).
        q = rope(q, start_pos=Dim(0), seq_len=seq_len)
        k = rope(k, start_pos=Dim(0), seq_len=seq_len)

        # GQA: expand K, V if needed.
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            k = self._repeat_kv(k, n_rep)
            v = self._repeat_kv(v, n_rep)

        # q/k/v are [B, S, H, D]; mask [B, 1, S, S] -> [B, S, S] broadcasts
        # across heads.
        attn_out = masked_flash_attention_gpu(
            q,
            k,
            v,
            mask=ops.squeeze(attention_bias, axis=1),
            scale=self.scale,
        )
        attn_out = ops.reshape(attn_out, (batch, seq_len, -1))
        return self.o_proj(attn_out)
