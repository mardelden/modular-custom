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

import math

from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.norm import RMSNorm
from max.experimental.nn.sequential import ModuleList
from max.experimental.tensor import Tensor
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_gpu as _flash_attention_gpu
from max.nn.kernels import (
    masked_flash_attention_gpu as _masked_flash_attention_gpu,
)

from .embeddings import apply_rotary_emb
from .quant_linear import NVFP4Linear

flash_attention_gpu = F.functional(_flash_attention_gpu)
masked_flash_attention_gpu = F.functional(_masked_flash_attention_gpu)


class ZImageAttention(Module[..., Tensor]):
    def __init__(
        self,
        dim: int,
        n_heads: int,
        qk_norm: bool,
        eps: float,
        quantize: bool = False,
    ):
        self.head_dim = dim // n_heads
        self.inner_dim = dim
        self.n_heads = n_heads

        proj = NVFP4Linear if quantize else Linear
        self.to_q = proj(dim, dim, bias=False)
        self.to_k = proj(dim, dim, bias=False)
        self.to_v = proj(dim, dim, bias=False)

        self.norm_q = RMSNorm(self.head_dim, eps=eps) if qk_norm else None
        self.norm_k = RMSNorm(self.head_dim, eps=eps) if qk_norm else None

        # Keep ModuleList naming for diffusers-compatible key loading.
        self.to_out = ModuleList([proj(dim, dim, bias=False)])

    def forward(
        self,
        hidden_states: Tensor,
        freqs_cis: tuple[Tensor, Tensor],
        attn_mask: Tensor | None = None,
    ) -> Tensor:
        batch_size = hidden_states.shape[0]
        seq_len = hidden_states.shape[1]

        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        query = F.reshape(
            query, [batch_size, seq_len, self.n_heads, self.head_dim]
        )
        key = F.reshape(key, [batch_size, seq_len, self.n_heads, self.head_dim])
        value = F.reshape(
            value, [batch_size, seq_len, self.n_heads, self.head_dim]
        )

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        query = apply_rotary_emb(
            query,
            freqs_cis,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        key = apply_rotary_emb(
            key,
            freqs_cis,
            use_real=True,
            use_real_unbind_dim=-1,
            sequence_dim=1,
        )
        query = query.cast(value.dtype)
        key = key.cast(value.dtype)

        scale = math.sqrt(1.0 / float(self.head_dim))
        if attn_mask is None:
            # Single-request / non-batched path: plain NULL_MASK kernel,
            # byte-identical to the unbatched behavior.
            out = flash_attention_gpu(
                query,
                key,
                value,
                mask_variant=MHAMaskVariant.NULL_MASK,
                scale=scale,
            )
        else:
            # Dynamic-batch path: ``attn_mask`` is a per-row additive key mask
            # of shape [batch, kv_seq] (0 for real positions, large negative
            # for right-padded text keys). Broadcast to a [batch, q_seq, kv_seq]
            # additive mask so each query excludes pad keys from its softmax --
            # this fully isolates a request from its batch neighbors (a pure
            # key-padding mask, unlike the ragged valid_length kernel).
            mask = F.unsqueeze(attn_mask, 1)  # [batch, 1, kv_seq]
            # The mask's kv dim comes in as a separate symbolic dim from the
            # attention sequence (which is a concat-derived dim); rebind asserts
            # they are equal so broadcast_to can unify them.
            mask = F.rebind(mask, [batch_size, 1, seq_len])
            mask = F.broadcast_to(mask, [batch_size, seq_len, seq_len])
            mask = mask.cast(value.dtype)
            out = masked_flash_attention_gpu(query, key, value, mask, scale=scale)

        out = F.reshape(out, [batch_size, seq_len, self.inner_dim])
        return self.to_out[0](out)
