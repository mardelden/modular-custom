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
"""NVFP4 on-disk layout fixups (shared with FLUX.2-Klein).

The ComfyUI/comfy-kitchen NVFP4 checkpoint stores the same layout as Klein's
modelopt/BFL format, so these are the same transforms as
``flux2/nvfp4_weight_adapter.py`` (kept local to keep z_image self-contained).
Validated: applying both to the ComfyUI Z-Image file reconstructs the bf16 base
weights at correlation 0.9956.
"""

from __future__ import annotations

import numpy as np
from max.driver import Buffer
from max.dtype import DType
from max.graph.weights import WeightData


def swap_fp4_nibbles(value: WeightData) -> WeightData:
    """Swap packed-FP4 byte nibble order (hi-first on disk -> lo-first kernel)."""
    raw = np.from_dlpack(value.data).view(np.uint8)
    swapped = (
        ((raw & np.uint8(0x0F)) << np.uint8(4)) | ((raw >> np.uint8(4)) & np.uint8(0x0F))
    ).astype(np.uint8)
    return WeightData(swapped, value.name, value.dtype, value.shape)


def deinterleave_scales(value: WeightData) -> WeightData:
    """Convert TCGEN5-interleaved ``[M, K//16]`` fp8 scales to true row-major.

    Storage is the 5D layout ``(M//128, K//64, 32, 4, 4)`` flattened row-major;
    transpose ``(0, 3, 2, 1, 4)`` and reshape back to ``(M, K//16)``.
    """
    SF_MN_GROUP_SIZE = 128
    SF_ATOM_M0 = 32
    SF_ATOM_K = 4

    M, K_div16 = int(value.shape[0]), int(value.shape[1])
    if M % SF_MN_GROUP_SIZE != 0 or K_div16 % SF_ATOM_K != 0:
        return value

    buf = value.to_buffer()
    arr = buf.view(DType.uint8).to_numpy()
    scales_5d = arr.reshape(
        M // SF_MN_GROUP_SIZE,
        K_div16 // SF_ATOM_K,
        SF_ATOM_M0,
        SF_MN_GROUP_SIZE // SF_ATOM_M0,
        SF_ATOM_K,
    )
    deinterleaved = np.ascontiguousarray(
        scales_5d.transpose(0, 3, 2, 1, 4).reshape(M, K_div16)
    )
    deint_buf = Buffer.from_numpy(deinterleaved).view(value.dtype, buf.shape)
    return WeightData(deint_buf, value.name, value.dtype, value.shape)
