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
"""NVFP4 weight adapter for the Z-Image (``z_image_modulev3``) transformer.

Loads a ComfyUI/comfy-kitchen NVFP4 single-file checkpoint (e.g. SECourses'
``Z_Image_Turbo_NVFP4.safetensors``) into MAX parameter naming. The on-disk
NVFP4 layout is *identical* to Klein's modelopt/BFL format (validated: dequant
vs bf16 base correlates 0.9956 after the two transforms below), so we reuse
Klein's nibble-swap + TCGEN5 scale-deinterleave verbatim. Only the naming map
and the fused-QKV split are Z-Image-specific.

Per-Linear NVFP4 tensors expected on disk (ComfyUI naming):
  <name>.weight         uint8   [N, K//2]     (E2M1, hi-nibble-first)
  <name>.weight_scale   f8e4m3  [N, K//16]    (TCGEN5-interleaved)
  <name>.weight_scale_2 f32     scalar
  <name>.input_scale    f32     scalar
"""

from __future__ import annotations

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData

from .nvfp4_layout import deinterleave_scales, swap_fp4_nibbles

# ComfyUI DiT key fragment -> MAX z_image_modulev3 key fragment. Applied by
# substring replace, longest-first. QKV is handled separately (it splits).
_COMFY_TO_MAX = {
    ".attention.out.": ".attention.to_out.0.",
    ".attention.q_norm.": ".attention.norm_q.",
    ".attention.k_norm.": ".attention.norm_k.",
    ".adaLN_modulation.0.": ".adaLN_modulation.",
    "final_layer.adaLN_modulation.1.": "final_layer.adaLN_modulation.",
    "t_embedder.mlp.0.": "t_embedder.linear_1.",
    "t_embedder.mlp.2.": "t_embedder.linear_2.",
    "cap_embedder.0.": "cap_norm.",
    "cap_embedder.1.": "cap_proj.",
}

_DROP = ("x_pad_token", "cap_pad_token", "siglip_")

# The fused-QKV Linear splits its output (row) dim into three equal parts.
_QKV_SUFFIXES = ("weight", "weight_scale", "weight_scale_2", "input_scale")


def _rename(key: str) -> str:
    for before, after in _COMFY_TO_MAX.items():
        key = key.replace(before, after)
    return key


def _transform_quant(name: str, value: WeightData) -> WeightData:
    """Apply the FP4 layout fixups shared with Klein."""
    if value.dtype == DType.uint8:
        return swap_fp4_nibbles(value)
    if (
        name.endswith(".weight_scale")
        and value.dtype == DType.float8_e4m3fn
        and len(value.shape) == 2
    ):
        return deinterleave_scales(value)
    return value


def _split_rows(value: WeightData, n: int) -> list[WeightData]:
    """Split a rank>=1 WeightData into ``n`` equal chunks along axis 0.

    Scalars (rank 0) and per-tensor scales are replicated (the split Q/K/V
    projections share the fused tensor's ``weight_scale_2`` / ``input_scale``).
    """
    if len(value.shape) == 0 or int(value.shape[0]) % n != 0:
        return [value] * n
    buf = value.to_buffer()
    arr = buf.view(DType.uint8).to_numpy()  # dtype-agnostic byte view
    parts = np.split(arr, n, axis=0)
    from max.driver import Buffer

    out: list[WeightData] = []
    for p in parts:
        sub_shape = (int(value.shape[0]) // n, *tuple(int(s) for s in value.shape[1:]))
        b = Buffer.from_numpy(np.ascontiguousarray(p)).view(value.dtype, sub_shape)
        out.append(WeightData(b, value.name, value.dtype, sub_shape))
    return out


def convert_z_image_nvfp4_state_dict(
    state_dict: dict[str, WeightData],
) -> dict[str, WeightData]:
    """Convert a ComfyUI NVFP4 Z-Image checkpoint to MAX parameter naming."""
    new: dict[str, WeightData] = {}

    # Group fused-QKV tensors by block prefix so we can split them together.
    qkv_groups: dict[str, dict[str, WeightData]] = {}

    for key, value in state_dict.items():
        if key.startswith(_DROP):
            continue

        # Collect ".attention.qkv.<suffix>" for splitting.
        if ".attention.qkv." in key:
            prefix, suffix = key.split(".attention.qkv.", 1)
            if suffix in _QKV_SUFFIXES:
                qkv_groups.setdefault(prefix, {})[suffix] = _transform_quant(
                    key, value
                )
                continue

        name = _rename(key)
        new[name] = _transform_quant(name, value)

    # Split each fused QKV into to_q / to_k / to_v.
    for prefix, tensors in qkv_groups.items():
        for suffix, value in tensors.items():
            parts = _split_rows(value, 3)
            for proj, part in zip(("to_q", "to_k", "to_v"), parts):
                new[f"{prefix}.attention.{proj}.{suffix}"] = part

    return new
