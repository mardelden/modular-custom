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
"""NVFP4 W4A4 quantize-aware Linear for the modulev3 Z-Image transformer.

Mirrors ``deepseekV3_modulev3/layers/quant_linear.py`` but for NVFP4 W4A4:
routes to the existing native FP4xFP4 op (``mo.matmul.block.scaled.cuda.w4a4``)
via ``F.functional``. The activation is dynamically quantized to FP4 inside the
launcher; only ``MODULAR_NVFP4_W4A4=1`` on sm_120a takes this path.

Leaves are declared flat on the module (not wrapped) so their parameter names
(``weight``/``weight_scale``/``weight_scale_2``/``input_scale``) match the
converted checkpoint exactly (see ``nvfp4_weight_adapter.py``).
"""

from __future__ import annotations

from typing import Literal

from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.tensor import Tensor
from max.nn.kernels import _cuda_w4a4_matmul

# Wrap the Graph-API W4A4 launcher so it accepts experimental Tensors.
_w4a4 = F.functional(_cuda_w4a4_matmul)

NVFP4_BLOCK_K = 16


def nvfp4_matmul(
    x: Tensor, weight: Tensor, weight_scale: Tensor, weight_scale_2: Tensor
) -> Tensor:
    """``x @ dequant(weight).T`` via the native W4A4 kernel.

    ``weight`` is ``uint8 [N, K//2]``, ``weight_scale`` is
    ``float8_e4m3fn [N, K//16]`` (row-major), ``weight_scale_2`` a scalar.
    The launcher requires a rank-2 bf16 activation, so we flatten and cast.
    """
    shp = x.shape
    x2 = x.reshape([-1, shp[-1]]) if len(shp) > 2 else x
    x2 = x2.cast(DType.bfloat16)
    out = _w4a4(x2, weight, weight_scale, weight_scale_2)
    if len(shp) > 2:
        out = out.reshape([*shp[:-1], out.shape[-1]])
    return out


class NVFP4Linear(Module[[Tensor], Tensor]):
    """W4A4 replacement for ``max.experimental.nn.Linear`` (no bias by default)."""

    weight: Tensor
    weight_scale: Tensor
    weight_scale_2: Tensor
    input_scale: Tensor
    bias: Tensor | Literal[0]

    def __init__(self, in_dim: int, out_dim: int, *, bias: bool = False) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.weight = Tensor.zeros((self.out_dim, self.in_dim // 2), dtype=DType.uint8)
        self.weight_scale = Tensor.zeros(
            (self.out_dim, self.in_dim // NVFP4_BLOCK_K), dtype=DType.float8_e4m3fn
        )
        self.weight_scale_2 = Tensor.zeros((), dtype=DType.float32)
        # Present in the checkpoint; the W4A4 kernel dynamic-quants activations
        # so this is carried but unused by the matmul.
        self.input_scale = Tensor.zeros((), dtype=DType.float32)
        self.bias = Tensor.zeros((self.out_dim,)) if bias else 0

    def forward(self, x: Tensor) -> Tensor:
        out = nvfp4_matmul(x, self.weight, self.weight_scale, self.weight_scale_2)
        return out + self.bias
