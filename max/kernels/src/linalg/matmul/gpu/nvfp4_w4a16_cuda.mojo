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
"""NVFP4 weight-only (W4A16) matmul for NVIDIA GPUs: dequant -> dense bf16 GEMM.

The NVIDIA sibling of the Apple (`_enqueue_apple_fp4_materialize_dense` in
`matmul/gpu/apple/fp4_matmul.mojo`) and AMD (`mxfp4_dequant_matmul_amd`)
weight-only launchers. It targets NVIDIA GPUs that lack the SM100 native
block-scaled FP4 tensor-core path -- e.g. sm_120 (RTX PRO 6000 Blackwell), which
has no tcgen05/UMMA and cannot consume the SM100 rank-5 TCGEN05 scale interleave.

Approach (materialize -> dense): the packed NVFP4 weight is dequantized to a
transient dense bf16 `[N, K]` buffer with the hardware-neutral
`enqueue_fp4_materialize` (reused verbatim from the Apple path -- it is a plain
LUT + block-scale decode with no PTX/MFMA intrinsics), then the existing dense
bf16 GPU GEMM (`_matmul_gpu`, `transpose_b=True`) computes `out = a @ w_dense^T`.
The activation `a` stays bf16 (W4A16 -- it is NOT dynamically quantized to FP4),
and the weight block scales are PLAIN rank-2 `[N, K // 16]` (NOT the SM100 rank-5
interleave). The NVFP4 per-tensor `weight_scale_2` scalar is applied at the graph
level by the caller (a post-matmul multiply), so it is not handled here.

The transformer weights stay 4-bit-resident in DRAM (the VRAM win); the dense
bf16 weight is a per-op transient. The dense half is the same code path the bf16
model already runs on this GPU, so it adds no new dense-GEMM risk. The routine is
intentionally model- and pipeline-agnostic: any NVFP4 (E2M1 weights + fp8-e4m3
block-16 scales) Linear routed through `_matmul_float4` uses it.
"""

from std.gpu.host import DeviceContext
from layout import Idx, TileTensor, row_major

from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.apple.fp4_dequant import enqueue_fp4_materialize


def nvfp4_w4a16_matmul_cuda(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.bfloat16, ...],
    b_packed: TileTensor[DType.uint8, ...],
    b_scales: TileTensor[DType.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Weight-only NVFP4 matmul: materialize FP4 weight to bf16, then dense GEMM.

    The activation, weight, and scale dtypes are pinned in the signature (bf16 /
    uint8 packed E2M1 / fp8-e4m3 block scales) so they bind directly to
    `enqueue_fp4_materialize`'s concretely-typed operands.

    Args:
        c: Output `[M, N]` (bfloat16 on the supported FLUX.2 path).
        a: Activations `[M, K]` in bfloat16.
        b_packed: Packed NVFP4 weights `[N, K // 2]` in uint8 (two E2M1 nibbles
            per byte, low nibble first).
        b_scales: FP8-E4M3 block scales `[N, K // 16]` (block size 16 along K).
        ctx: Device context.
    """
    # The weight dims N (= c's free dim) and K (= a's contraction dim) are static
    # model dimensions; only M (tokens) is dynamic. Use the static extents for
    # the transient dense weight so the dense GEMM tiles like the AMD sibling.
    comptime static_N = type_of(c).static_shape[1]
    comptime static_K = type_of(a).static_shape[1]

    # Step 1: dequantize the packed NVFP4 weight into a transient dense bf16
    # `[N, K]` buffer (hardware-neutral LUT + per-16-element block-scale decode).
    var wdense_buf = ctx.enqueue_create_buffer[DType.bfloat16](
        static_N * static_K
    )
    var wdense_tt = TileTensor(
        wdense_buf, row_major((Idx[static_N], Idx[static_K]))
    )
    enqueue_fp4_materialize[DType.bfloat16](wdense_tt, b_packed, b_scales, ctx)

    # Step 2: dense bf16 GEMM, `out = a @ w_dense^T` (the same tensor-core path
    # the bf16 model already uses on this GPU). `wdense_tt` is passed directly
    # (read-only in the GEMM), matching the AMD `mxfp4_dequant_matmul_amd` call.
    _matmul_gpu[use_tensor_core=True, transpose_b=True](
        c, a, wdense_tt, ctx
    )

    # Keep the transient weight alive through the async materialize + GEMM
    # enqueue. `DeviceBuffer` frees are stream-ordered, so the free cannot race
    # the GEMM; the `^` transfer pins the handle past both enqueues (mirrors the
    # Apple/AMD sibling launchers).
    _ = wdense_buf^
