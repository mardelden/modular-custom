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
"""Elaboration + numeric smoke test for the fused NVFP4 (W4A16) CUDA matmul.

NVIDIA only. Constructs concrete static-N/K, dynamic-M `TileTensor`s (matching
the op-registration operand contract) and calls `nvfp4_w4a16_fused_matmul_cuda`,
forcing the fused kernel to fully elaborate + GPU-codegen at build time (the
`//max/kernels/src/linalg:linalg` package build only parses it -- the generic op
`execute` is not instantiated there). When run, it also checks the fused output
against an independent fp32 host reference (`E2M1_TO_FLOAT32[nibble] * |scale|`)
across a clean tile, an edge-M/N tile, and a partial-K tail.

The FP4 weight is the B operand (`out = x @ W^T`, `transpose_b=True`): W is
`[N, K]`, packed `[N, K // 2]` (low nibble = even K), scales `[N, K // 16]`.
"""

from std.random import random_si64, seed
from std.gpu.host import DeviceContext

from layout import Idx, TileTensor
from layout.tile_layout import row_major

from linalg.fp4_utils import E2M1_TO_FLOAT32, NVFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu.nvfp4_w4a16_fused_cuda import (
    nvfp4_w4a16_fused_matmul_cuda,
)


def _host_dequant_weight(
    byte: UInt8, nibble_hi: Bool, scale: Scalar[DType.float8_e4m3fn]
) -> Float32:
    """Host mirror of the device dequant: `E2M1_TO_FLOAT32[nibble] * |scale|`."""
    var shift = UInt8(4) if nibble_hi else UInt8(0)
    var nibble = Int((byte >> shift) & UInt8(0xF))
    return E2M1_TO_FLOAT32[nibble] * abs(scale.cast[DType.float32]())


def _run_fused[
    M: Int, N: Int, K: Int
](ctx: DeviceContext, name: String) raises:
    """Elaborate + numerically check the fused kernel for one `[M, N, K]`."""
    print("== fused", name, M, "x", N, "x", K)
    comptime packed_k = K // 2
    comptime scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE

    var act_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var packed_host = ctx.enqueue_create_host_buffer[DType.uint8](N * packed_k)
    var scale_host = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](
        N * scale_k
    )
    ctx.synchronize()
    for i in range(M * K):
        act_host[i] = random_si64(Int64(-2), Int64(2)).cast[DType.bfloat16]()
    for i in range(N * packed_k):
        var lo = UInt8(random_si64(Int64(0), Int64(15)).cast[DType.uint8]())
        var hi = UInt8(random_si64(Int64(0), Int64(15)).cast[DType.uint8]())
        packed_host[i] = lo | (hi << 4)
    for i in range(N * scale_k):
        var pick = random_si64(Int64(1), Int64(4)).cast[DType.float32]()
        scale_host[i] = (pick * Float32(0.5)).cast[DType.float8_e4m3fn]()

    var act_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var packed_dev = ctx.enqueue_create_buffer[DType.uint8](N * packed_k)
    var scale_dev = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N * scale_k)
    var out_dev = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    ctx.enqueue_copy(act_dev, act_host)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(scale_dev, scale_host)

    # Static N/K, dynamic M -- mirrors the op-registration operand contract.
    var m = M
    var c_tt = TileTensor(out_dev.unsafe_ptr(), row_major(m, Idx[N]))
    var a_tt = TileTensor(
        act_dev.unsafe_ptr(), row_major(m, Idx[K])
    ).as_immut()
    var packed_tt = TileTensor(
        packed_dev.unsafe_ptr(), row_major(Idx[N], Idx[packed_k])
    ).as_immut()
    var scale_tt = TileTensor(
        scale_dev.unsafe_ptr(), row_major(Idx[N], Idx[scale_k])
    ).as_immut()

    nvfp4_w4a16_fused_matmul_cuda(c_tt, a_tt, packed_tt, scale_tt, ctx)

    var out_host = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var pass_ = True
    for i in range(M):
        for j in range(N):
            var acc = Float32(0)
            for k in range(K):
                var av = Float32(act_host[i * K + k])
                var byte = packed_host[j * packed_k + (k // 2)]
                var sc = scale_host[j * scale_k + (k // NVFP4_SF_VECTOR_SIZE)]
                acc += av * _host_dequant_weight(byte, (k % 2) == 1, sc)
            var got = Float32(out_host[i * N + j])
            if abs(got - acc) > Float32(1e-2) + Float32(1.6e-2) * abs(acc):
                if pass_:
                    print("FAIL:", i, j, "got", got, "exp", acc)
                pass_ = False

    _ = act_dev^
    _ = packed_dev^
    _ = scale_dev^
    _ = out_dev^

    if not pass_:
        raise Error("FAILED (", name, "; see FAIL lines above)")
    print("PASS")


def main() raises:
    var ctx = DeviceContext()
    seed(0)
    # Clean multi-tile interior (N = 4*BN, K = 16*BK).
    _run_fused[64, 256, 512](ctx, "clean")
    # Edge M (M < BM) -- the small-image regime this kernel targets.
    _run_fused[8, 256, 512](ctx, "edge-m")
    # Ragged M and N (partial M and N tiles).
    _run_fused[100, 200, 64](ctx, "ragged-mn")
    # Partial K tail (K a multiple of 16 but not BK=32).
    _run_fused[64, 128, 48](ctx, "k-tail")
    print("ALL TESTS PASSED")
