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
"""Device-timed latency benchmark for the native NVFP4 W4A4 CUDA matmul vs bf16.

NVIDIA only (targets sm_120 / RTX PRO 6000). This is the D2/M2 "does W4A4 beat
bf16?" measurement. Times `out = a[M,K] @ W[N,K]^T` on one GPU and reports median
latency (us), GEMM throughput (GFLOP/s), and the ratio w4a4/bf16 for a sweep of M
(tokens) at fixed FLUX.2-Klein transformer-Linear (N, K):

- `w4a4` : `nvfp4_w4a4_matmul_cuda` -- quantize the bf16 activation to FP4 (dynamic
           per-block-16) then the NATIVE FP4xFP4 block-scaled tensor-core GEMM
           (sm_120a `mma.sync…kind::mxf4nvf4.block_scale`). Includes the activation
           quant kernel + the op's internal transient allocations (the as-deployed
           path). This is the *correctness-first* GEMM (one warp per [16,8] tile);
           optimization is the point of this milestone.
- `bf16` : `_matmul_gpu[use_tensor_core, transpose_b]` on a dense bf16 [N,K] weight
           -- the "if we just ran bf16" baseline we must beat.

L2-FLUSH (realistic ruler): naively timing the same weight back-to-back keeps it
L2-resident, so the DRAM-bandwidth-bound bf16 baseline looks fake-fast. Each timed
iteration reads a DIFFERENT weight copy (rotate over `R` copies whose total bytes
swamp the Blackwell L2), so every timed weight read is a cold DRAM miss. bf16 is
reported BOTH ways: `bf16(flush)` (rotated, realistic -- the number to beat) and
`bf16(resid)` (L2-resident, optimistic). w4a4 always uses rotated weights.

FLOP count is the GEMM's `2*M*N*K` for both paths (so GFLOP/s is comparable).
Timing uses the GPU device timer, median over several batches of back-to-back
launches (steady-state GPU time per launch, no host-sync overhead).
"""

from std.math import ceildiv

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys.info import _accelerator_arch

from layout import Idx, TileTensor
from layout.tile_layout import row_major

from linalg.fp4_utils import NVFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.apple.fp4_dequant import enqueue_fp4_materialize
from linalg.matmul.gpu.nvfp4_w4a4_cuda import (
    nvfp4_w4a4_matmul_cuda,
    nvfp4_w4a4_matmul_cuda_tiled,
)


def _fill_packed(
    packed: UnsafePointer[mut=True, Scalar[DType.uint8], _],
    scales: UnsafePointer[mut=True, Scalar[DType.float8_e4m3fn], _],
    npacked: Int,
    nscale: Int,
    seed: UInt64,
):
    """Fill packed FP4 nibbles + FP8 block scales deterministically (xorshift64).
    """
    var state = seed
    for i in range(npacked):
        state ^= state << UInt64(13)
        state ^= state >> UInt64(7)
        state ^= state << UInt64(17)
        packed[i] = UInt8(state & UInt64(0xFF))
    for i in range(nscale):
        state ^= state << UInt64(13)
        state ^= state >> UInt64(7)
        state ^= state << UInt64(17)
        var v = Int(state % UInt64(4)) + 1
        scales[i] = (Float32(v) * Float32(0.5)).cast[DType.float8_e4m3fn]()


def _median(mut xs: List[Float64]) -> Float64:
    """Median of `xs` (mutates: sorts ascending in place via insertion sort)."""
    var n = len(xs)
    for i in range(1, n):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key
    return xs[n // 2]


def _bench_nk[N: Int, K: Int](ctx: DeviceContext) raises:
    """Sweep M for one static (N, K); time w4a4 / bf16(flush) / bf16(resid)."""
    comptime MAXM = 4096
    comptime packed_k = K // 2
    comptime scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE
    comptime warmup = 10
    comptime batches = 7

    comptime FLUSH_BYTES = 256 * 1024 * 1024
    comptime R = max(4, ceildiv(FLUSH_BYTES, N * packed_k))

    # ---- Host buffers (one copy; uploaded to every rotation slot). ----
    var act_host = ctx.enqueue_create_host_buffer[DType.bfloat16](MAXM * K)
    var packed_host = ctx.enqueue_create_host_buffer[DType.uint8](N * packed_k)
    var scale_host = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](
        N * scale_k
    )
    ctx.synchronize()
    for i in range(MAXM * K):
        act_host[i] = Scalar[DType.bfloat16](Float32((i % 5) - 2))
    _fill_packed(
        packed_host.unsafe_ptr(),
        scale_host.unsafe_ptr(),
        N * packed_k,
        N * scale_k,
        UInt64(0xF94ED7042B),
    )

    var act_dev = ctx.enqueue_create_buffer[DType.bfloat16](MAXM * K)
    var out_dev = ctx.enqueue_create_buffer[DType.bfloat16](MAXM * N)
    # weight_scale_2 for the fused epilogue fold (1.0 -> numerics unchanged).
    var s2_dev = ctx.enqueue_create_buffer[DType.float32](1)
    s2_dev.enqueue_fill(Float32(1.0))
    var s2_tt = TileTensor(
        s2_dev.unsafe_ptr(), row_major(Idx[1], Idx[1])
    ).as_immut()
    ctx.enqueue_copy(act_dev, act_host)

    # ---- R rotated weight copies (packed FP4 / FP8 scales / dense bf16). ----
    var packed_slots = List[DeviceBuffer[DType.uint8]](capacity=R)
    var scale_slots = List[DeviceBuffer[DType.float8_e4m3fn]](capacity=R)
    var wbf16_slots = List[DeviceBuffer[DType.bfloat16]](capacity=R)
    for _j in range(R):
        var pb = ctx.enqueue_create_buffer[DType.uint8](N * packed_k)
        var sb = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N * scale_k)
        var wb = ctx.enqueue_create_buffer[DType.bfloat16](N * K)
        ctx.enqueue_copy(pb, packed_host)
        ctx.enqueue_copy(sb, scale_host)
        var pv = TileTensor(
            pb.unsafe_ptr(), row_major(Idx[N], Idx[packed_k])
        ).as_immut()
        var sv = TileTensor(
            sb.unsafe_ptr(), row_major(Idx[N], Idx[scale_k])
        ).as_immut()
        var wv = TileTensor(wb.unsafe_ptr(), row_major(Idx[N], Idx[K]))
        enqueue_fp4_materialize[DType.bfloat16](wv, pv, sv, ctx)
        packed_slots.append(pb)
        scale_slots.append(sb)
        wbf16_slots.append(wb)
    ctx.synchronize()

    var mlist = List[Int]()
    mlist.append(64)
    mlist.append(256)
    mlist.append(1024)
    mlist.append(4096)

    print("== N =", N, "K =", K, "| L2-flush R =", R, "copies")
    for mi in range(len(mlist)):
        var m = mlist[mi]
        var inner = 100 if m <= 256 else 20

        var a_tt = TileTensor(
            act_dev.unsafe_ptr(), row_major(m, Idx[K])
        ).as_immut()
        var c_tt = TileTensor(out_dev.unsafe_ptr(), row_major(m, Idx[N]))

        @always_inline
        @parameter
        def _t_w4a4(c: DeviceContext, i: Int) raises:
            var s = i % R
            var pv = TileTensor(
                packed_slots[s].unsafe_ptr(), row_major(Idx[N], Idx[packed_k])
            ).as_immut()
            var sv = TileTensor(
                scale_slots[s].unsafe_ptr(), row_major(Idx[N], Idx[scale_k])
            ).as_immut()
            nvfp4_w4a4_matmul_cuda(c_tt, a_tt, pv, sv, s2_tt, c)

        @always_inline
        @parameter
        def _t_w4a4t(c: DeviceContext, i: Int) raises:
            var s = i % R
            var pv = TileTensor(
                packed_slots[s].unsafe_ptr(), row_major(Idx[N], Idx[packed_k])
            ).as_immut()
            var sv = TileTensor(
                scale_slots[s].unsafe_ptr(), row_major(Idx[N], Idx[scale_k])
            ).as_immut()
            nvfp4_w4a4_matmul_cuda_tiled(c_tt, a_tt, pv, sv, s2_tt, c)

        @always_inline
        @parameter
        def _t_bf16_flush(c: DeviceContext, i: Int) raises:
            var s = i % R
            var wv = TileTensor(
                wbf16_slots[s].unsafe_ptr(), row_major(Idx[N], Idx[K])
            ).as_immut()
            _matmul_gpu[use_tensor_core=True, transpose_b=True](
                c_tt, a_tt, wv, c
            )

        @always_inline
        @parameter
        def _t_bf16_res(c: DeviceContext) raises:
            var wv = TileTensor(
                wbf16_slots[0].unsafe_ptr(), row_major(Idx[N], Idx[K])
            ).as_immut()
            _matmul_gpu[use_tensor_core=True, transpose_b=True](
                c_tt, a_tt, wv, c
            )

        for w in range(warmup):
            _t_w4a4(ctx, w)
            _t_w4a4t(ctx, w)
            _t_bf16_flush(ctx, w)
            _t_bf16_res(ctx)
        ctx.synchronize()

        var s_w4a4 = List[Float64]()
        var s_w4a4t = List[Float64]()
        var s_bf16f = List[Float64]()
        var s_bf16r = List[Float64]()
        for _b in range(batches):
            s_w4a4.append(
                Float64(ctx.execution_time_iter[_t_w4a4](inner))
                / Float64(inner)
            )
            s_w4a4t.append(
                Float64(ctx.execution_time_iter[_t_w4a4t](inner))
                / Float64(inner)
            )
            s_bf16f.append(
                Float64(ctx.execution_time_iter[_t_bf16_flush](inner))
                / Float64(inner)
            )
            s_bf16r.append(
                Float64(ctx.execution_time[_t_bf16_res](inner)) / Float64(inner)
            )

        var ns_w4a4 = _median(s_w4a4)
        var ns_w4a4t = _median(s_w4a4t)
        var ns_bf16f = _median(s_bf16f)
        var ns_bf16r = _median(s_bf16r)

        var flops = 2.0 * Float64(m) * Float64(N) * Float64(K)
        print(
            "  M=",
            m,
            "| w4a4",
            ns_w4a4 / 1000.0,
            "us | tiled",
            ns_w4a4t / 1000.0,
            "us",
            flops / ns_w4a4t,
            "GF/s | bf16(flush)",
            ns_bf16f / 1000.0,
            "us",
            flops / ns_bf16f,
            "GF/s | bf16(resid)",
            ns_bf16r / 1000.0,
            "us | tiled/bf16f",
            ns_w4a4t / ns_bf16f,
            "| tiled/naive",
            ns_w4a4t / ns_w4a4,
        )

    _ = act_host^
    _ = packed_host^
    _ = scale_host^
    _ = act_dev^
    _ = out_dev^
    _ = s2_dev^
    _ = packed_slots^
    _ = scale_slots^
    _ = wbf16_slots^


def main() raises:
    comptime if "metal" in _accelerator_arch():
        print("SKIP: NVIDIA GPU required")
        return
    var ctx = DeviceContext()
    print(
        "== bench_nvfp4_w4a4 (warmup=10, batches=7, device-timed, L2-flushed)"
        " -- GPU:",
        ctx.name(),
    )
    # Real FLUX.2-Klein-9B NVFP4 Linear (N, K): model dim 4096.
    _bench_nk[4096, 4096](ctx)      # attention proj (x16)
    _bench_nk[12288, 4096](ctx)     # qkv-ish (x14)
    _bench_nk[4096, 16384](ctx)     # mlp down (x24)
    _bench_nk[36864, 4096](ctx)     # big output (x24)
