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
"""Device-timed latency benchmark for the fused NVFP4 (W4A16) CUDA matmul.

NVIDIA only (targets sm_120 / RTX PRO 6000). Times THREE paths of
`out = a[M,K] @ dequant(W[N,K])^T` on one GPU and reports median latency (us),
GEMM throughput (GFLOP/s), and the ratios fused/bf16 and fused/materialize for a
sweep of M (tokens) at fixed transformer-Linear (N, K):

- `fused`      : `nvfp4_w4a16_fused_matmul_cuda` -- decode packed FP4 -> bf16 in
                 SMEM inside the GEMM mainloop (weight read 4-bit from DRAM,
                 never materialized).
- `materialize`: `nvfp4_w4a16_matmul_cuda` (Phase C) -- dequant the whole packed
                 FP4 weight into a transient dense bf16 [N,K] buffer (allocated
                 per call, as in production), then a dense bf16 GEMM.
- `bf16`       : `_matmul_gpu[use_tensor_core, transpose_b]` on a dense bf16
                 [N,K] weight -- the "if we just ran bf16" baseline.

L2-FLUSH (realistic ruler): naively timing the same weight back-to-back keeps it
L2-resident, so the DRAM-bandwidth-bound baselines (bf16 especially) look
fake-fast (a small-M bf16 GEMM that re-reads an L2-resident weight measures L2
bandwidth, not the DRAM read every real forward pass pays). To fix the ruler,
each timed iteration reads a DIFFERENT copy of the weight -- we allocate `R`
copies whose total bytes swamp the L2 (Blackwell L2 is tens of MB) and rotate
over them via `execution_time_iter` (iteration `i` -> copy `i % R`), so every
timed weight read is a cold DRAM miss. We report the bf16 baseline BOTH ways:
`bf16(flush)` (rotated, realistic -- the number to beat) and `bf16(resid)`
(non-rotated, L2-resident -- optimistic, for reference). `fused` and
`materialize` are always measured on rotated (realistic) weights.

All three write the same bf16 [M,N] output; the FLOP count is the GEMM's
`2*M*N*K` for all three (so GFLOP/s is comparable). Timing uses the GPU device
timer (`ctx.execution_time` / `execution_time_iter`), taking the median over
several batches of back-to-back launches (steady-state GPU time per launch, no
host-sync overhead).
"""

from std.math import ceildiv

from std.gpu.host import DeviceBuffer, DeviceContext
from std.sys.info import _accelerator_arch

from layout import Idx, TileTensor
from layout.tile_layout import row_major

from linalg.fp4_utils import NVFP4_SF_VECTOR_SIZE
from linalg.matmul.gpu import _matmul_gpu
from linalg.matmul.gpu.apple.fp4_dequant import enqueue_fp4_materialize
from linalg.matmul.gpu.nvfp4_w4a16_cuda import nvfp4_w4a16_matmul_cuda
from linalg.matmul.gpu.nvfp4_w4a16_fused_cuda import (
    nvfp4_w4a16_fused_matmul_cuda,
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
    """Sweep M for one static (N, K); time fused / materialize / bf16 each M.

    L2-flush: each timed iteration reads a DIFFERENT copy of the weight (rotate
    over `R` copies whose total bytes >> L2), so the weight is fetched from DRAM
    every iter -- the realistic regime. A separate non-rotated bf16 timing keeps
    the (optimistic) L2-resident number for reference.
    """
    comptime MAXM = 2048
    comptime packed_k = K // 2
    comptime scale_k = (K + NVFP4_SF_VECTOR_SIZE - 1) // NVFP4_SF_VECTOR_SIZE
    comptime warmup = 10
    comptime batches = 7

    # Rotate over enough weight copies that the total weight bytes swamp L2
    # (Blackwell L2 is tens of MB); 256 MB of packed weight guarantees each
    # reuse is a cold DRAM miss (the 4x-larger dense bf16 set is then ~1 GB).
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

    # ---- Activation / output device buffers (NOT rotated; common to all
    # paths and small at small M, so their L2 residency doesn't bias the
    # fused-vs-bf16 comparison). ----
    var act_dev = ctx.enqueue_create_buffer[DType.bfloat16](MAXM * K)
    var out_dev = ctx.enqueue_create_buffer[DType.bfloat16](MAXM * N)
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
        # Materialize this slot's dense bf16 weight from its packed/scale (the
        # values are irrelevant to timing; this just gives each slot plausible
        # bf16 bytes so the dense GEMM reads a distinct DRAM region).
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
    mlist.append(1)
    mlist.append(8)
    mlist.append(32)
    mlist.append(64)
    mlist.append(128)
    mlist.append(256)
    mlist.append(512)
    mlist.append(2048)

    print("== N =", N, "K =", K, "| L2-flush R =", R, "copies")
    for mi in range(len(mlist)):
        var m = mlist[mi]
        # More inner iters at small M (cheap, latency-bound) for a stable median.
        var inner = 100 if m <= 256 else 20

        var a_tt = TileTensor(
            act_dev.unsafe_ptr(), row_major(m, Idx[K])
        ).as_immut()
        var c_tt = TileTensor(out_dev.unsafe_ptr(), row_major(m, Idx[N]))

        @always_inline
        @parameter
        def _t_fused(c: DeviceContext, i: Int) raises:
            var s = i % R
            var pv = TileTensor(
                packed_slots[s].unsafe_ptr(), row_major(Idx[N], Idx[packed_k])
            ).as_immut()
            var sv = TileTensor(
                scale_slots[s].unsafe_ptr(), row_major(Idx[N], Idx[scale_k])
            ).as_immut()
            nvfp4_w4a16_fused_matmul_cuda(c_tt, a_tt, pv, sv, c)

        @always_inline
        @parameter
        def _t_mat(c: DeviceContext, i: Int) raises:
            var s = i % R
            var pv = TileTensor(
                packed_slots[s].unsafe_ptr(), row_major(Idx[N], Idx[packed_k])
            ).as_immut()
            var sv = TileTensor(
                scale_slots[s].unsafe_ptr(), row_major(Idx[N], Idx[scale_k])
            ).as_immut()
            nvfp4_w4a16_matmul_cuda(c_tt, a_tt, pv, sv, c)

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
            # Non-rotated: always slot 0, so the weight stays L2-resident.
            var wv = TileTensor(
                wbf16_slots[0].unsafe_ptr(), row_major(Idx[N], Idx[K])
            ).as_immut()
            _matmul_gpu[use_tensor_core=True, transpose_b=True](
                c_tt, a_tt, wv, c
            )

        # Warmup all paths (rotate to JIT/autotune every launch shape).
        for w in range(warmup):
            _t_fused(ctx, w)
            _t_mat(ctx, w)
            _t_bf16_flush(ctx, w)
            _t_bf16_res(ctx)
        ctx.synchronize()

        var s_fused = List[Float64]()
        var s_mat = List[Float64]()
        var s_bf16f = List[Float64]()
        var s_bf16r = List[Float64]()
        for _b in range(batches):
            s_fused.append(
                Float64(ctx.execution_time_iter[_t_fused](inner))
                / Float64(inner)
            )
            s_mat.append(
                Float64(ctx.execution_time_iter[_t_mat](inner)) / Float64(inner)
            )
            s_bf16f.append(
                Float64(ctx.execution_time_iter[_t_bf16_flush](inner))
                / Float64(inner)
            )
            s_bf16r.append(
                Float64(ctx.execution_time[_t_bf16_res](inner)) / Float64(inner)
            )

        var ns_fused = _median(s_fused)
        var ns_mat = _median(s_mat)
        var ns_bf16f = _median(s_bf16f)
        var ns_bf16r = _median(s_bf16r)

        # GFLOP/s = flops / seconds / 1e9 = (2*M*N*K) / ns_per_launch.
        var flops = 2.0 * Float64(m) * Float64(N) * Float64(K)
        print(
            "  M=",
            m,
            "| fused",
            ns_fused / 1000.0,
            "us",
            flops / ns_fused,
            "GF/s | mat",
            ns_mat / 1000.0,
            "us | bf16(flush)",
            ns_bf16f / 1000.0,
            "us",
            flops / ns_bf16f,
            "GF/s | bf16(resid)",
            ns_bf16r / 1000.0,
            "us | fused/bf16f",
            ns_fused / ns_bf16f,
            "| fused/mat",
            ns_fused / ns_mat,
        )

    _ = act_host^
    _ = packed_host^
    _ = scale_host^
    _ = act_dev^
    _ = out_dev^
    _ = packed_slots^
    _ = scale_slots^
    _ = wbf16_slots^


def main() raises:
    comptime if "metal" in _accelerator_arch():
        print("SKIP: NVIDIA GPU required")
        return
    var ctx = DeviceContext()
    print(
        "== bench_nvfp4_fused (warmup=10, batches=7, device-timed, L2-flushed)"
        " -- GPU:",
        ctx.name(),
    )
    # Representative transformer Linear (N, K): square, MLP up-proj, down-proj.
    _bench_nk[3072, 3072](ctx)
    _bench_nk[4096, 4096](ctx)
    _bench_nk[12288, 3072](ctx)
    _bench_nk[3072, 12288](ctx)
