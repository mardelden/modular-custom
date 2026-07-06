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
"""D2/M2 correctness: 3-way check of the W4A4 GEMMs against a host reference.

Both lib launchers -- `nvfp4_w4a4_matmul_cuda` (naive one-warp-per-tile) and
`nvfp4_w4a4_matmul_cuda_tiled` (SMEM-tiled) -- run on the same inputs and are
compared against a host-computed reference built from the DEVICE quant output
(e2e-test style: run `_quant_act_kernel` on test-owned buffers, copy the packed
fp4 + fp8 scales back, dot-product on host) so host/device fp8 rounding cannot
skew the reference. A host re-implementation of the quant recipe is also byte-
compared against the device quant as a diagnostic.

HARNESS LIFETIME RULE (the bug that produced two rounds of false failures, in
`nvfp4_w4a4_launch_test.mojo` and in earlier versions of this file): every
DeviceBuffer must be kept alive (`_ = buf^` at the END) past kernel EXECUTION,
not just past enqueue. A buffer whose last program-order use is
`TileTensor(buf.unsafe_ptr(), ...)` is ASAP-destroyed BEFORE the launcher
enqueues its kernels; the stream-ordered free precedes the kernels in the
stream, the allocator recycles the memory into the launchers' transient quant
buffers, and the quant kernel overwrites the still-unread inputs.
"""

from std.math import ceildiv
from std.gpu.host import DeviceContext
from layout import Idx, TileTensor
from layout.tile_layout import row_major

from linalg.matmul.gpu.nvfp4_w4a4_cuda import (
    _quant_act_kernel,
    nvfp4_w4a4_matmul_cuda,
    nvfp4_w4a4_matmul_cuda_tiled,
    nvfp4_w4a4_matmul_cuda_tiled_fusedq,
)

comptime E2M1 = SIMD[DType.float32, 16](
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


@always_inline
def _round_e2m1_nibble(v: Float32) -> UInt8:
    var a = abs(v)
    var sign = UInt8(8) if v < 0.0 else UInt8(0)
    var mag: UInt8
    if a > 5.0:
        mag = 7
    elif a >= 3.5:
        mag = 6
    elif a >= 2.5:
        mag = 5
    elif a >= 1.75:
        mag = 4
    elif a >= 1.25:
        mag = 3
    elif a >= 0.75:
        mag = 2
    elif a >= 0.25:
        mag = 1
    else:
        mag = 0
    return sign | mag


def _check[
    N: Int, K: Int
](ctx: DeviceContext, M: Int, s2_val: Float32 = 1.0) raises -> Bool:
    comptime KHALF = K // 2
    comptime KB = K // 16

    var x_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
    var wpk_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KHALF)
    var wsc_h = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](N * KB)
    var apk_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KHALF)
    var asc_h = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](M * KB)
    var ref_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    var tst_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    ctx.synchronize()

    for i in range(M * K):
        var h = (i * 1103515245 + 12345) % 2000
        x_h[i] = ((Float32(h) / 1000.0 - 1.0) * 3.0).cast[DType.bfloat16]()
    for i in range(N * KHALF):
        var h = (i * 22695477 + 1) % 256
        wpk_h[i] = UInt8(h)
    for i in range(N * KB):
        var h = (i * 69069 + 5) % 4
        var sv = (Float32(h) + 1.0) * 0.5
        wsc_h[i] = sv.cast[DType.float8_e4m3fn]()

    var x_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
    var wpk_d = ctx.enqueue_create_buffer[DType.uint8](N * KHALF)
    var wsc_d = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N * KB)
    var apk_d = ctx.enqueue_create_buffer[DType.uint8](M * KHALF)
    var asc_d = ctx.enqueue_create_buffer[DType.float8_e4m3fn](M * KB)
    var ref_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    var tst_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    # Per-tensor weight_scale_2, folded in the GEMM epilogue.
    var s2_d = ctx.enqueue_create_buffer[DType.float32](1)
    s2_d.enqueue_fill(s2_val)
    ctx.enqueue_copy(x_d, x_h)
    ctx.enqueue_copy(wpk_d, wpk_h)
    ctx.enqueue_copy(wsc_d, wsc_h)

    var a_tt = TileTensor(x_d.unsafe_ptr(), row_major(M, Idx[K]))
    var bp_tt = TileTensor(wpk_d.unsafe_ptr(), row_major(Idx[N], Idx[KHALF]))
    var bs_tt = TileTensor(wsc_d.unsafe_ptr(), row_major(Idx[N], Idx[KB]))
    var ref_tt = TileTensor(ref_d.unsafe_ptr(), row_major(M, Idx[N]))
    var tst_tt = TileTensor(tst_d.unsafe_ptr(), row_major(M, Idx[N]))
    var s2_tt = TileTensor(s2_d.unsafe_ptr(), row_major(Idx[1], Idx[1]))

    # ---- Device quant on TEST-owned buffers (the reference's A operand). ----
    var a_im = a_tt.as_immut()
    var apk_tt = TileTensor(apk_d.unsafe_ptr(), row_major(M, Idx[KHALF]))
    var asc_tt = TileTensor(asc_d.unsafe_ptr(), row_major(M, Idx[KB]))
    ctx.enqueue_function[
        _quant_act_kernel[
            K,
            type_of(a_im).LayoutType,
            type_of(apk_tt).LayoutType,
            type_of(asc_tt).LayoutType,
        ]
    ](
        a_im, apk_tt, asc_tt, M,
        grid_dim=ceildiv(M * KB, 128), block_dim=128,
    )
    ctx.enqueue_copy(apk_h, apk_d)
    ctx.enqueue_copy(asc_h, asc_d)

    # ---- All three GEMM paths under test (naive, tiled, fused-quant). ----
    var fq_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
    var fq_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
    var fq_tt = TileTensor(fq_d.unsafe_ptr(), row_major(M, Idx[N]))
    nvfp4_w4a4_matmul_cuda(ref_tt, a_tt, bp_tt, bs_tt, s2_tt, ctx)
    nvfp4_w4a4_matmul_cuda_tiled(tst_tt, a_tt, bp_tt, bs_tt, s2_tt, ctx)
    nvfp4_w4a4_matmul_cuda_tiled_fusedq(fq_tt, a_tt, bp_tt, bs_tt, s2_tt, ctx)
    ctx.enqueue_copy(ref_h, ref_d)
    ctx.enqueue_copy(tst_h, tst_d)
    ctx.enqueue_copy(fq_h, fq_d)
    ctx.synchronize()

    # ---- Diagnostic: host quant recipe vs device quant (byte compare). ----
    var q_nib_diff = 0
    var q_sc_diff = 0
    for m in range(M):
        for kb in range(KB):
            var base = m * K + kb * 16
            var amax = Float32(0)
            for j in range(16):
                amax = max(amax, abs(Float32(x_h[base + j])))
            var bs_fp8 = (amax / 6.0).cast[DType.float8_e4m3fn]()
            if Float32(bs_fp8) != Float32(asc_h[m * KB + kb]):
                q_sc_diff += 1
            var bs = Float32(asc_h[m * KB + kb])  # use DEVICE scale for nibbles
            var inv = (Float32(1.0) / bs) if bs > 0.0 else Float32(0.0)
            for jb in range(8):
                var lo = _round_e2m1_nibble(Float32(x_h[base + 2 * jb]) * inv)
                var hi = _round_e2m1_nibble(
                    Float32(x_h[base + 2 * jb + 1]) * inv
                )
                if (lo | (hi << 4)) != apk_h[m * KHALF + kb * 8 + jb]:
                    q_nib_diff += 1

    # ---- 3-way compare vs host ref built from DEVICE quant bytes. ----
    var naive_bad = 0
    var tiled_bad = 0
    var fq_bad = 0
    var naive_maxd = Float32(0)
    var tiled_maxd = Float32(0)
    var nt_maxd = Float32(0)
    var fq_maxd = Float32(0)
    var shown = 0
    for m in range(M):
        for n in range(N):
            var host = Float32(0)
            for k in range(K):
                var kb2 = k // 16
                var ab = apk_h[m * KHALF + k // 2]
                var anib = Int((ab >> UInt8(4 * (k % 2))) & UInt8(0xF))
                var av = E2M1[anib] * Float32(asc_h[m * KB + kb2])
                var wb = wpk_h[n * KHALF + k // 2]
                var wnib = Int((wb >> UInt8(4 * (k % 2))) & UInt8(0xF))
                var wv = E2M1[wnib] * Float32(wsc_h[n * KB + kb2])
                host += av * wv
            # Kernels fold weight_scale_2 in the epilogue; mirror it here
            # (tolerance absorbs the kernels' double rounding).
            host *= s2_val
            var tol = Float32(0.5) + Float32(8e-3) * abs(host)
            var nv = Float32(ref_h[m * N + n])
            var tv = Float32(tst_h[m * N + n])
            var fv = Float32(fq_h[m * N + n])
            var nd = abs(nv - host)
            var td = abs(tv - host)
            naive_maxd = max(naive_maxd, nd)
            tiled_maxd = max(tiled_maxd, td)
            nt_maxd = max(nt_maxd, abs(nv - tv))
            # Fused-prologue-quant must be BIT-identical to the tiled path.
            fq_maxd = max(fq_maxd, abs(fv - tv))
            if fv != tv:
                fq_bad += 1
            if nd > tol:
                naive_bad += 1
            if td > tol:
                tiled_bad += 1
            if (nd > tol or td > tol) and shown < 4:
                print(
                    "  CELL [", m, ",", n, "] host", host,
                    "naive", nv, "tiled", tv,
                )
                shown += 1
    print(
        "  SHAPE M=", M, "N=", N, "K=", K,
        "| quant diffs: sc=", q_sc_diff, "nib=", q_nib_diff,
        "| naive: bad=", naive_bad, "maxd=", naive_maxd,
        "| tiled: bad=", tiled_bad, "maxd=", tiled_maxd,
        "| naive-vs-tiled maxd=", nt_maxd,
        "| fusedq-vs-tiled: bad=", fq_bad, "maxd=", fq_maxd,
    )

    # Keep every DeviceBuffer alive past kernel EXECUTION (see module docstring).
    _ = x_d^
    _ = wpk_d^
    _ = wsc_d^
    _ = apk_d^
    _ = asc_d^
    _ = ref_d^
    _ = tst_d^
    _ = fq_d^
    _ = s2_d^
    return naive_bad == 0 and tiled_bad == 0 and fq_bad == 0


def main() raises:
    with DeviceContext() as ctx:
        print("D2/M2: naive + tiled W4A4 GEMM, 3-way vs device-quant host ref")
        var ok = True
        ok = _check[128, 128](ctx, 64) and ok      # small path (M<256), N,K%128
        ok = _check[256, 256](ctx, 130) and ok     # small path, M edge, 2 K-tiles
        ok = _check[256, 256](ctx, 256) and ok     # BIG path (M>=256), 2x2 blocks
        ok = _check[384, 384](ctx, 300) and ok     # BIG path, M edge, N/K=3 tiles
        ok = _check[192, 128](ctx, 300) and ok     # FALLBACK (N%128!=0) -> 64x64
        # weight_scale_2 epilogue fold actually applied (both paths, both tiles).
        ok = _check[256, 256](ctx, 256, s2_val=0.5) and ok
        ok = _check[128, 128](ctx, 64, s2_val=2.0) and ok
        if ok:
            print("ALL PASS")
        else:
            raise Error("W4A4 GEMM mismatch vs host reference")
