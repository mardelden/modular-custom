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
"""D2/M1e (core): end-to-end W4A4 = quantize activation -> FP4xFP4 block-scaled GEMM.

Chains the M1d activation quantizer (bf16 A -> packed fp4 + fp8 block scales) into
the M1c GEMM (out = A_fp4 @ W_fp4^T, hardware applies block scales), on-device, and
verifies vs a host reference computed from the KERNEL's own quantized A + the given
packed-fp4 W. This is the numeric heart of the W4A4 path; the graph launcher/op is a
thin wrapper over these two kernels. Per-tensor input_scale is folded into the dynamic
per-block activation scale (so the epilogue fold is just x weight_scale_2, done later
at the graph level -- omitted here since we compare the raw block-scaled result).
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from std.memory import bitcast
from std.math import ceildiv


comptime M = 64
comptime N = 128
comptime K = 256
comptime KHALF = K // 2
comptime KB = K // 16
comptime NSTRIP = K // 64

comptime E2M1 = SIMD[DType.float32, 16](
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

comptime MMA_ASM = (
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
    ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}, "
    "$14, {0, 0}, $15, {0, 0};"
)


@always_inline
def ld_u32(p: UnsafePointer[UInt8, MutAnyOrigin], off: Int) -> UInt32:
    return (
        UInt32(p[off])
        | (UInt32(p[off + 1]) << 8)
        | (UInt32(p[off + 2]) << 16)
        | (UInt32(p[off + 3]) << 24)
    )


@always_inline
def round_e2m1_nibble(v: Float32) -> UInt8:
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


# ---- activation quantizer (input_scale folded into the dynamic block scale) ----
def quant_act_kernel(
    x: UnsafePointer[BFloat16, MutAnyOrigin],       # [M, K]
    out_packed: UnsafePointer[UInt8, MutAnyOrigin],  # [M, KHALF]
    out_scale: UnsafePointer[UInt8, MutAnyOrigin],   # [M, KB] fp8-e4m3
    m_dim: Int,
):
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= m_dim * KB:
        return
    var m = idx // KB
    var kb = idx % KB
    var base = m * K + kb * 16

    var amax = Float32(0)
    for j in range(16):
        amax = max(amax, abs(Float32(x[base + j])))
    var bs_f = amax / 6.0
    var bs_fp8 = SIMD[DType.float8_e4m3fn, 1](bs_f.cast[DType.float8_e4m3fn]())
    out_scale[m * KB + kb] = bitcast[DType.uint8, 1](bs_fp8)[0]
    var bs = Float32(bs_fp8[0])
    var inv = (Float32(1.0) / bs) if bs > 0.0 else Float32(0.0)

    for jb in range(8):
        var lo = round_e2m1_nibble(Float32(x[base + 2 * jb]) * inv)
        var hi = round_e2m1_nibble(Float32(x[base + 2 * jb + 1]) * inv)
        out_packed[m * KHALF + kb * 8 + jb] = lo | (hi << 4)


# ---- FP4xFP4 block-scaled GEMM (both operands fp4 + fp8 scales), bf16 out ----
def gemm_kernel(
    a_packed: UnsafePointer[UInt8, MutAnyOrigin],  # [M, KHALF]
    a_scale: UnsafePointer[UInt8, MutAnyOrigin],   # [M, KB]
    b_packed: UnsafePointer[UInt8, MutAnyOrigin],  # [N, KHALF]
    b_scale: UnsafePointer[UInt8, MutAnyOrigin],   # [N, KB]
    c: UnsafePointer[BFloat16, MutAnyOrigin],      # [M, N]
    m_dim: Int,
):
    var block_row = Int(block_idx.y) * 16
    var block_col = Int(block_idx.x) * 8
    var lane = Int(thread_idx.x)
    var group = lane >> 2
    var tid = lane & 3
    var afrow = block_row + (group if (tid == 0 or tid == 2) else group + 8)

    var ar0 = min(block_row + group, m_dim - 1)
    var ar1 = min(block_row + group + 8, m_dim - 1)
    var bc = block_col + group  # N is static, block always full in N here
    var sar = min(afrow, m_dim - 1)

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)

    for s in range(NSTRIP):
        var bo = s * 32
        var a0 = ld_u32(a_packed, ar0 * KHALF + bo + tid * 4)
        var a1 = ld_u32(a_packed, ar1 * KHALF + bo + tid * 4)
        var a2 = ld_u32(a_packed, ar0 * KHALF + bo + 16 + tid * 4)
        var a3 = ld_u32(a_packed, ar1 * KHALF + bo + 16 + tid * 4)
        var b0 = ld_u32(b_packed, bc * KHALF + bo + tid * 4)
        var b1 = ld_u32(b_packed, bc * KHALF + bo + 16 + tid * 4)
        var sa = ld_u32(a_scale, sar * KB + s * 4)
        var sb = ld_u32(b_scale, bc * KB + s * 4)
        var r = inlined_assembly[
            MMA_ASM,
            _RegisterPackType[Float32, Float32, Float32, Float32],
            constraints="=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r,r,r",
        ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sa, sb)
        c0 = r[0]
        c1 = r[1]
        c2 = r[2]
        c3 = r[3]

    var r0 = block_row + group
    var r1 = block_row + group + 8
    var col0 = block_col + 2 * tid
    var col1 = block_col + 2 * tid + 1
    if r0 < m_dim:
        c[r0 * N + col0] = c0.cast[DType.bfloat16]()
        c[r0 * N + col1] = c1.cast[DType.bfloat16]()
    if r1 < m_dim:
        c[r1 * N + col0] = c2.cast[DType.bfloat16]()
        c[r1 * N + col1] = c3.cast[DType.bfloat16]()


def main() raises:
    with DeviceContext() as ctx:
        print("D2/M1e core: W4A4 quantize+GEMM [", M, "x", N, "x", K, "]")

        var x_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
        var wpk_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KHALF)
        var wsc_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KB)
        var apk_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KHALF)
        var asc_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KB)
        var c_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
        ctx.synchronize()

        # Deterministic pseudo-random activation, weight nibbles, weight scales.
        for i in range(M * K):
            var h = (i * 1103515245 + 12345) % 2000
            x_h[i] = ((Float32(h) / 1000.0 - 1.0) * 3.0).cast[DType.bfloat16]()
        for i in range(N * KHALF):
            var h = (i * 22695477 + 1) % 256
            wpk_h[i] = UInt8(h)
        for i in range(N * KB):
            var h = (i * 69069 + 5) % 4  # scale in {0.5,1,1.5,2}
            var sv = (Float32(h) + 1.0) * 0.5
            var f8 = SIMD[DType.float8_e4m3fn, 1](sv.cast[DType.float8_e4m3fn]())
            wsc_h[i] = bitcast[DType.uint8, 1](f8)[0]

        var x_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
        var wpk_d = ctx.enqueue_create_buffer[DType.uint8](N * KHALF)
        var wsc_d = ctx.enqueue_create_buffer[DType.uint8](N * KB)
        var apk_d = ctx.enqueue_create_buffer[DType.uint8](M * KHALF)
        var asc_d = ctx.enqueue_create_buffer[DType.uint8](M * KB)
        var c_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
        ctx.enqueue_copy(x_d, x_h)
        ctx.enqueue_copy(wpk_d, wpk_h)
        ctx.enqueue_copy(wsc_d, wsc_h)

        # 1) quantize activation
        ctx.enqueue_function[quant_act_kernel](
            x_d.unsafe_ptr(), apk_d.unsafe_ptr(), asc_d.unsafe_ptr(), M,
            grid_dim=ceildiv(M * KB, 128), block_dim=128,
        )
        # 2) FP4xFP4 GEMM
        ctx.enqueue_function[gemm_kernel](
            apk_d.unsafe_ptr(), asc_d.unsafe_ptr(), wpk_d.unsafe_ptr(),
            wsc_d.unsafe_ptr(), c_d.unsafe_ptr(), M,
            grid_dim=(ceildiv(N, 8), ceildiv(M, 16)), block_dim=32,
        )
        ctx.enqueue_copy(apk_h, apk_d)
        ctx.enqueue_copy(asc_h, asc_d)
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()

        # Host reference from the KERNEL's quantized A + given W (block-scaled).
        var pass_ = True
        var n_bad = 0
        for m in range(M):
            for n in range(N):
                var acc = Float32(0)
                for k in range(K):
                    var kb = k // 16
                    var ab = apk_h[m * KHALF + k // 2]
                    var anib = Int((ab >> UInt8(4 * (k % 2))) & UInt8(0xF))
                    var af8 = bitcast[DType.float8_e4m3fn, 1](
                        SIMD[DType.uint8, 1](asc_h[m * KB + kb])
                    )
                    var av = E2M1[anib] * Float32(af8[0])
                    var wb = wpk_h[n * KHALF + k // 2]
                    var wnib = Int((wb >> UInt8(4 * (k % 2))) & UInt8(0xF))
                    var wf8 = bitcast[DType.float8_e4m3fn, 1](
                        SIMD[DType.uint8, 1](wsc_h[n * KB + kb])
                    )
                    var wv = E2M1[wnib] * Float32(wf8[0])
                    acc += av * wv
                var got = Float32(c_h[m * N + n])
                var tol = Float32(0.2) + Float32(6e-3) * abs(acc)
                if abs(got - acc) > tol:
                    if n_bad < 6:
                        print("  FAIL [", m, ",", n, "] got", got, "exp", acc)
                    n_bad += 1
                    pass_ = False
        if pass_:
            print("  PASS (", M * N, "cells; W4A4 quant+GEMM matches host ref )")
        else:
            print("  FAILED (", n_bad, "mismatched )")
