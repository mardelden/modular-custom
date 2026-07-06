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
"""D2/M1c: full [M,N,K] NVFP4 block-scaled GEMM (out = A @ B^T, block scales).

One warp per [16,8] output tile; grid = (ceil(N/8), ceil(M/16)); K-loop over
m16n8k64 strips. OOB reads are index-clamped (safe) and OOB outputs are simply
not stored, so arbitrary M / N work (token counts aren't 16-aligned). f32 output
(the per-tensor input_scale * weight_scale_2 fold is added at the launcher in
M1e). Verified vs a full W4A4 host reference at an aligned and a ragged size.
"""

from std.gpu import thread_idx, block_idx
from std.gpu.host import DeviceContext
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from std.memory import bitcast
from std.math import ceildiv


comptime E2M1 = SIMD[DType.float32, 16](
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
comptime NIB_HALF = 1
comptime NIB_ONE = 2
comptime NIB_ONEHALF = 3
comptime NIB_TWO = 4

comptime K = 256
comptime KBYTES = K // 2
comptime KB = K // 16
comptime NSTRIP = K // 64

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


def nvfp4_gemm_kernel(
    a_packed: UnsafePointer[UInt8, MutAnyOrigin],  # [M, KBYTES]
    b_packed: UnsafePointer[UInt8, MutAnyOrigin],  # [N, KBYTES]
    sfa: UnsafePointer[UInt8, MutAnyOrigin],       # [M, KB]
    sfb: UnsafePointer[UInt8, MutAnyOrigin],       # [N, KB]
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [M, N]
    m_dim: Int,
    n_dim: Int,
):
    var block_row = Int(block_idx.y) * 16
    var block_col = Int(block_idx.x) * 8
    var lane = Int(thread_idx.x)
    var group = lane >> 2
    var tid = lane & 3

    # Global rows/cols this lane touches; clamp for safe OOB reads.
    var arow0 = block_row + group
    var arow1 = block_row + group + 8
    var bcol = block_col + group
    var sfa_grow = block_row + (group if (tid == 0 or tid == 2) else group + 8)
    var ar0 = min(arow0, m_dim - 1)
    var ar1 = min(arow1, m_dim - 1)
    var bc = min(bcol, n_dim - 1)
    var sar = min(sfa_grow, m_dim - 1)

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)

    for s in range(NSTRIP):
        var bo = s * 32
        var a0 = ld_u32(a_packed, ar0 * KBYTES + bo + tid * 4)
        var a1 = ld_u32(a_packed, ar1 * KBYTES + bo + tid * 4)
        var a2 = ld_u32(a_packed, ar0 * KBYTES + bo + 16 + tid * 4)
        var a3 = ld_u32(a_packed, ar1 * KBYTES + bo + 16 + tid * 4)
        var b0 = ld_u32(b_packed, bc * KBYTES + bo + tid * 4)
        var b1 = ld_u32(b_packed, bc * KBYTES + bo + 16 + tid * 4)
        var sa = ld_u32(sfa, sar * KB + s * 4)
        var sb = ld_u32(sfb, bc * KB + s * 4)

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
        if col0 < n_dim:
            out_ptr[r0 * n_dim + col0] = c0
        if col1 < n_dim:
            out_ptr[r0 * n_dim + col1] = c1
    if r1 < m_dim:
        if col0 < n_dim:
            out_ptr[r1 * n_dim + col0] = c2
        if col1 < n_dim:
            out_ptr[r1 * n_dim + col1] = c3


# ---- logical inputs (patterns repeat mod 4, valid for any m/n/k) ----
def logic_a_nib(m: Int, k: Int) -> Int:
    var sel = (m + k) % 4
    if sel == 0:
        return NIB_ONE
    elif sel == 1:
        return NIB_TWO
    elif sel == 2:
        return NIB_HALF
    else:
        return NIB_ONEHALF


def logic_b_nib(k: Int, n: Int) -> Int:
    var sel = (2 * n + k) % 4
    if sel == 0:
        return NIB_TWO
    elif sel == 1:
        return NIB_ONE
    elif sel == 2:
        return NIB_ONEHALF
    else:
        return NIB_HALF


def logic_sfa(m: Int, kb: Int) -> Float32:
    var v = SIMD[DType.float32, 4](1.0, 0.5, 2.0, 1.5)
    var f = Float32(1.0) if (m % 2 == 0) else Float32(0.5)
    return v[kb % 4] * f


def logic_sfb(kb: Int, n: Int) -> Float32:
    var v = SIMD[DType.float32, 4](2.0, 1.0, 1.5, 0.5)
    var f = Float32(1.0) if (n % 2 == 0) else Float32(2.0)
    return v[kb % 4] * f


def e4m3_byte(v: Float32) -> UInt8:
    var f8 = SIMD[DType.float8_e4m3fn, 1](v.cast[DType.float8_e4m3fn]())
    return bitcast[DType.uint8, 1](f8)[0]


def e4m3_roundtrip(v: Float32) -> Float32:
    return v.cast[DType.float8_e4m3fn]().cast[DType.float32]()


def host_reference(m: Int, n: Int) -> Float32:
    var acc = Float32(0)
    for k in range(K):
        var kb = k // 16
        var av = E2M1[logic_a_nib(m, k)] * e4m3_roundtrip(logic_sfa(m, kb))
        var bv = E2M1[logic_b_nib(k, n)] * e4m3_roundtrip(logic_sfb(kb, n))
        acc += av * bv
    return acc


def build_natural(
    m_dim: Int, n_dim: Int,
    a_pk: UnsafePointer[UInt8, MutAnyOrigin],
    b_pk: UnsafePointer[UInt8, MutAnyOrigin],
    sfa_pk: UnsafePointer[UInt8, MutAnyOrigin],
    sfb_pk: UnsafePointer[UInt8, MutAnyOrigin],
):
    for m in range(m_dim):
        for b in range(KBYTES):
            var lo = UInt8(logic_a_nib(m, 2 * b))
            var hi = UInt8(logic_a_nib(m, 2 * b + 1))
            a_pk[m * KBYTES + b] = lo | (hi << 4)
        for kb in range(KB):
            sfa_pk[m * KB + kb] = e4m3_byte(logic_sfa(m, kb))
    for n in range(n_dim):
        for b in range(KBYTES):
            var lo = UInt8(logic_b_nib(2 * b, n))
            var hi = UInt8(logic_b_nib(2 * b + 1, n))
            b_pk[n * KBYTES + b] = lo | (hi << 4)
        for kb in range(KB):
            sfb_pk[n * KB + kb] = e4m3_byte(logic_sfb(kb, n))


def run_size(ctx: DeviceContext, m_dim: Int, n_dim: Int) raises -> Bool:
    print("== GEMM [", m_dim, "x", n_dim, "x", K, "]")

    var a_h = ctx.enqueue_create_host_buffer[DType.uint8](m_dim * KBYTES)
    var b_h = ctx.enqueue_create_host_buffer[DType.uint8](n_dim * KBYTES)
    var sfa_h = ctx.enqueue_create_host_buffer[DType.uint8](m_dim * KB)
    var sfb_h = ctx.enqueue_create_host_buffer[DType.uint8](n_dim * KB)
    var out_h = ctx.enqueue_create_host_buffer[DType.float32](m_dim * n_dim)
    ctx.synchronize()

    build_natural(
        m_dim, n_dim, a_h.unsafe_ptr(), b_h.unsafe_ptr(), sfa_h.unsafe_ptr(),
        sfb_h.unsafe_ptr(),
    )

    var a_d = ctx.enqueue_create_buffer[DType.uint8](m_dim * KBYTES)
    var b_d = ctx.enqueue_create_buffer[DType.uint8](n_dim * KBYTES)
    var sfa_d = ctx.enqueue_create_buffer[DType.uint8](m_dim * KB)
    var sfb_d = ctx.enqueue_create_buffer[DType.uint8](n_dim * KB)
    var out_d = ctx.enqueue_create_buffer[DType.float32](m_dim * n_dim)
    out_d.enqueue_fill(Float32(-1))
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)

    ctx.enqueue_function[nvfp4_gemm_kernel](
        a_d.unsafe_ptr(), b_d.unsafe_ptr(), sfa_d.unsafe_ptr(),
        sfb_d.unsafe_ptr(), out_d.unsafe_ptr(), m_dim, n_dim,
        grid_dim=(ceildiv(n_dim, 8), ceildiv(m_dim, 16)),
        block_dim=32,
    )

    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    var pass_ = True
    var n_bad = 0
    for m in range(m_dim):
        for n in range(n_dim):
            var got = out_h[m * n_dim + n]
            var exp = host_reference(m, n)
            var tol = Float32(1e-2) + Float32(2e-3) * abs(exp)
            if abs(got - exp) > tol:
                if n_bad < 6:
                    print("  FAIL [", m, ",", n, "] got", got, "exp", exp)
                n_bad += 1
                pass_ = False

    _ = a_d^
    _ = b_d^
    _ = sfa_d^
    _ = sfb_d^
    _ = out_d^

    if pass_:
        print("  PASS (", m_dim * n_dim, "cells )")
    else:
        print("  FAILED (", n_bad, "mismatched )")
    return pass_


def main() raises:
    with DeviceContext() as ctx:
        print("D2/M1c: NVFP4 [M,N,K] block-scaled GEMM")
        var ok0 = run_size(ctx, 48, 40)   # aligned (M%16=0, N%8=0) -> grid (5,3)
        var ok1 = run_size(ctx, 50, 44)   # ragged M and N -> edge clamp/guard
        var ok2 = run_size(ctx, 129, 33)  # ragged, bigger
        print("---")
        if ok0 and ok1 and ok2:
            print("ALL PASSED")
        else:
            print("PARTIAL/FAIL")
