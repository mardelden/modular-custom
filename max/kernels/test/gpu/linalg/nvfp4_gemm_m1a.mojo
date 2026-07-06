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
"""D2/M1a: single m16n8k64 NVFP4 MMA tile, fragments loaded from packed GLOBAL
memory (natural row-major packed fp4 + ue4m3 scales) -- NOT host-built per-lane
registers like the spike. Proves the on-device fp4 fragment + scale load path,
the crux of the D2 GEMM.

Because fp4 packs 2 nibbles/byte, each A/B fragment register is exactly a
4-byte-aligned u32 slice of the packed row, and each scale register is the 4
ue4m3 K-block bytes of the owned row/col -- so the on-device load is just aligned
u32 reads at computed offsets. Reuses the validated atom + host reference from
`nvfp4_mma_spike.mojo`; if this matches the host reference for all modes, the load
mapping is correct.
"""

from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from std.memory import bitcast


comptime E2M1 = SIMD[DType.float32, 16](
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)
comptime NIB_HALF = 1
comptime NIB_ONE = 2
comptime NIB_ONEHALF = 3
comptime NIB_TWO = 4

comptime MMA_ASM = (
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
    ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}, "
    "$14, {0, 0}, $15, {0, 0};"
)


@always_inline
def ld_u32(p: UnsafePointer[UInt8, MutAnyOrigin], off: Int) -> UInt32:
    """Load a little-endian u32 from 4 bytes at `off` (offsets are 4B-aligned)."""
    return (
        UInt32(p[off])
        | (UInt32(p[off + 1]) << 8)
        | (UInt32(p[off + 2]) << 16)
        | (UInt32(p[off + 3]) << 24)
    )


# ===----------------------------------------------------------------------=== #
# GPU kernel: one warp, one MMA tile, fragments loaded from packed global mem.
# ===----------------------------------------------------------------------=== #
def nvfp4_tile_kernel(
    a_packed: UnsafePointer[UInt8, MutAnyOrigin],  # [16 rows * 32 bytes] (64 nib/row)
    b_packed: UnsafePointer[UInt8, MutAnyOrigin],  # [ 8 rows * 32 bytes]
    sfa: UnsafePointer[UInt8, MutAnyOrigin],       # [16 rows * 4 ue4m3]
    sfb: UnsafePointer[UInt8, MutAnyOrigin],       # [ 8 rows * 4 ue4m3]
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [16 * 8]
):
    var lane = Int(thread_idx.x)
    var group = lane >> 2  # 0..7
    var tid = lane & 3      # 0..3

    # A fragment (4 u32): rows group / group+8, k-groups [0:32) / [32:64).
    var a0 = ld_u32(a_packed, group * 32 + tid * 4)
    var a1 = ld_u32(a_packed, (group + 8) * 32 + tid * 4)
    var a2 = ld_u32(a_packed, group * 32 + 16 + tid * 4)
    var a3 = ld_u32(a_packed, (group + 8) * 32 + 16 + tid * 4)
    # B fragment (2 u32): col=group, k-groups [0:32) / [32:64).
    var b0 = ld_u32(b_packed, group * 32 + tid * 4)
    var b1 = ld_u32(b_packed, group * 32 + 16 + tid * 4)
    # SFA: row owned by tid 0/2 -> group, tid 1/3 -> group+8 (4 K-block bytes).
    var sfa_row = group if (tid == 0 or tid == 2) else (group + 8)
    var sa = ld_u32(sfa, sfa_row * 4)
    # SFB: col=group (4 K-block bytes).
    var sb = ld_u32(sfb, group * 4)

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)

    var r = inlined_assembly[
        MMA_ASM,
        _RegisterPackType[Float32, Float32, Float32, Float32],
        constraints="=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r,r,r",
    ](a0, a1, a2, a3, b0, b1, c0, c1, c2, c3, sa, sb)

    var d0 = r[0]
    var d1 = r[1]
    var d2 = r[2]
    var d3 = r[3]

    out_ptr[(group) * 8 + 2 * tid + 0] = d0
    out_ptr[(group) * 8 + 2 * tid + 1] = d1
    out_ptr[(group + 8) * 8 + 2 * tid + 0] = d2
    out_ptr[(group + 8) * 8 + 2 * tid + 1] = d3


# ===----------------------------------------------------------------------=== #
# Logical (host) definitions -- identical to nvfp4_mma_spike.mojo.
# ===----------------------------------------------------------------------=== #
def logic_a_nib(mode: Int, m: Int, k: Int) -> Int:
    if mode == 3:
        var sel = (m + k) % 4
        if sel == 0:
            return NIB_ONE
        elif sel == 1:
            return NIB_TWO
        elif sel == 2:
            return NIB_HALF
        else:
            return NIB_ONEHALF
    return NIB_ONE


def logic_b_nib(mode: Int, k: Int, n: Int) -> Int:
    if mode == 3:
        var sel = (2 * n + k) % 4
        if sel == 0:
            return NIB_TWO
        elif sel == 1:
            return NIB_ONE
        elif sel == 2:
            return NIB_ONEHALF
        else:
            return NIB_HALF
    return NIB_ONE


def logic_sfa(mode: Int, m: Int, kb: Int) -> Float32:
    if mode == 1:
        var v = SIMD[DType.float32, 4](1.0, 2.0, 0.5, 4.0)
        return v[kb]
    if mode == 3:
        var v = SIMD[DType.float32, 4](1.0, 0.5, 2.0, 1.5)
        var f = Float32(1.0) if (m % 2 == 0) else Float32(0.5)
        return v[kb] * f
    return Float32(1.0)


def logic_sfb(mode: Int, kb: Int, n: Int) -> Float32:
    if mode == 2:
        var v = SIMD[DType.float32, 4](1.0, 2.0, 0.5, 4.0)
        return v[kb]
    if mode == 3:
        var v = SIMD[DType.float32, 4](2.0, 1.0, 1.5, 0.5)
        var f = Float32(1.0) if (n % 2 == 0) else Float32(2.0)
        return v[kb] * f
    return Float32(1.0)


def e4m3_byte(v: Float32) -> UInt8:
    var f8 = SIMD[DType.float8_e4m3fn, 1](v.cast[DType.float8_e4m3fn]())
    return bitcast[DType.uint8, 1](f8)[0]


def e4m3_roundtrip(v: Float32) -> Float32:
    return v.cast[DType.float8_e4m3fn]().cast[DType.float32]()


def host_reference(mode: Int, m: Int, n: Int) -> Float32:
    var acc = Float32(0)
    for k in range(64):
        var kb = k // 16
        var av = E2M1[logic_a_nib(mode, m, k)] * e4m3_roundtrip(
            logic_sfa(mode, m, kb)
        )
        var bv = E2M1[logic_b_nib(mode, k, n)] * e4m3_roundtrip(
            logic_sfb(mode, kb, n)
        )
        acc += av * bv
    return acc


# ===----------------------------------------------------------------------=== #
# Host builds the NATURAL packed layout (what a real GEMM tile looks like).
# ===----------------------------------------------------------------------=== #
def build_natural(
    mode: Int,
    a_pk: UnsafePointer[UInt8, MutAnyOrigin],   # [16*32]
    b_pk: UnsafePointer[UInt8, MutAnyOrigin],   # [8*32]
    sfa_pk: UnsafePointer[UInt8, MutAnyOrigin],  # [16*4]
    sfb_pk: UnsafePointer[UInt8, MutAnyOrigin],  # [8*4]
):
    for m in range(16):
        for b in range(32):
            var lo = UInt8(logic_a_nib(mode, m, 2 * b))
            var hi = UInt8(logic_a_nib(mode, m, 2 * b + 1))
            a_pk[m * 32 + b] = lo | (hi << 4)
    for n in range(8):
        for b in range(32):
            var lo = UInt8(logic_b_nib(mode, 2 * b, n))
            var hi = UInt8(logic_b_nib(mode, 2 * b + 1, n))
            b_pk[n * 32 + b] = lo | (hi << 4)
    for m in range(16):
        for kb in range(4):
            sfa_pk[m * 4 + kb] = e4m3_byte(logic_sfa(mode, m, kb))
    for n in range(8):
        for kb in range(4):
            sfb_pk[n * 4 + kb] = e4m3_byte(logic_sfb(mode, kb, n))


def run_mode(ctx: DeviceContext, mode: Int, name: String) raises -> Bool:
    print("== mode", mode, "(", name, ")")

    var a_h = ctx.enqueue_create_host_buffer[DType.uint8](16 * 32)
    var b_h = ctx.enqueue_create_host_buffer[DType.uint8](8 * 32)
    var sfa_h = ctx.enqueue_create_host_buffer[DType.uint8](16 * 4)
    var sfb_h = ctx.enqueue_create_host_buffer[DType.uint8](8 * 4)
    var out_h = ctx.enqueue_create_host_buffer[DType.float32](16 * 8)
    ctx.synchronize()

    build_natural(
        mode,
        a_h.unsafe_ptr(),
        b_h.unsafe_ptr(),
        sfa_h.unsafe_ptr(),
        sfb_h.unsafe_ptr(),
    )

    var a_d = ctx.enqueue_create_buffer[DType.uint8](16 * 32)
    var b_d = ctx.enqueue_create_buffer[DType.uint8](8 * 32)
    var sfa_d = ctx.enqueue_create_buffer[DType.uint8](16 * 4)
    var sfb_d = ctx.enqueue_create_buffer[DType.uint8](8 * 4)
    var out_d = ctx.enqueue_create_buffer[DType.float32](16 * 8)
    out_d.enqueue_fill(Float32(-1))
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)
    ctx.enqueue_copy(sfa_d, sfa_h)
    ctx.enqueue_copy(sfb_d, sfb_h)

    ctx.enqueue_function[nvfp4_tile_kernel](
        a_d.unsafe_ptr(),
        b_d.unsafe_ptr(),
        sfa_d.unsafe_ptr(),
        sfb_d.unsafe_ptr(),
        out_d.unsafe_ptr(),
        grid_dim=1,
        block_dim=32,
    )

    ctx.enqueue_copy(out_h, out_d)
    ctx.synchronize()

    var pass_ = True
    var n_bad = 0
    for m in range(16):
        for n in range(8):
            var got = out_h[m * 8 + n]
            var exp = host_reference(mode, m, n)
            var tol = Float32(1e-3) + Float32(1e-3) * abs(exp)
            if abs(got - exp) > tol:
                if n_bad < 8:
                    print("  FAIL [", m, ",", n, "] got", got, "exp", exp)
                n_bad += 1
                pass_ = False

    var lo = out_h[0]
    var hi = out_h[0]
    for i in range(16 * 8):
        lo = min(lo, out_h[i])
        hi = max(hi, out_h[i])
    print("  out range [", lo, ",", hi, "]  (0,0)=", out_h[0], "/",
          host_reference(mode, 0, 0))

    _ = a_d^
    _ = b_d^
    _ = sfa_d^
    _ = sfb_d^
    _ = out_d^

    if pass_:
        print("  PASS")
    else:
        print("  FAILED (", n_bad, "of 128 )")
    return pass_


def main() raises:
    with DeviceContext() as ctx:
        print("D2/M1a: NVFP4 single-tile GEMM from packed global memory")
        var ok0 = run_mode(ctx, 0, "uniform -> 64")
        var ok1 = run_mode(ctx, 1, "SFA per-Kblock -> 120")
        var ok2 = run_mode(ctx, 2, "SFB per-Kblock -> 120")
        var ok3 = run_mode(ctx, 3, "full structured host reference")
        print("---")
        if ok0 and ok1 and ok2 and ok3:
            print("ALL PASSED")
        else:
            print("PARTIAL/FAIL")
