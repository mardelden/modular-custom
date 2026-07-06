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
"""FEASIBILITY SPIKE: one warp-level NVFP4 block-scaled tensor-core MMA on sm_120a.

Proves (go/no-go) that a single
  `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X
   .f32.e2m1.e2m1.f32.ue4m3`
instruction ASSEMBLES for sm_120a (RTX PRO 6000 Blackwell) via RAW inline PTX
(NOT the llvm.nvvm intrinsic -- see CUTLASS issue #3227) AND produces
numerically correct output vs a host reference.

Single warp (block_dim=32), one m16n8k64 tile. A(16x64 e2m1), B(64x8 e2m1),
per-16-element block scales SFA(16x4 ue4m3) / SFB(4x8 ue4m3), f32 accumulate.

Four escalating tests, chosen so assembly+execution can be proven even before
the fine fragment permutation is nailed:
  mode 0 (uniform):  A=B=1.0, SFA=SFB=1.0             -> D[m,n] = 64.0 (layout-robust)
  mode 1 (SFA only): A=B=1.0, SFB=1.0, SFA=[1,2,.5,4] -> D = 16*7.5 = 120
  mode 2 (SFB only): A=B=1.0, SFA=1.0, SFB=[1,2,.5,4] -> D = 16*7.5 = 120
  mode 3 (full):     distinct A,B,SFA,SFB per element -> full host reference
Modes 0-2 are independent of the per-element A/B permutation (uniform operands),
so they isolate: (0) does it run at all, (1) SFA byte/K-block order, (2) SFB
byte/K-block order. Mode 3 is the gold-standard full-layout check.
"""

from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from std.gpu.host.compile import _compile_code, get_gpu_target
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from std.sys.info import _accelerator_arch
from std.memory import bitcast


# E2M1 (fp4) nibble -> float value (matches linalg.fp4_utils.E2M1_TO_FLOAT32).
comptime E2M1 = SIMD[DType.float32, 16](
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)

# fp4 nibbles for a few exact values used to build structured inputs.
comptime NIB_HALF = 1     # 0.5
comptime NIB_ONE = 2      # 1.0
comptime NIB_ONEHALF = 3  # 1.5
comptime NIB_TWO = 4      # 2.0

# The instruction mnemonic. Form proven to assemble on sm_120a per CUTLASS
# issue #3227 (m16n8k64 first, then kind::mxf4nvf4).
comptime MMA_ASM = (
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
    ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}, "
    "$14, {0, 0}, $15, {0, 0};"
)


# ===----------------------------------------------------------------------=== #
# GPU kernel: one warp, one MMA tile.
# ===----------------------------------------------------------------------=== #
def nvfp4_mma_kernel(
    a_regs: UnsafePointer[UInt32, MutAnyOrigin],   # [32 lanes * 4]
    b_regs: UnsafePointer[UInt32, MutAnyOrigin],   # [32 lanes * 2]
    sa_regs: UnsafePointer[UInt32, MutAnyOrigin],  # [32 lanes]
    sb_regs: UnsafePointer[UInt32, MutAnyOrigin],  # [32 lanes]
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [16 * 8]
):
    var lane = Int(thread_idx.x)

    var a0 = a_regs[lane * 4 + 0]
    var a1 = a_regs[lane * 4 + 1]
    var a2 = a_regs[lane * 4 + 2]
    var a3 = a_regs[lane * 4 + 3]
    var b0 = b_regs[lane * 2 + 0]
    var b1 = b_regs[lane * 2 + 1]
    var sa = sa_regs[lane]
    var sb = sb_regs[lane]

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

    # C/D fragment layout (standard m16n8): group = lane>>2, tid = lane&3.
    #   d0 -> (group,     2*tid + 0)
    #   d1 -> (group,     2*tid + 1)
    #   d2 -> (group + 8, 2*tid + 0)
    #   d3 -> (group + 8, 2*tid + 1)
    var group = lane >> 2
    var tid = lane & 3
    out_ptr[(group) * 8 + 2 * tid + 0] = d0
    out_ptr[(group) * 8 + 2 * tid + 1] = d1
    out_ptr[(group + 8) * 8 + 2 * tid + 0] = d2
    out_ptr[(group + 8) * 8 + 2 * tid + 1] = d3


# ===----------------------------------------------------------------------=== #
# Logical (host) definitions of A, B, SFA, SFB for each test mode.
# ===----------------------------------------------------------------------=== #
def logic_a_nib(mode: Int, m: Int, k: Int) -> Int:
    if mode == 3:
        # Distinct, small, exactly-representable pattern.
        var sel = (m + k) % 4
        if sel == 0:
            return NIB_ONE
        elif sel == 1:
            return NIB_TWO
        elif sel == 2:
            return NIB_HALF
        else:
            return NIB_ONEHALF
    return NIB_ONE  # 1.0 for modes 0,1,2


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
    return NIB_ONE  # 1.0 for modes 0,1,2


def logic_sfa(mode: Int, m: Int, kb: Int) -> Float32:
    if mode == 1:
        var v = SIMD[DType.float32, 4](1.0, 2.0, 0.5, 4.0)
        return v[kb]
    if mode == 3:
        var v = SIMD[DType.float32, 4](1.0, 0.5, 2.0, 1.5)
        var f = Float32(1.0) if (m % 2 == 0) else Float32(0.5)
        return v[kb] * f
    return Float32(1.0)  # modes 0,2


def logic_sfb(mode: Int, kb: Int, n: Int) -> Float32:
    if mode == 2:
        var v = SIMD[DType.float32, 4](1.0, 2.0, 0.5, 4.0)
        return v[kb]
    if mode == 3:
        var v = SIMD[DType.float32, 4](2.0, 1.0, 1.5, 0.5)
        var f = Float32(1.0) if (n % 2 == 0) else Float32(2.0)
        return v[kb] * f
    return Float32(1.0)  # modes 0,1


# ===----------------------------------------------------------------------=== #
# Helpers: fp4 nibble packing and ue4m3 (fp8-e4m3) scale byte packing.
# ===----------------------------------------------------------------------=== #
def e4m3_byte(v: Float32) -> UInt32:
    """Round a float to fp8-e4m3 and return its 8 bits (ue4m3 scale byte)."""
    var f8 = SIMD[DType.float8_e4m3fn, 1](v.cast[DType.float8_e4m3fn]())
    var b = bitcast[DType.uint8, 1](f8)
    return UInt32(b[0])


def e4m3_roundtrip(v: Float32) -> Float32:
    """Value after e4m3 rounding (host reference must use the stored scale)."""
    return v.cast[DType.float8_e4m3fn]().cast[DType.float32]()


# ===----------------------------------------------------------------------=== #
# Host-side fragment builders (map logical A/B/SFA/SFB -> per-lane registers).
# ===----------------------------------------------------------------------=== #
def build_fragments(
    mode: Int,
    a_host: UnsafePointer[UInt32, MutAnyOrigin],
    b_host: UnsafePointer[UInt32, MutAnyOrigin],
    sa_host: UnsafePointer[UInt32, MutAnyOrigin],
    sb_host: UnsafePointer[UInt32, MutAnyOrigin],
):
    for lane in range(32):
        var group = lane >> 2  # 0..7
        var tid = lane & 3     # 0..3

        # --- A fragment (4 x u32). Each u32 = 8 fp4 nibbles along K.
        #   a0: row=group,     k in kgroup0 [tid*8 .. +7]
        #   a1: row=group+8,   k in kgroup0
        #   a2: row=group,     k in kgroup1 [32+tid*8 .. +7]
        #   a3: row=group+8,   k in kgroup1
        var a0 = UInt32(0)
        var a1 = UInt32(0)
        var a2 = UInt32(0)
        var a3 = UInt32(0)
        for j in range(8):
            var k0 = tid * 8 + j
            var k1 = 32 + tid * 8 + j
            var sh = UInt32(4 * j)
            a0 |= UInt32(logic_a_nib(mode, group, k0)) << sh
            a1 |= UInt32(logic_a_nib(mode, group + 8, k0)) << sh
            a2 |= UInt32(logic_a_nib(mode, group, k1)) << sh
            a3 |= UInt32(logic_a_nib(mode, group + 8, k1)) << sh
        a_host[lane * 4 + 0] = a0
        a_host[lane * 4 + 1] = a1
        a_host[lane * 4 + 2] = a2
        a_host[lane * 4 + 3] = a3

        # --- B fragment (2 x u32). col=group; K partitioned by tid.
        #   b0: k in kgroup0 [tid*8 .. +7], n=group
        #   b1: k in kgroup1 [32+tid*8 .. +7], n=group
        var b0 = UInt32(0)
        var b1 = UInt32(0)
        for j in range(8):
            var k0 = tid * 8 + j
            var k1 = 32 + tid * 8 + j
            var sh = UInt32(4 * j)
            b0 |= UInt32(logic_b_nib(mode, k0, group)) << sh
            b1 |= UInt32(logic_b_nib(mode, k1, group)) << sh
        b_host[lane * 2 + 0] = b0
        b_host[lane * 2 + 1] = b1

        # --- SFA (4 ue4m3 bytes = 4 K-block scales for the owned row).
        #   tid==0 -> row=group ; tid==1 -> row=group+8. tid 2/3 duplicate
        #   (HW ignores non-owners; identical for row-uniform modes).
        var sfa_row = group if (tid == 0 or tid == 2) else (group + 8)
        var sa = UInt32(0)
        for kb in range(4):
            sa |= e4m3_byte(logic_sfa(mode, sfa_row, kb)) << UInt32(8 * kb)
        sa_host[lane] = sa

        # --- SFB (4 ue4m3 bytes = 4 K-block scales for column=group).
        var sb = UInt32(0)
        for kb in range(4):
            sb |= e4m3_byte(logic_sfb(mode, kb, group)) << UInt32(8 * kb)
        sb_host[lane] = sb


def host_reference(mode: Int, m: Int, n: Int) -> Float32:
    """D[m,n] = sum_k E2M1[A]*SFA(m,k/16) * E2M1[B]*SFB(k/16,n)."""
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
# Test driver.
# ===----------------------------------------------------------------------=== #
def run_mode(ctx: DeviceContext, mode: Int, name: String) raises -> Bool:
    print("== mode", mode, "(", name, ")")

    var a_host = ctx.enqueue_create_host_buffer[DType.uint32](32 * 4)
    var b_host = ctx.enqueue_create_host_buffer[DType.uint32](32 * 2)
    var sa_host = ctx.enqueue_create_host_buffer[DType.uint32](32)
    var sb_host = ctx.enqueue_create_host_buffer[DType.uint32](32)
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](16 * 8)
    ctx.synchronize()

    build_fragments(
        mode,
        a_host.unsafe_ptr(),
        b_host.unsafe_ptr(),
        sa_host.unsafe_ptr(),
        sb_host.unsafe_ptr(),
    )

    var a_dev = ctx.enqueue_create_buffer[DType.uint32](32 * 4)
    var b_dev = ctx.enqueue_create_buffer[DType.uint32](32 * 2)
    var sa_dev = ctx.enqueue_create_buffer[DType.uint32](32)
    var sb_dev = ctx.enqueue_create_buffer[DType.uint32](32)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](16 * 8)
    out_dev.enqueue_fill(Float32(-1))
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(sa_dev, sa_host)
    ctx.enqueue_copy(sb_dev, sb_host)

    ctx.enqueue_function[nvfp4_mma_kernel](
        a_dev.unsafe_ptr(),
        b_dev.unsafe_ptr(),
        sa_dev.unsafe_ptr(),
        sb_dev.unsafe_ptr(),
        out_dev.unsafe_ptr(),
        grid_dim=1,
        block_dim=32,
    )

    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var pass_ = True
    var n_bad = 0
    for m in range(16):
        for n in range(8):
            var got = out_host[m * 8 + n]
            var exp = host_reference(mode, m, n)
            var tol = Float32(1e-3) + Float32(1e-3) * abs(exp)
            if abs(got - exp) > tol:
                if n_bad < 8:
                    print("  FAIL [", m, ",", n, "] got", got, "exp", exp)
                n_bad += 1
                pass_ = False

    # Output spread + a few scattered cells vs their references -- proves the
    # result is non-degenerate (not a constant that trivially matches).
    var lo = out_host[0]
    var hi = out_host[0]
    for i in range(16 * 8):
        lo = min(lo, out_host[i])
        hi = max(hi, out_host[i])
    print("  out range [", lo, ",", hi, "]")
    print(
        "  cells: (0,0)=", out_host[0], "/", host_reference(mode, 0, 0),
        " (3,5)=", out_host[3 * 8 + 5], "/", host_reference(mode, 3, 5),
        " (15,7)=", out_host[15 * 8 + 7], "/", host_reference(mode, 15, 7),
    )

    _ = a_dev^
    _ = b_dev^
    _ = sa_dev^
    _ = sb_dev^
    _ = out_dev^

    if pass_:
        print("  PASS")
    else:
        print("  FAILED (", n_bad, "of 128 mismatched )")
    return pass_


def main() raises:
    with DeviceContext() as ctx:
        print("NVFP4 sm_120 warp MMA spike")
        print("ACCEL_ARCH:", _accelerator_arch())
        # Dump the PTX so the report can show the exact instruction + target
        # arch that ptxas accepted (successful launch below proves acceptance).
        print("=== PTX BEGIN ===")
        print(_compile_code[nvfp4_mma_kernel]().asm)
        print("=== PTX END ===")
        var ok0 = run_mode(ctx, 0, "uniform A=B=1, SFA=SFB=1 -> 64")
        var ok1 = run_mode(ctx, 1, "SFA per-Kblock [1,2,.5,4] -> 120")
        var ok2 = run_mode(ctx, 2, "SFB per-Kblock [1,2,.5,4] -> 120")
        var ok3 = run_mode(ctx, 3, "full structured host reference")
        print("---")
        print(
            "mode0(run):", ok0,
            " mode1(SFA):", ok1,
            " mode2(SFB):", ok2,
            " mode3(full):", ok3,
        )
        if ok0 and ok1 and ok2 and ok3:
            print("ALL PASSED")
        else:
            print("PARTIAL/FAIL -- see per-mode results")
