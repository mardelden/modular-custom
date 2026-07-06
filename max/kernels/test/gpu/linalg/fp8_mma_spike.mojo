# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
# ===----------------------------------------------------------------------=== #
"""FEASIBILITY SPIKE: one warp-level e4m3 (fp8) tensor-core MMA on sm_120a.

Goal (go/no-go for the fp8 flash-attention kernel): prove that the STANDARD
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` reached through the MAX
`std.gpu.compute.mma.mma` primitive (the same one `TensorCore`/`multistage_mma`
uses) both ASSEMBLES for sm_120a (RTX PRO 6000 Blackwell) AND computes
correctly. If yes, forking `mha_single_batch` with e4m3 operands needs no raw
PTX -- the abstraction drives the MMA.

Single warp (32 lanes), one m16n8k32 tile. A(16x32 e4m3) = 16 regs/lane,
B(32x8 e4m3) = 8 regs/lane, C/D(16x8 f32) = 4 regs/lane. Uniform operands make
the result independent of the per-lane fragment permutation, so the magnitude
alone verifies (a) the instruction runs and (b) all K=32 are summed:
  D[m,n] = sum_{k=0..31} A*B = 32 * A * B  (A,B exactly representable in e4m3).
Modes: (1,1)->32, (2,1)->64, (1,3)->96, (2,3)->192.
"""

from std.gpu import thread_idx
from std.gpu.host import DeviceContext
from std.gpu.host.compile import _compile_code
from std.gpu.compute.mma import mma
from std.sys.info import _accelerator_arch


def fp8_mma_kernel(
    a_val: Float32,
    b_val: Float32,
    out_ptr: UnsafePointer[Float32, MutAnyOrigin],  # [16 * 8]
):
    # Uniform fragments: every A/B element = a_val / b_val (exact in e4m3).
    var a = SIMD[DType.float8_e4m3fn, 16](a_val.cast[DType.float8_e4m3fn]())
    var b = SIMD[DType.float8_e4m3fn, 8](b_val.cast[DType.float8_e4m3fn]())
    var c = SIMD[DType.float32, 4](0)
    var d = c
    mma(d, a, b, c)

    # Standard m16n8 C/D fragment layout: group = lane>>2, tid = lane&3.
    var lane = Int(thread_idx.x)
    var group = lane >> 2
    var tid = lane & 3
    out_ptr[(group) * 8 + 2 * tid + 0] = d[0]
    out_ptr[(group) * 8 + 2 * tid + 1] = d[1]
    out_ptr[(group + 8) * 8 + 2 * tid + 0] = d[2]
    out_ptr[(group + 8) * 8 + 2 * tid + 1] = d[3]


def run_mode(ctx: DeviceContext, a_val: Float32, b_val: Float32) raises -> Bool:
    var exp = Float32(32) * a_val * b_val
    print("== A=", a_val, "B=", b_val, "-> expect", exp)

    var out_host = ctx.enqueue_create_host_buffer[DType.float32](16 * 8)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](16 * 8)
    out_dev.enqueue_fill(Float32(-1))
    ctx.synchronize()

    ctx.enqueue_function[fp8_mma_kernel](
        a_val, b_val, out_dev.unsafe_ptr(), grid_dim=1, block_dim=32
    )
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    var pass_ = True
    var n_bad = 0
    var lo = out_host[0]
    var hi = out_host[0]
    for i in range(16 * 8):
        var got = out_host[i]
        lo = min(lo, got)
        hi = max(hi, got)
        if abs(got - exp) > Float32(1e-3) + Float32(1e-3) * abs(exp):
            n_bad += 1
            pass_ = False
    print("  out range [", lo, ",", hi, "]  (0,0)=", out_host[0])

    _ = out_dev^
    if pass_:
        print("  PASS")
    else:
        print("  FAILED (", n_bad, "of 128 mismatched)")
    return pass_


def main() raises:
    with DeviceContext() as ctx:
        print("fp8 (e4m3) sm_120 warp MMA spike")
        print("ACCEL_ARCH:", _accelerator_arch())
        print("=== PTX BEGIN ===")
        print(_compile_code[fp8_mma_kernel]().asm)
        print("=== PTX END ===")
        var ok0 = run_mode(ctx, 1.0, 1.0)
        var ok1 = run_mode(ctx, 2.0, 1.0)
        var ok2 = run_mode(ctx, 1.0, 3.0)
        var ok3 = run_mode(ctx, 2.0, 3.0)
        print("---")
        if ok0 and ok1 and ok2 and ok3:
            print("ALL PASSED")
        else:
            print("PARTIAL/FAIL")
