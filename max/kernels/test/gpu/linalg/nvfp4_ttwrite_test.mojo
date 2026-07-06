# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# ===----------------------------------------------------------------------=== #
"""Write to a mutable fp8 TileTensor via .ptr (like the quant scale output),
then read those bytes back via a uint8 bitcast view (like the GEMM reads them)."""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from std.memory import bitcast
from layout import Idx, TileTensor, TensorLayout
from layout.tile_layout import row_major

comptime N = 128
comptime SVALS = SIMD[DType.float32, 4](0.5, 1.0, 1.5, 2.0)


def write_f8[
    l: TensorLayout
](t: TileTensor[DType.float8_e4m3fn, l, MutAnyOrigin], n: Int):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var p = t.ptr.address_space_cast[AddressSpace.GLOBAL]()
    p[i] = SVALS[i % 4].cast[DType.float8_e4m3fn]()


def read_bytes[
    l: TensorLayout
](
    t: TileTensor[DType.float8_e4m3fn, l, ImmutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    n: Int,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var p = t.ptr.address_space_cast[AddressSpace.GLOBAL]().bitcast[UInt8]()
    dst[i] = p[i]


def main() raises:
    with DeviceContext() as ctx:
        var d = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N)
        d.enqueue_fill(Scalar[DType.float8_e4m3fn](0))
        var tt = TileTensor(d.unsafe_ptr(), row_major(Idx[8], Idx[16]))
        ctx.enqueue_function[write_f8[type_of(tt).LayoutType]](
            tt, N, grid_dim=ceildiv(N, 128), block_dim=128
        )
        # Read back both as fp8 values AND as raw bytes (GEMM's bitcast view).
        var od = ctx.enqueue_create_buffer[DType.uint8](N)
        var tim = tt.as_immut()
        ctx.enqueue_function[read_bytes[type_of(tim).LayoutType]](
            tim, od.unsafe_ptr(), N, grid_dim=ceildiv(N, 128), block_dim=128
        )
        var hf = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](N)
        var hb = ctx.enqueue_create_host_buffer[DType.uint8](N)
        ctx.enqueue_copy(hf, d)
        ctx.enqueue_copy(hb, od)
        ctx.synchronize()
        var okv = True
        var okb = True
        for i in range(N):
            var exp = SVALS[i % 4].cast[DType.float8_e4m3fn]()
            if hf[i] != exp:
                okv = False
            # byte read must equal the fp8 byte
            if hb[i] != bitcast[DType.uint8, 1](SIMD[DType.float8_e4m3fn, 1](exp))[0]:
                okb = False
        print("fp8 write persists (value):", "OK" if okv else "WRONG")
        print("fp8 byte read via bitcast :", "OK" if okb else "WRONG")
        print("first 6 fp8 vals:", Float32(hf[0]), Float32(hf[1]),
              Float32(hf[2]), Float32(hf[3]), Float32(hf[4]), Float32(hf[5]))
