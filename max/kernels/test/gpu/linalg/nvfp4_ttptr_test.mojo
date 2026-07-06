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
"""Minimal: does flat-indexing a TileTensor's .ptr read the buffer correctly?
The GEMM launcher bug shows identical-data rows reading differently; isolate the
one suspect op: TileTensor(row_major) -> .ptr.address_space_cast[GLOBAL] -> [i].
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.math import ceildiv
from layout import Idx, TileTensor, TensorLayout
from layout.tile_layout import row_major


comptime NROW = 8
comptime NCOL = 32  # bytes/row (like KHALF)


def read_kernel[
    l: TensorLayout
](
    t: TileTensor[DType.uint8, l, ImmutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
    n: Int,
):
    var i = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if i >= n:
        return
    var p = t.ptr.address_space_cast[AddressSpace.GLOBAL]()
    dst[i] = p[i]


def read_row_kernel[
    l: TensorLayout
](
    t: TileTensor[DType.uint8, l, ImmutAnyOrigin],
    dst: UnsafePointer[UInt8, MutAnyOrigin],
):
    # One thread per row: read byte 0 of each row via flat index row*NCOL.
    var r = Int(thread_idx.x)
    if r >= NROW:
        return
    var p = t.ptr.address_space_cast[AddressSpace.GLOBAL]()
    dst[r] = p[r * NCOL]


def main() raises:
    with DeviceContext() as ctx:
        comptime N = NROW * NCOL
        var h = ctx.enqueue_create_host_buffer[DType.uint8](N)
        var oh = ctx.enqueue_create_host_buffer[DType.uint8](N)
        var rh = ctx.enqueue_create_host_buffer[DType.uint8](NROW)
        ctx.synchronize()
        # Rows 0 and 2 identical (period-2), like the failing GEMM weights.
        for i in range(N):
            var row = i // NCOL
            var col = i % NCOL
            h[i] = UInt8((col * 40503 + 7) % 256)  # same for every row

        var d = ctx.enqueue_create_buffer[DType.uint8](N)
        var od = ctx.enqueue_create_buffer[DType.uint8](N)
        var rd = ctx.enqueue_create_buffer[DType.uint8](NROW)
        ctx.enqueue_copy(d, h)
        var tt = TileTensor(d.unsafe_ptr(), row_major(Idx[NROW], Idx[NCOL]))
        var tim = tt.as_immut()

        ctx.enqueue_function[read_kernel[type_of(tim).LayoutType]](
            tim, od.unsafe_ptr(), N,
            grid_dim=ceildiv(N, 128), block_dim=128,
        )
        ctx.enqueue_function[read_row_kernel[type_of(tim).LayoutType]](
            tim, rd.unsafe_ptr(), grid_dim=1, block_dim=NROW,
        )
        ctx.enqueue_copy(oh, od)
        ctx.enqueue_copy(rh, rd)
        ctx.synchronize()

        var ok = True
        for i in range(N):
            if oh[i] != h[i]:
                if i < 8:
                    print("  flat mismatch at", i, "got", oh[i], "exp", h[i])
                ok = False
        print("flat read of TileTensor.ptr:", "OK" if ok else "WRONG")
        print("row byte0 via row*NCOL:  ",
              rh[0], rh[1], rh[2], rh[3], "(rows 0,2 should match)")
