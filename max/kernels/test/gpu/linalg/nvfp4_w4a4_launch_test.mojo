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
"""D2/M1e: W4A4 launcher over TileTensor kernels (fused idiom: kernels take
TileTensors, derive the raw pointer inside via `.ptr.address_space_cast[GLOBAL]`).
Standalone-tested before moving to the kernel lib + graph op.
"""

from std.gpu import thread_idx, block_idx, block_dim
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from std.memory import bitcast
from std.math import ceildiv
from layout import Idx, TileTensor, TensorLayout
from layout.tile_layout import row_major


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


def quant_act_kernel[
    K: Int, x_l: TensorLayout, pk_l: TensorLayout, sc_l: TensorLayout
](
    x: TileTensor[DType.bfloat16, x_l, ImmutAnyOrigin],
    out_packed: TileTensor[DType.uint8, pk_l, MutAnyOrigin],
    out_scale: TileTensor[DType.float8_e4m3fn, sc_l, MutAnyOrigin],
    m_dim: Int,
):
    comptime KHALF = K // 2
    comptime KB = K // 16
    var idx = Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    if idx >= m_dim * KB:
        return
    var m = idx // KB
    var kb = idx % KB
    var base = m * K + kb * 16
    var xp = x.ptr.address_space_cast[AddressSpace.GLOBAL]()
    var pkp = out_packed.ptr.address_space_cast[AddressSpace.GLOBAL]()
    var scp = out_scale.ptr.address_space_cast[AddressSpace.GLOBAL]()

    var amax = Float32(0)
    for j in range(16):
        amax = max(amax, abs(Float32(xp[base + j])))
    var bs_fp8 = SIMD[DType.float8_e4m3fn, 1](
        (amax / 6.0).cast[DType.float8_e4m3fn]()
    )
    scp[m * KB + kb] = bs_fp8[0]
    var bs = Float32(bs_fp8[0])
    var inv = (Float32(1.0) / bs) if bs > 0.0 else Float32(0.0)
    for jb in range(8):
        var lo = round_e2m1_nibble(Float32(xp[base + 2 * jb]) * inv)
        var hi = round_e2m1_nibble(Float32(xp[base + 2 * jb + 1]) * inv)
        pkp[m * KHALF + kb * 8 + jb] = lo | (hi << 4)


def gemm_kernel[
    N: Int, K: Int, ap_l: TensorLayout, asc_l: TensorLayout,
    bp_l: TensorLayout, bsc_l: TensorLayout, c_t: DType, c_l: TensorLayout
](
    a_packed: TileTensor[DType.uint8, ap_l, ImmutAnyOrigin],
    a_scale: TileTensor[DType.float8_e4m3fn, asc_l, ImmutAnyOrigin],
    b_packed: TileTensor[DType.uint8, bp_l, ImmutAnyOrigin],
    b_scale: TileTensor[DType.float8_e4m3fn, bsc_l, ImmutAnyOrigin],
    c: TileTensor[c_t, c_l, MutAnyOrigin],
    m_dim: Int,
):
    comptime KHALF = K // 2
    comptime KB = K // 16
    comptime NSTRIP = K // 64
    var apb = a_packed.ptr.address_space_cast[AddressSpace.GLOBAL]()
    var bpb = b_packed.ptr.address_space_cast[AddressSpace.GLOBAL]()
    var sapb = a_scale.ptr.address_space_cast[AddressSpace.GLOBAL]().bitcast[
        UInt8
    ]()
    var sbpb = b_scale.ptr.address_space_cast[AddressSpace.GLOBAL]().bitcast[
        UInt8
    ]()
    var cp = c.ptr.address_space_cast[AddressSpace.GLOBAL]()

    @always_inline
    @parameter
    def u32a(o: Int) -> UInt32:
        return (
            UInt32(apb[o]) | (UInt32(apb[o + 1]) << 8)
            | (UInt32(apb[o + 2]) << 16) | (UInt32(apb[o + 3]) << 24)
        )

    @always_inline
    @parameter
    def u32b(o: Int) -> UInt32:
        return (
            UInt32(bpb[o]) | (UInt32(bpb[o + 1]) << 8)
            | (UInt32(bpb[o + 2]) << 16) | (UInt32(bpb[o + 3]) << 24)
        )

    @always_inline
    @parameter
    def u32sa(o: Int) -> UInt32:
        return (
            UInt32(sapb[o]) | (UInt32(sapb[o + 1]) << 8)
            | (UInt32(sapb[o + 2]) << 16) | (UInt32(sapb[o + 3]) << 24)
        )

    @always_inline
    @parameter
    def u32sb(o: Int) -> UInt32:
        return (
            UInt32(sbpb[o]) | (UInt32(sbpb[o + 1]) << 8)
            | (UInt32(sbpb[o + 2]) << 16) | (UInt32(sbpb[o + 3]) << 24)
        )

    var block_row = Int(block_idx.y) * 16
    var block_col = Int(block_idx.x) * 8
    var lane = Int(thread_idx.x)
    var group = lane >> 2
    var tid = lane & 3
    var afrow = block_row + (group if (tid == 0 or tid == 2) else group + 8)
    var ar0 = min(block_row + group, m_dim - 1)
    var ar1 = min(block_row + group + 8, m_dim - 1)
    var bc = block_col + group
    var sar = min(afrow, m_dim - 1)

    var c0 = Float32(0)
    var c1 = Float32(0)
    var c2 = Float32(0)
    var c3 = Float32(0)
    for s in range(NSTRIP):
        var bo = s * 32
        var a0 = u32a(ar0 * KHALF + bo + tid * 4)
        var a1 = u32a(ar1 * KHALF + bo + tid * 4)
        var a2 = u32a(ar0 * KHALF + bo + 16 + tid * 4)
        var a3 = u32a(ar1 * KHALF + bo + 16 + tid * 4)
        var b0 = u32b(bc * KHALF + bo + tid * 4)
        var b1 = u32b(bc * KHALF + bo + 16 + tid * 4)
        var sa = u32sa(sar * KB + s * 4)
        var sb = u32sb(bc * KB + s * 4)
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
        if col0 < N:
            cp[r0 * N + col0] = c0.cast[c_t]()
        if col1 < N:
            cp[r0 * N + col1] = c1.cast[c_t]()
    if r1 < m_dim:
        if col0 < N:
            cp[r1 * N + col0] = c2.cast[c_t]()
        if col1 < N:
            cp[r1 * N + col1] = c3.cast[c_t]()


def nvfp4_w4a4_matmul_cuda(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.bfloat16, ...],
    b_packed: TileTensor[DType.uint8, ...],
    b_scales: TileTensor[DType.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    comptime static_N = type_of(c).static_shape[1]
    comptime static_K = type_of(a).static_shape[1]
    var m = Int(a.dim[0]())

    var a_pk_buf = ctx.enqueue_create_buffer[DType.uint8](m * (static_K // 2))
    var a_sc_buf = ctx.enqueue_create_buffer[DType.float8_e4m3fn](
        m * (static_K // 16)
    )
    var a_pk_tt = TileTensor(
        a_pk_buf.unsafe_ptr(), row_major(m, Idx[static_K // 2])
    )
    var a_sc_tt = TileTensor(
        a_sc_buf.unsafe_ptr(), row_major(m, Idx[static_K // 16])
    )

    var a_im = a.as_immut()
    ctx.enqueue_function[
        quant_act_kernel[
            static_K,
            type_of(a_im).LayoutType,
            type_of(a_pk_tt).LayoutType,
            type_of(a_sc_tt).LayoutType,
        ]
    ](
        a_im,
        a_pk_tt,
        a_sc_tt,
        m,
        grid_dim=ceildiv(m * (static_K // 16), 128),
        block_dim=128,
    )
    var apk_im = a_pk_tt.as_immut()
    var asc_im = a_sc_tt.as_immut()
    var bp_im = b_packed.as_immut()
    var bs_im = b_scales.as_immut()
    ctx.enqueue_function[
        gemm_kernel[
            static_N,
            static_K,
            type_of(apk_im).LayoutType,
            type_of(asc_im).LayoutType,
            type_of(bp_im).LayoutType,
            type_of(bs_im).LayoutType,
            type_of(c).dtype,
            type_of(c).LayoutType,
        ]
    ](
        apk_im,
        asc_im,
        bp_im,
        bs_im,
        c,
        m,
        grid_dim=(ceildiv(static_N, 8), ceildiv(m, 16)),
        block_dim=32,
    )
    _ = a_pk_buf^
    _ = a_sc_buf^


comptime M = 64
comptime N = 128
comptime K = 256


def main() raises:
    with DeviceContext() as ctx:
        print("D2/M1e launcher test [", M, "x", N, "x", K, "]")
        comptime KHALF = K // 2
        comptime KB = K // 16

        var x_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * K)
        var wpk_h = ctx.enqueue_create_host_buffer[DType.uint8](N * KHALF)
        var wsc_h = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](N * KB)
        var c_h = ctx.enqueue_create_host_buffer[DType.bfloat16](M * N)
        ctx.synchronize()
        for i in range(M * K):
            var h = (i * 1103515245 + 12345) % 2000
            x_h[i] = ((Float32(h) / 1000.0 - 1.0) * 3.0).cast[DType.bfloat16]()
        for i in range(N * KHALF):
            wpk_h[i] = UInt8((i * 40503 + 7) % 256)
        for i in range(N * KB):
            var sv = (Float32((i * 13 + 5) % 4) + 1.0) * 0.5
            wsc_h[i] = sv.cast[DType.float8_e4m3fn]()

        var x_d = ctx.enqueue_create_buffer[DType.bfloat16](M * K)
        var wpk_d = ctx.enqueue_create_buffer[DType.uint8](N * KHALF)
        var wsc_d = ctx.enqueue_create_buffer[DType.float8_e4m3fn](N * KB)
        var c_d = ctx.enqueue_create_buffer[DType.bfloat16](M * N)
        ctx.enqueue_copy(x_d, x_h)
        ctx.enqueue_copy(wpk_d, wpk_h)
        ctx.enqueue_copy(wsc_d, wsc_h)

        var mm = M
        var c_tt = TileTensor(c_d.unsafe_ptr(), row_major(mm, Idx[N]))
        var a_tt = TileTensor(x_d.unsafe_ptr(), row_major(mm, Idx[K]))
        var wpk_tt = TileTensor(
            wpk_d.unsafe_ptr(), row_major(Idx[N], Idx[KHALF])
        )
        var wsc_tt = TileTensor(wsc_d.unsafe_ptr(), row_major(Idx[N], Idx[KB]))

        nvfp4_w4a4_matmul_cuda(c_tt, a_tt, wpk_tt, wsc_tt, ctx)

        # Expose the kernel's ACTUAL quantized activation (deterministic re-run)
        # so the reference uses the same quant the launcher used (e2e approach).
        var apk_d = ctx.enqueue_create_buffer[DType.uint8](M * KHALF)
        var asc_d = ctx.enqueue_create_buffer[DType.float8_e4m3fn](M * KB)
        var apk_tt = TileTensor(apk_d.unsafe_ptr(), row_major(mm, Idx[KHALF]))
        var asc_tt = TileTensor(asc_d.unsafe_ptr(), row_major(mm, Idx[KB]))
        var a_im2 = a_tt.as_immut()
        ctx.enqueue_function[
            quant_act_kernel[
                K,
                type_of(a_im2).LayoutType,
                type_of(apk_tt).LayoutType,
                type_of(asc_tt).LayoutType,
            ]
        ](
            a_im2, apk_tt, asc_tt, mm,
            grid_dim=ceildiv(M * KB, 128), block_dim=128,
        )
        var apk_h = ctx.enqueue_create_host_buffer[DType.uint8](M * KHALF)
        var asc_h = ctx.enqueue_create_host_buffer[DType.float8_e4m3fn](M * KB)
        ctx.enqueue_copy(apk_h, apk_d)
        ctx.enqueue_copy(asc_h, asc_d)
        ctx.enqueue_copy(c_h, c_d)
        ctx.synchronize()

        var pass_ = True
        var n_bad = 0
        for mi in range(M):
            for ni in range(N):
                var acc = Float32(0)
                for k in range(K):
                    var kb = k // 16
                    var an = Int(
                        (apk_h[mi * KHALF + k // 2] >> UInt8(4 * (k % 2)))
                        & UInt8(0xF)
                    )
                    var av = E2M1[an] * Float32(asc_h[mi * KB + kb])
                    var wn = Int(
                        (wpk_h[ni * KHALF + k // 2] >> UInt8(4 * (k % 2)))
                        & UInt8(0xF)
                    )
                    var wv = E2M1[wn] * Float32(wsc_h[ni * KB + kb])
                    acc += av * wv
                var got = Float32(c_h[mi * N + ni])
                if abs(got - acc) > Float32(0.25) + Float32(6e-3) * abs(acc):
                    if n_bad < 6:
                        print("  FAIL [", mi, ",", ni, "] got", got, "exp", acc)
                    n_bad += 1
                    pass_ = False
        if pass_:
            print("  PASS (launcher TileTensor -> quant -> FP4 GEMM matches ref)")
        else:
            print("  FAILED (", n_bad, ")")
