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
"""NVFP4 W4A4 matmul for NVIDIA sm_120: native FP4xFP4 block-scaled tensor cores.

Unlike the W4A16 path (which decodes the FP4 weight to bf16 and runs a bf16 GEMM),
this quantizes the bf16 activation to FP4 too and runs the NATIVE block-scaled FP4
tensor-core MMA (`mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X`,
sm_120a), which does the block-scale application in hardware. Per-tensor
`weight_scale_2` (and the activation's per-tensor scale, which cancels into the
dynamic per-block activation scale) are folded post-matmul by the graph caller.

Pipeline: quantize activation (bf16 -> packed FP4 [M, K/2] + fp8-e4m3 block-16
scales [M, K/16]) -> FP4xFP4 block-scaled GEMM (out = a @ w^T) -> bf16.
Correctness-first tiling (one warp per [16,8] tile); perf tuning is a later phase.
"""

from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
)
from std.memory import bitcast
from std.sys import _RegisterPackType
from std.sys._assembly import inlined_assembly
from layout import Idx, Layout, LayoutTensor, TileTensor, TensorLayout
from layout.tile_layout import row_major


comptime MMA_ASM = (
    "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale"
    ".scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
    "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, {$10, $11, $12, $13}, "
    "$14, {0, 0}, $15, {0, 0};"
)


@always_inline
def _round_e2m1_nibble(v: Float32) -> UInt8:
    """Round to the nearest signed E2M1 code (0..15). Grid {0,.5,1,1.5,2,3,4,6}."""
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


def _quant_act_kernel[
    K: Int, x_l: TensorLayout, pk_l: TensorLayout, sc_l: TensorLayout
](
    x: TileTensor[DType.bfloat16, x_l, ImmutAnyOrigin],
    out_packed: TileTensor[DType.uint8, pk_l, MutAnyOrigin],
    out_scale: TileTensor[DType.float8_e4m3fn, sc_l, MutAnyOrigin],
    m_dim: Int,
):
    """One thread per (row, K-block): dynamic per-block-16 NVFP4 activation quant."""
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
        var lo = _round_e2m1_nibble(Float32(xp[base + 2 * jb]) * inv)
        var hi = _round_e2m1_nibble(Float32(xp[base + 2 * jb + 1]) * inv)
        pkp[m * KHALF + kb * 8 + jb] = lo | (hi << 4)


def _gemm_kernel[
    N: Int, K: Int, ap_l: TensorLayout, asc_l: TensorLayout,
    bp_l: TensorLayout, bsc_l: TensorLayout, c_t: DType, c_l: TensorLayout,
    s2_l: TensorLayout,
](
    a_packed: TileTensor[DType.uint8, ap_l, ImmutAnyOrigin],
    a_scale: TileTensor[DType.float8_e4m3fn, asc_l, ImmutAnyOrigin],
    b_packed: TileTensor[DType.uint8, bp_l, ImmutAnyOrigin],
    b_scale: TileTensor[DType.float8_e4m3fn, bsc_l, ImmutAnyOrigin],
    c: TileTensor[c_t, c_l, MutAnyOrigin],
    s2_t: TileTensor[DType.float32, s2_l, ImmutAnyOrigin],
    m_dim: Int,
):
    """One warp -> one [16,8] output tile; K-loop over m16n8k64 strips.

    `s2_t` is the per-tensor `weight_scale_2` (1 f32). It is applied in the
    epilogue with the same double rounding the graph used to do post-matmul
    (acc -> c_t -> f32 -> *s2 -> c_t), so fused output is BYTE-IDENTICAL to
    the previous unfused `(cast(f32)(C) * s2).cast(bf16)` graph fold.
    """
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
    var s2 = s2_t.ptr.address_space_cast[AddressSpace.GLOBAL]()[0]

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
    # Double-rounded s2 fold (see docstring): acc -> c_t -> f32 -> *s2 -> c_t.
    @always_inline
    @parameter
    def _fold(v: Float32) -> Scalar[c_t]:
        return (v.cast[c_t]().cast[DType.float32]() * s2).cast[c_t]()

    if r0 < m_dim:
        if col0 < N:
            cp[r0 * N + col0] = _fold(c0)
        if col1 < N:
            cp[r0 * N + col1] = _fold(c1)
    if r1 < m_dim:
        if col0 < N:
            cp[r1 * N + col0] = _fold(c2)
        if col1 < N:
            cp[r1 * N + col1] = _fold(c3)


def _gemm_kernel_tiled[
    N: Int, K: Int, BM: Int, BN: Int, BK: Int, WM: Int, WN: Int,
    ap_l: TensorLayout, asc_l: TensorLayout,
    bp_l: TensorLayout, bsc_l: TensorLayout, c_t: DType, c_l: TensorLayout,
    s2_l: TensorLayout,
    NUM_STAGES: Int = 2,
](
    a_packed: TileTensor[DType.uint8, ap_l, ImmutAnyOrigin],
    a_scale: TileTensor[DType.float8_e4m3fn, asc_l, ImmutAnyOrigin],
    b_packed: TileTensor[DType.uint8, bp_l, ImmutAnyOrigin],
    b_scale: TileTensor[DType.float8_e4m3fn, bsc_l, ImmutAnyOrigin],
    c: TileTensor[c_t, c_l, MutAnyOrigin],
    s2_t: TileTensor[DType.float32, s2_l, ImmutAnyOrigin],
    m_dim: Int,
):
    """SMEM-tiled native FP4xFP4 GEMM: block computes [BM,BN] with operand reuse.

    Each threadblock (num_warps warps) computes a [BM,BN] C tile. Per K-tile of
    depth BK it cooperatively stages the packed-FP4 A[BM,BK] + B[BN,BK] sub-tiles
    and their FP8 block scales into shared memory (coalesced aligned u32 loads),
    then every warp runs its WMxWN grid of m16n8k64 mxf4nvf4 MMAs reading fragments
    straight from SMEM (aligned u32, no byte assembly) -- so each staged operand
    is reused across the block's MMAs. f32 accumulation, bf16 epilogue. Fragment +
    scale addressing is identical to the validated `_gemm_kernel` (the oracle),
    just re-sourced from SMEM. Requires K%BK==0, BK%64==0, BM%WM==0, BN%WN==0,
    WM%16==0, WN%8==0; N%BN==0 (all Klein Linears satisfy these). M is dynamic.
    """
    comptime KHALF = K // 2
    comptime KB = K // 16
    comptime BKH = BK // 2               # packed bytes per staged row
    comptime AROWU = BKH // 4            # u32 per staged packed row (= BK//8)
    comptime BKB = BK // 16              # fp8 scales per staged row
    comptime ASROWU = BKB // 4           # u32 per staged scale row (= BK//64)
    comptime NST = BK // 64              # m16n8k64 strips per staged K-tile
    comptime num_m_mmas = WM // 16
    comptime num_n_mmas = WN // 8
    comptime num_warps_n = BN // WN
    comptime num_threads = (BM // WM) * num_warps_n * 32
    comptime NMMA = num_m_mmas * num_n_mmas

    # SMEM as two TYPED u32 LayoutTensors, indexed through the tensors everywhere
    # (never via an extracted raw pointer -- `.ptr` is used ONLY as the cp.async
    # `dst`, while the MMA still reads via `pk_s[i,j]`, so the tensor value stays
    # live and LLVM can't overlay the alloca). NUM_STAGES double-buffer: stage
    # `buf` occupies rows `[buf*PKROWS : (buf+1)*PKROWS]`; within a stage rows
    # 0..BM-1 hold A, rows BM..BM+BN-1 hold B (distinct shapes, no aliasing).
    comptime PKROWS = BM + BN
    var pk_s = LayoutTensor[
        DType.uint32, Layout.row_major(NUM_STAGES * PKROWS, AROWU), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var sc_s = LayoutTensor[
        DType.uint32, Layout.row_major(NUM_STAGES * PKROWS, ASROWU),
        MutAnyOrigin, address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var gA = a_packed.ptr.address_space_cast[AddressSpace.GLOBAL]().bitcast[
        UInt32
    ]()
    var gAs = a_scale.ptr.address_space_cast[AddressSpace.GLOBAL]().bitcast[
        UInt32
    ]()
    var gB = b_packed.ptr.address_space_cast[AddressSpace.GLOBAL]().bitcast[
        UInt32
    ]()
    var gBs = b_scale.ptr.address_space_cast[AddressSpace.GLOBAL]().bitcast[
        UInt32
    ]()
    var cp = c.ptr.address_space_cast[AddressSpace.GLOBAL]()
    var s2 = s2_t.ptr.address_space_cast[AddressSpace.GLOBAL]()[0]

    var tid = Int(thread_idx.x)
    var warp = tid // 32
    var lane = tid % 32
    var wy = warp // num_warps_n
    var wx = warp % num_warps_n
    var group = lane >> 2
    var t4 = lane & 3

    var block_row0 = Int(block_idx.y) * BM
    var block_col0 = Int(block_idx.x) * BN

    # f32 accumulators, accessed only through the tensor (see SMEM note above).
    var acc = (
        LayoutTensor[
            DType.float32, Layout.row_major(NMMA, 4), MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    comptime KROW_U32 = K // 8           # u32 per weight/act row (KHALF//4)
    comptime KSROW_U32 = KB // 4         # u32 per scale row
    comptime CPR = AROWU // 4            # 16-byte (4-u32) chunks per packed row
    var nkt = K // BK

    # ---- cp.async load of one K-tile into SMEM stage `buf`. Packed A/B stream as
    # 16-byte coalesced copies; the tiny fp8 scales as 4-byte copies. OOB-M rows
    # (A only; N%BN==0 -> B has no edge) zero-fill via src_size=0. K%BK==0 -> no
    # K-tail. `.ptr` is only the cp.async dst; the MMA reads via pk_s/sc_s[i,j].
    # Caller commits the group.
    @always_inline
    @parameter
    def _load_strip(buf: Int, kt: Int):
        var ku = (kt * BK) // 8
        var ksu = (kt * BK) // 64
        var sr = buf * PKROWS
        for ch in range(tid, BM * CPR, num_threads):
            var row = ch // CPR
            var uc = (ch % CPR) * 4
            var m_abs = block_row0 + row
            var dst = pk_s.ptr + ((sr + row) * AROWU + uc)
            var src = gA + (m_abs * KROW_U32 + ku + uc)
            if m_abs < m_dim:
                async_copy[16](src, dst)
            else:
                async_copy[16, fill = Scalar[DType.uint32](0)](
                    src, dst, src_size=Int32(0)
                )
        for ch in range(tid, BN * CPR, num_threads):
            var row = ch // CPR
            var uc = (ch % CPR) * 4
            var dst = pk_s.ptr + ((sr + BM + row) * AROWU + uc)
            async_copy[16](gB + ((block_col0 + row) * KROW_U32 + ku + uc), dst)
        for u in range(tid, BM * ASROWU, num_threads):
            var row = u // ASROWU
            var uc = u % ASROWU
            var m_abs = block_row0 + row
            var dst = sc_s.ptr + ((sr + row) * ASROWU + uc)
            var src = gAs + (m_abs * KSROW_U32 + ksu + uc)
            if m_abs < m_dim:
                async_copy[4](src, dst)
            else:
                async_copy[4, fill = Scalar[DType.uint32](0)](
                    src, dst, src_size=Int32(0)
                )
        for u in range(tid, BN * ASROWU, num_threads):
            var row = u // ASROWU
            var uc = u % ASROWU
            var dst = sc_s.ptr + ((sr + BM + row) * ASROWU + uc)
            async_copy[4](gBs + ((block_col0 + row) * KSROW_U32 + ksu + uc), dst)

    # ---- NUM_STAGES-deep cp.async pipeline: prologue issues the first
    # NUM_STAGES-1 K-tiles; each iter prefetches NUM_STAGES-1 ahead, drains to the
    # in-flight depth (wait_group needs an IMMEDIATE, so enumerate at comptime),
    # then MMAs stage `cur`. Barrier after wait (cross-thread SMEM visible) and
    # after MMA (readers done before this stage is overwritten NUM_STAGES iters on).
    comptime for s in range(NUM_STAGES - 1):
        if s < nkt:
            _load_strip(s, s)
            async_copy_commit_group()

    for kt in range(nkt):
        var cur = kt % NUM_STAGES
        var pf = kt + (NUM_STAGES - 1)
        if pf < nkt:
            _load_strip(pf % NUM_STAGES, pf)
            async_copy_commit_group()
        var remaining = min(NUM_STAGES - 1, nkt - 1 - kt)
        comptime for rem in range(NUM_STAGES):
            if remaining == rem:
                async_copy_wait_group(Int32(rem))
        barrier()

        # ---- MMA: every warp runs its WMxWN grid from SMEM stage `cur`. ----
        var sbase = cur * PKROWS
        comptime for ks in range(NST):
            comptime lo = ks * 8 + 0
            comptime hi = ks * 8 + 4
            comptime for mi in range(num_m_mmas):
                comptime for ni in range(num_n_mmas):
                    var arow = wy * WM + mi * 16
                    var brow = BM + wx * WN + ni * 8
                    var a0 = pk_s[sbase + arow + group, lo + t4][0]
                    var a1 = pk_s[sbase + arow + group + 8, lo + t4][0]
                    var a2 = pk_s[sbase + arow + group, hi + t4][0]
                    var a3 = pk_s[sbase + arow + group + 8, hi + t4][0]
                    var b0 = pk_s[sbase + brow + group, lo + t4][0]
                    var b1 = pk_s[sbase + brow + group, hi + t4][0]
                    var arsel = arow + (group if (t4 & 1) == 0 else group + 8)
                    var sa = sc_s[sbase + arsel, ks][0]
                    var sb = sc_s[sbase + brow + group, ks][0]
                    comptime idx = mi * num_n_mmas + ni
                    var r = inlined_assembly[
                        MMA_ASM,
                        _RegisterPackType[Float32, Float32, Float32, Float32],
                        constraints="=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r,r,r",
                    ](
                        a0, a1, a2, a3, b0, b1,
                        acc[idx, 0][0], acc[idx, 1][0],
                        acc[idx, 2][0], acc[idx, 3][0], sa, sb,
                    )
                    acc[idx, 0] = r[0]
                    acc[idx, 1] = r[1]
                    acc[idx, 2] = r[2]
                    acc[idx, 3] = r[3]
        barrier()

    # ---- epilogue: f32 accumulators -> C (bounds-guarded on M). The per-tensor
    # `weight_scale_2` is folded here with the same double rounding the graph
    # used to do post-matmul (acc -> c_t -> f32 -> *s2 -> c_t), so the fused
    # output stays BYTE-IDENTICAL to the previous unfused graph fold. ----
    @always_inline
    @parameter
    def _fold(v: Float32) -> Scalar[c_t]:
        return (v.cast[c_t]().cast[DType.float32]() * s2).cast[c_t]()

    comptime for mi in range(num_m_mmas):
        comptime for ni in range(num_n_mmas):
            comptime idx = mi * num_n_mmas + ni
            var r0 = block_row0 + wy * WM + mi * 16 + group
            var r1 = r0 + 8
            var col0 = block_col0 + wx * WN + ni * 8 + 2 * t4
            var col1 = col0 + 1
            if r0 < m_dim:
                if col0 < N:
                    cp[r0 * N + col0] = _fold(acc[idx, 0][0])
                if col1 < N:
                    cp[r0 * N + col1] = _fold(acc[idx, 1][0])
            if r1 < m_dim:
                if col0 < N:
                    cp[r1 * N + col0] = _fold(acc[idx, 2][0])
                if col1 < N:
                    cp[r1 * N + col1] = _fold(acc[idx, 3][0])


def nvfp4_w4a4_matmul_cuda(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.bfloat16, ...],
    b_packed: TileTensor[DType.uint8, ...],
    b_scales: TileTensor[DType.float8_e4m3fn, ...],
    s2: TileTensor[DType.float32, ...],
    ctx: DeviceContext,
) raises:
    """W4A4: quantize bf16 activation -> FP4+scales, then native FP4xFP4 GEMM.

    Args:
        c: Output `[M, N]` bf16.
        a: Activations `[M, K]` bf16.
        b_packed: Packed FP4 weight `[N, K // 2]` uint8 (low nibble = even K).
        b_scales: FP8-e4m3 weight block scales `[N, K // 16]` (block 16).
        s2: Per-tensor `weight_scale_2` (1 f32), folded in the GEMM epilogue
            with double rounding -- byte-identical to the old graph-side fold.
        ctx: Device context.
    """
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
        _quant_act_kernel[
            static_K,
            type_of(a_im).LayoutType,
            type_of(a_pk_tt).LayoutType,
            type_of(a_sc_tt).LayoutType,
        ]
    ](
        a_im, a_pk_tt, a_sc_tt, m,
        grid_dim=ceildiv(m * (static_K // 16), 128), block_dim=128,
    )
    var apk_im = a_pk_tt.as_immut()
    var asc_im = a_sc_tt.as_immut()
    var bp_im = b_packed.as_immut()
    var bs_im = b_scales.as_immut()
    var s2_im = s2.as_immut()
    ctx.enqueue_function[
        _gemm_kernel[
            static_N,
            static_K,
            type_of(apk_im).LayoutType,
            type_of(asc_im).LayoutType,
            type_of(bp_im).LayoutType,
            type_of(bs_im).LayoutType,
            type_of(c).dtype,
            type_of(c).LayoutType,
            type_of(s2_im).LayoutType,
        ]
    ](
        apk_im, asc_im, bp_im, bs_im, c, s2_im, m,
        grid_dim=(ceildiv(static_N, 8), ceildiv(m, 16)), block_dim=32,
    )
    _ = a_pk_buf^
    _ = a_sc_buf^


def _gemm_kernel_tiled_fq[
    N: Int, K: Int, BM: Int, BN: Int, BK: Int, WM: Int, WN: Int,
    abf_l: TensorLayout,
    bp_l: TensorLayout, bsc_l: TensorLayout, c_t: DType, c_l: TensorLayout,
    s2_l: TensorLayout,
    NUM_STAGES: Int = 2,
](
    a_bf16: TileTensor[DType.bfloat16, abf_l, ImmutAnyOrigin],
    b_packed: TileTensor[DType.uint8, bp_l, ImmutAnyOrigin],
    b_scale: TileTensor[DType.float8_e4m3fn, bsc_l, ImmutAnyOrigin],
    c: TileTensor[c_t, c_l, MutAnyOrigin],
    s2_t: TileTensor[DType.float32, s2_l, ImmutAnyOrigin],
    m_dim: Int,
):
    """`_gemm_kernel_tiled` with the activation quant FUSED into the prologue.

    The activation arrives as bf16 `[M, K]`; each K-tile's A sub-tile is
    quantized register-direct into the SAME `pk_s`/`sc_s` SMEM bytes the
    packed-A kernel stages via cp.async: one thread owns one (row, 64-elem
    k-strip) chunk, loads 16B-vectorized bf16 straight from global (L2-resident
    across the grid's N block-columns), runs the EXACT `_quant_act_kernel`
    recipe (amax/6 -> e4m3 roundtrip -> `_round_e2m1_nibble`), and assembles
    the identical little-endian u32 packed/scale words. Deterministic and
    bit-identical to the two-pass path; redundant per-block-column quant is
    cheap ALU on L2-fed data. B (and its scales) keep the cp.async pipeline;
    `wait_group` therefore only tracks B copies. SMEM layout/size unchanged.
    """
    comptime KHALF = K // 2
    comptime KB = K // 16
    comptime BKH = BK // 2
    comptime AROWU = BKH // 4
    comptime BKB = BK // 16
    comptime ASROWU = BKB // 4
    comptime NST = BK // 64
    comptime num_m_mmas = WM // 16
    comptime num_n_mmas = WN // 8
    comptime num_warps_n = BN // WN
    comptime num_threads = (BM // WM) * num_warps_n * 32
    comptime NMMA = num_m_mmas * num_n_mmas

    comptime PKROWS = BM + BN
    var pk_s = LayoutTensor[
        DType.uint32, Layout.row_major(NUM_STAGES * PKROWS, AROWU), MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var sc_s = LayoutTensor[
        DType.uint32, Layout.row_major(NUM_STAGES * PKROWS, ASROWU),
        MutAnyOrigin, address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var gab = a_bf16.ptr.address_space_cast[AddressSpace.GLOBAL]()
    var gB = b_packed.ptr.address_space_cast[AddressSpace.GLOBAL]().bitcast[
        UInt32
    ]()
    var gBs = b_scale.ptr.address_space_cast[AddressSpace.GLOBAL]().bitcast[
        UInt32
    ]()
    var cp = c.ptr.address_space_cast[AddressSpace.GLOBAL]()
    var s2 = s2_t.ptr.address_space_cast[AddressSpace.GLOBAL]()[0]

    var tid = Int(thread_idx.x)
    var warp = tid // 32
    var lane = tid % 32
    var wy = warp // num_warps_n
    var wx = warp % num_warps_n
    var group = lane >> 2
    var t4 = lane & 3

    var block_row0 = Int(block_idx.y) * BM
    var block_col0 = Int(block_idx.x) * BN

    var acc = (
        LayoutTensor[
            DType.float32, Layout.row_major(NMMA, 4), MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    comptime KROW_U32 = K // 8
    comptime KSROW_U32 = KB // 4
    comptime CPR = AROWU // 4
    var nkt = K // BK

    # ---- cp.async load of one K-tile of B (packed + scales) into stage `buf`.
    # Identical to the B half of `_gemm_kernel_tiled._load_strip`.
    @always_inline
    @parameter
    def _load_strip_b(buf: Int, kt: Int):
        var ku = (kt * BK) // 8
        var ksu = (kt * BK) // 64
        var sr = buf * PKROWS
        for ch in range(tid, BN * CPR, num_threads):
            var row = ch // CPR
            var uc = (ch % CPR) * 4
            var dst = pk_s.ptr + ((sr + BM + row) * AROWU + uc)
            async_copy[16](gB + ((block_col0 + row) * KROW_U32 + ku + uc), dst)
        for u in range(tid, BN * ASROWU, num_threads):
            var row = u // ASROWU
            var uc = u % ASROWU
            var dst = sc_s.ptr + ((sr + BM + row) * ASROWU + uc)
            async_copy[4](gBs + ((block_col0 + row) * KSROW_U32 + ksu + uc), dst)

    # ---- register-direct quant of one K-tile of A into stage `buf`. One
    # thread per (row, 64-elem strip): 4 blocks of 16 -> 1 scale u32 + 8 packed
    # u32 (little-endian byte order matching the cp.async'd memory image). OOB
    # rows (M edge) write zeros, matching the packed path's zero-fill.
    @always_inline
    @parameter
    def _stage_a_quant(buf: Int, kt: Int):
        var sr = buf * PKROWS
        for ch in range(tid, BM * NST, num_threads):
            var row = ch // NST
            var ks = ch % NST
            var m_abs = block_row0 + row
            var sa_u32 = UInt32(0)
            var pk_w = SIMD[DType.uint32, 8](0)
            if m_abs < m_dim:
                var base = m_abs * K + kt * BK + ks * 64
                comptime for b4 in range(4):
                    var v16 = (
                        gab.load[width=8](base + b4 * 16)
                        .cast[DType.float32]()
                        .join(
                            gab.load[width=8](base + b4 * 16 + 8).cast[
                                DType.float32
                            ]()
                        )
                    )
                    var amax = abs(v16).reduce_max()
                    var bs8 = SIMD[DType.float8_e4m3fn, 1](
                        (amax / 6.0).cast[DType.float8_e4m3fn]()
                    )
                    sa_u32 |= (
                        UInt32(bitcast[DType.uint8, 1](bs8)[0])
                        << UInt32(8 * b4)
                    )
                    var bs = Float32(bs8[0])
                    var inv = (
                        Float32(1.0) / bs
                    ) if bs > 0.0 else Float32(0.0)
                    comptime for j in range(8):
                        var lo = _round_e2m1_nibble(v16[2 * j] * inv)
                        var hi = _round_e2m1_nibble(v16[2 * j + 1] * inv)
                        comptime pb = b4 * 8 + j
                        pk_w[pb // 4] |= (
                            UInt32(lo | (hi << 4)) << UInt32(8 * (pb % 4))
                        )
            sc_s[sr + row, ks] = sa_u32
            comptime for w in range(8):
                pk_s[sr + row, ks * 8 + w] = pk_w[w]

    # ---- NUM_STAGES pipeline: B via cp.async, A quantized in-place. The
    # existing end-of-iteration barrier already orders the plain SMEM stores
    # of the NEXT stage against the current stage's MMA readers.
    comptime for s in range(NUM_STAGES - 1):
        if s < nkt:
            _load_strip_b(s, s)
            async_copy_commit_group()
            _stage_a_quant(s, s)

    for kt in range(nkt):
        var cur = kt % NUM_STAGES
        var pf = kt + (NUM_STAGES - 1)
        if pf < nkt:
            _load_strip_b(pf % NUM_STAGES, pf)
            async_copy_commit_group()
            _stage_a_quant(pf % NUM_STAGES, pf)
        var remaining = min(NUM_STAGES - 1, nkt - 1 - kt)
        comptime for rem in range(NUM_STAGES):
            if remaining == rem:
                async_copy_wait_group(Int32(rem))
        barrier()

        # ---- MMA: identical to `_gemm_kernel_tiled`. ----
        var sbase = cur * PKROWS
        comptime for ks in range(NST):
            comptime lo = ks * 8 + 0
            comptime hi = ks * 8 + 4
            comptime for mi in range(num_m_mmas):
                comptime for ni in range(num_n_mmas):
                    var arow = wy * WM + mi * 16
                    var brow = BM + wx * WN + ni * 8
                    var a0 = pk_s[sbase + arow + group, lo + t4][0]
                    var a1 = pk_s[sbase + arow + group + 8, lo + t4][0]
                    var a2 = pk_s[sbase + arow + group, hi + t4][0]
                    var a3 = pk_s[sbase + arow + group + 8, hi + t4][0]
                    var b0 = pk_s[sbase + brow + group, lo + t4][0]
                    var b1 = pk_s[sbase + brow + group, hi + t4][0]
                    var arsel = arow + (group if (t4 & 1) == 0 else group + 8)
                    var sa = sc_s[sbase + arsel, ks][0]
                    var sb = sc_s[sbase + brow + group, ks][0]
                    comptime idx = mi * num_n_mmas + ni
                    var r = inlined_assembly[
                        MMA_ASM,
                        _RegisterPackType[Float32, Float32, Float32, Float32],
                        constraints="=f,=f,=f,=f,r,r,r,r,r,r,r,r,r,r,r,r",
                    ](
                        a0, a1, a2, a3, b0, b1,
                        acc[idx, 0][0], acc[idx, 1][0],
                        acc[idx, 2][0], acc[idx, 3][0], sa, sb,
                    )
                    acc[idx, 0] = r[0]
                    acc[idx, 1] = r[1]
                    acc[idx, 2] = r[2]
                    acc[idx, 3] = r[3]
        barrier()

    # ---- epilogue: identical to `_gemm_kernel_tiled` (double-rounded s2). ----
    @always_inline
    @parameter
    def _fold(v: Float32) -> Scalar[c_t]:
        return (v.cast[c_t]().cast[DType.float32]() * s2).cast[c_t]()

    comptime for mi in range(num_m_mmas):
        comptime for ni in range(num_n_mmas):
            comptime idx = mi * num_n_mmas + ni
            var r0 = block_row0 + wy * WM + mi * 16 + group
            var r1 = r0 + 8
            var col0 = block_col0 + wx * WN + ni * 8 + 2 * t4
            var col1 = col0 + 1
            if r0 < m_dim:
                if col0 < N:
                    cp[r0 * N + col0] = _fold(acc[idx, 0][0])
                if col1 < N:
                    cp[r0 * N + col1] = _fold(acc[idx, 1][0])
            if r1 < m_dim:
                if col0 < N:
                    cp[r1 * N + col0] = _fold(acc[idx, 2][0])
                if col1 < N:
                    cp[r1 * N + col1] = _fold(acc[idx, 3][0])


def nvfp4_w4a4_matmul_cuda_tiled_fusedq(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.bfloat16, ...],
    b_packed: TileTensor[DType.uint8, ...],
    b_scales: TileTensor[DType.float8_e4m3fn, ...],
    s2: TileTensor[DType.float32, ...],
    ctx: DeviceContext,
) raises:
    """W4A4 tiled GEMM with the activation quant FUSED into the GEMM prologue.

    Same contract and BIT-IDENTICAL result as `nvfp4_w4a4_matmul_cuda_tiled`,
    but no standalone quant kernel and no packed/scale DRAM round-trip: the
    GEMM reads bf16 `a` directly and quantizes each A tile in-kernel
    (`_gemm_kernel_tiled_fq`). One kernel launch per matmul.
    """
    comptime NUM_STAGES = 2

    comptime static_N = type_of(c).static_shape[1]
    comptime static_K = type_of(a).static_shape[1]
    var m = Int(a.dim[0]())

    var a_im = a.as_immut()
    var bp_im = b_packed.as_immut()
    var bs_im = b_scales.as_immut()
    var s2_im = s2.as_immut()
    @parameter
    def _gemm[BM_: Int, BN_: Int, WM_: Int, WN_: Int, BK_: Int]() raises:
        comptime nthreads = (BM_ // WM_) * (BN_ // WN_) * 32
        ctx.enqueue_function[
            _gemm_kernel_tiled_fq[
                static_N,
                static_K,
                BM_,
                BN_,
                BK_,
                WM_,
                WN_,
                type_of(a_im).LayoutType,
                type_of(bp_im).LayoutType,
                type_of(bs_im).LayoutType,
                type_of(c).dtype,
                type_of(c).LayoutType,
                type_of(s2_im).LayoutType,
                NUM_STAGES,
            ]
        ](
            a_im, bp_im, bs_im, c, s2_im, m,
            grid_dim=(ceildiv(static_N, BN_), ceildiv(m, BM_)),
            block_dim=nthreads,
        )

    # Same tile-size dispatch as the two-pass tiled launcher.
    comptime SMALL_BK = 256 if static_K % 256 == 0 else (
        128 if static_K % 128 == 0 else 64
    )
    comptime if static_N % 128 == 0 and static_K % 128 == 0:
        if m >= 256:
            _gemm[128, 128, 64, 32, 128]()
        else:
            _gemm[64, 64, 32, 32, SMALL_BK]()
    else:
        _gemm[64, 64, 32, 32, SMALL_BK]()


def nvfp4_quant_act_cuda(
    out_packed: TileTensor[mut=True, ...],
    out_scale: TileTensor[mut=True, ...],
    a: TileTensor[DType.bfloat16, ...],
    ctx: DeviceContext,
) raises:
    """Dynamic per-block-16 NVFP4 activation quant (standalone graph op).

    Same kernel + launch config the fused-in-launcher path used, exposed as its
    own op so the graph can (a) quantize an activation ONCE when several
    quantized Linears consume it (q/k/v projections, per-block modulation) and
    (b) let the graph allocator own the packed/scale buffers instead of
    per-call `enqueue_create_buffer` churn.

    Args:
        out_packed: Packed FP4 activation `[M, K // 2]` uint8 (low nibble = even K).
        out_scale: FP8-e4m3 activation block scales `[M, K // 16]` (block 16).
        a: Activations `[M, K]` bf16.
        ctx: Device context.
    """
    comptime static_K = type_of(a).static_shape[1]
    var m = Int(a.dim[0]())
    var a_im = a.as_immut()
    ctx.enqueue_function[
        _quant_act_kernel[
            static_K,
            type_of(a_im).LayoutType,
            type_of(out_packed).LayoutType,
            type_of(out_scale).LayoutType,
        ]
    ](
        a_im, out_packed, out_scale, m,
        grid_dim=ceildiv(m * (static_K // 16), 128), block_dim=128,
    )


def nvfp4_w4a4_matmul_cuda_tiled_prequant(
    c: TileTensor[mut=True, ...],
    a_packed: TileTensor[DType.uint8, ...],
    a_scales: TileTensor[DType.float8_e4m3fn, ...],
    b_packed: TileTensor[DType.uint8, ...],
    b_scales: TileTensor[DType.float8_e4m3fn, ...],
    s2: TileTensor[DType.float32, ...],
    ctx: DeviceContext,
) raises:
    """SMEM-tiled FP4xFP4 GEMM over a PRE-quantized activation.

    The GEMM half of `nvfp4_w4a4_matmul_cuda_tiled`: identical kernel,
    dispatch, and fused `weight_scale_2` epilogue, but the activation arrives
    already packed (from `nvfp4_quant_act_cuda`) so a shared activation is
    quantized once, not once per consuming Linear.

    Args:
        c: Output `[M, N]` bf16.
        a_packed: Packed FP4 activation `[M, K // 2]` uint8.
        a_scales: FP8-e4m3 activation block scales `[M, K // 16]`.
        b_packed: Packed FP4 weight `[N, K // 2]` uint8.
        b_scales: FP8-e4m3 weight block scales `[N, K // 16]`.
        s2: Per-tensor `weight_scale_2` (1 f32), double-round epilogue fold.
        ctx: Device context.
    """
    comptime NUM_STAGES = 2  # cp.async pipeline depth

    comptime static_N = type_of(c).static_shape[1]
    # K is static on the packed operands: [N, K//2].
    comptime static_K = 2 * type_of(b_packed).static_shape[1]
    var m = Int(a_packed.dim[0]())

    var apk_im = a_packed.as_immut()
    var asc_im = a_scales.as_immut()
    var bp_im = b_packed.as_immut()
    var bs_im = b_scales.as_immut()
    var s2_im = s2.as_immut()
    @parameter
    def _gemm[BM_: Int, BN_: Int, WM_: Int, WN_: Int, BK_: Int]() raises:
        comptime nthreads = (BM_ // WM_) * (BN_ // WN_) * 32
        ctx.enqueue_function[
            _gemm_kernel_tiled[
                static_N,
                static_K,
                BM_,
                BN_,
                BK_,
                WM_,
                WN_,
                type_of(apk_im).LayoutType,
                type_of(asc_im).LayoutType,
                type_of(bp_im).LayoutType,
                type_of(bs_im).LayoutType,
                type_of(c).dtype,
                type_of(c).LayoutType,
                type_of(s2_im).LayoutType,
                NUM_STAGES,
            ]
        ](
            apk_im, asc_im, bp_im, bs_im, c, s2_im, m,
            grid_dim=(ceildiv(static_N, BN_), ceildiv(m, BM_)),
            block_dim=nthreads,
        )

    # Tile-size dispatch. The big 128x128 tile (BK=128, 8 warps, ~36 KB SMEM)
    # beats bf16 outright at M>=256 (0.64-0.77x at M=4096) via high operand
    # reuse (A 16x, B 8x), but wastes rows at small M; the 64x64 tile (BK up to
    # 256) wins at small M. Big tile needs N%128==0 & K%128==0; else fall back to
    # 64x64 (needs N%64==0, always true for Klein). Small-path BK = largest
    # dividing K (256 preferred, fewest barriers). All Klein NVFP4 Linears are
    # N%128==0, K%256==0, so the render (M~4096) always takes the big tile.
    comptime SMALL_BK = 256 if static_K % 256 == 0 else (
        128 if static_K % 128 == 0 else 64
    )
    comptime if static_N % 128 == 0 and static_K % 128 == 0:
        if m >= 256:
            _gemm[128, 128, 64, 32, 128]()
        else:
            _gemm[64, 64, 32, 32, SMALL_BK]()
    else:
        _gemm[64, 64, 32, 32, SMALL_BK]()


def nvfp4_w4a4_matmul_cuda_tiled(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.bfloat16, ...],
    b_packed: TileTensor[DType.uint8, ...],
    b_scales: TileTensor[DType.float8_e4m3fn, ...],
    s2: TileTensor[DType.float32, ...],
    ctx: DeviceContext,
) raises:
    """W4A4 (tiled GEMM): quantize activation -> FP4, then SMEM-tiled FP4xFP4 GEMM.

    Same contract/result as `nvfp4_w4a4_matmul_cuda` but uses `_gemm_kernel_tiled`
    (shared-memory operand reuse) instead of the correctness-first one-warp-per-tile
    `_gemm_kernel`. Args as in `nvfp4_w4a4_matmul_cuda` (including the fused,
    double-rounded `weight_scale_2` epilogue fold via `s2`). Convenience
    all-in-one wrapper over `nvfp4_quant_act_cuda` +
    `nvfp4_w4a4_matmul_cuda_tiled_prequant` (the graph path calls those two ops
    directly so a shared activation is quantized once).
    """
    # Validated bit-identical to the naive `nvfp4_w4a4_matmul_cuda` and against a
    # device-quant host reference across full/edge-M, multi-K-tile, and tall-N
    # shapes (`nvfp4_w4a4_tiled_test.mojo`). Earlier "corruption" was a TEST
    # HARNESS DeviceBuffer-lifetime bug, not this kernel -- see the test's
    # module docstring and plans/nvfp4-d2-m2-benchmark-optimize.md.
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

    nvfp4_quant_act_cuda(a_pk_tt, a_sc_tt, a, ctx)
    nvfp4_w4a4_matmul_cuda_tiled_prequant(
        c, a_pk_tt, a_sc_tt, b_packed, b_scales, s2, ctx
    )
    _ = a_pk_buf^
    _ = a_sc_buf^
