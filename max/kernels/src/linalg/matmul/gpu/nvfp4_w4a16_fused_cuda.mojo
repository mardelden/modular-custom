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
"""NVFP4 weight-only (W4A16) FUSED matmul for NVIDIA GPUs (e.g. sm_120).

The FUSED sibling of `nvfp4_w4a16_cuda.mojo` (the Phase C "materialize -> dense"
launcher). Same operand contract and same numeric result, but instead of
dequantizing the ENTIRE packed FP4 weight into a transient dense bf16 `[N, K]`
DRAM buffer every forward pass, this kernel decodes packed FP4 weight sub-tiles
to bf16 IN SHARED MEMORY inside the GEMM mainloop and feeds them straight to the
standard bf16 tensor-core MMA. The weight is read from DRAM as 4-bit (1/4 the
bytes) and never materialized -- removing the transient-buffer DRAM traffic that
makes the materialize path ~4x slower than dense bf16 at small images.

Design (correctness-first; performance tuning is a later milestone):

  - The activation `a` (bf16 `[M, K]`) is the dense MMA A operand; the FP4 weight
    is the B operand with `transpose_b=True` (W is `[N, K]`, so
    `out = a @ dequant(W)^T`).
  - Each threadblock computes a `[BM, BN]` output tile. Per K-strip it
    cooperatively (1) copies the `[BM, BK]` A sub-tile into `a_smem` (bf16), and
    (2) DECODES the `[BN, BK]` packed-FP4 weight sub-tile + its FP8 block scales
    into `b_smem` (bf16). Both SMEM tiles are PLAIN row-major (NO ldmatrix
    swizzle), so the MMA consumer reads them with the un-swizzled per-tile
    `TensorCore.load_a` / `load_b` fragment loaders (the same ones the
    `test/gpu/layout/test_layout_mma.mojo` reference uses), never the swizzled
    warp-tile loaders. This keeps the SMEM write layout and the MMA read layout
    trivially consistent.
  - Dequant math is bit-identical to the Phase C materialize oracle
    (`fp4_dequant.mojo`): `w_bf16 = (E2M1_TO_FLOAT32[nibble] * |f32(scale)|)`
    cast to bf16, with the multiply done in f32. `decode_e2m1_to_f32` is
    bit-identical to indexing `E2M1_TO_FLOAT32`, so both paths feed the exact
    same bf16 values into the exact same bf16 MMA.
  - Accumulation is f32 (`get_accum_type[bf16]`); the f32 accumulators are cast
    to the output dtype in a bounds-guarded epilogue. `M` is dynamic; `N`, `K`
    are static model dims. Edge tiles on `M` and `N` (and a partial last K-strip)
    are zero-filled in SMEM / skipped at store, so a padded lane contributes a
    clean zero -- matching the dense reference's sum over `k < K`, `n < N`.

The NVFP4 per-tensor `weight_scale_2` scalar is applied OUTSIDE this kernel by
the graph lowering (a post-matmul multiply), identically to the Phase C path.
"""

from std.math import ceildiv

from std.gpu import (
    WARP_SIZE,
    barrier,
    block_dim,
    block_idx,
    lane_id,
    thread_idx,
)
from std.gpu.host import DeviceContext
from std.gpu.memory import (
    AddressSpace,
    async_copy,
    async_copy_commit_group,
    async_copy_wait_group,
)

from std.utils.numerics import get_accum_type

from layout import Coord, Idx, Layout, LayoutTensor, TensorLayout, TileTensor
from layout._utils import load_to_simd
from layout.tensor_core import TensorCore, get_fragment_size, get_mma_shape

from linalg.fp4_utils import (
    decode_e2m1_to_bf16,
    decode_e2m1_to_f32,
    NVFP4_SF_VECTOR_SIZE,
)


@__name(t"nvfp4_w4a16_fused_gemm_{c_type}")
def nvfp4_w4a16_fused_gemm_kernel[
    c_type: DType,
    c_layout: TensorLayout,
    a_layout: TensorLayout,
    bp_layout: TensorLayout,
    bs_layout: TensorLayout,
    BM: Int,
    BN: Int,
    BK: Int,
    WM: Int,
    WN: Int,
    SPLIT_K: Int = 1,
    NUM_STAGES: Int = 2,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin],
    a: TileTensor[DType.bfloat16, a_layout, ImmutAnyOrigin],
    b_packed: TileTensor[DType.uint8, bp_layout, ImmutAnyOrigin],
    b_scales: TileTensor[DType.float8_e4m3fn, bs_layout, ImmutAnyOrigin],
    work_space: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
):
    """One threadblock computes a `[BM, BN]` output tile of `a @ dequant(b)^T`.

    With `SPLIT_K > 1` the K reduction is parallelized across `block_idx.z`:
    the grid is `(grid_n, grid_m, SPLIT_K)` and each z-block accumulates only its
    contiguous 1/`SPLIT_K` slice of the K-strips (requires `SPLIT_K` to divide
    `ceildiv(K, BK)`), writing its `[BM, BN]` f32 partial to `work_space` at
    partition `z` (`work_space + z*M*N`). A separate reduction kernel then sums
    the `SPLIT_K` partials and casts to `c`. With `SPLIT_K == 1` (`work_space`
    unused / may be null) the f32 accumulators are cast straight into `c` as
    before -- no workspace, no reduction pass.

    `c` is `[M, N]` (output), `a` is `[M, K]` bf16, `b_packed` is `[N, K // 2]`
    uint8 (two E2M1 nibbles/byte, low nibble = even K), `b_scales` is
    `[N, K // 16]` fp8-e4m3 (block size 16 along K). The block reads `a` (SRAM)
    and decodes `b` (packed FP4 -> bf16 SRAM) one `BK`-deep K-strip at a time and
    accumulates the bf16 tensor-core MMA in f32.

    Parameters:
        c_type: Output element type (bf16 on the supported FLUX.2 path). Accum f32.
        c_layout: `TensorLayout` of `c`.
        a_layout: `TensorLayout` of `a`.
        bp_layout: `TensorLayout` of `b_packed`.
        bs_layout: `TensorLayout` of `b_scales`.
        BM: Threadblock M-tile height.
        BN: Threadblock N-tile width.
        BK: K-strip depth (cols of A / weight processed per cooperative stage).
        WM: Warp M-tile height (`BM % WM == 0`).
        WN: Warp N-tile width (`BN % WN == 0`).
        SPLIT_K: Number of K-slices reduced in parallel across `block_idx.z`
            (must divide `ceildiv(K, BK)`); `1` = no split (write straight to
            `c`), `> 1` = write f32 partials to `work_space`.
        NUM_STAGES: Depth of the cp.async software pipeline (buffers for the
            streamed A + packed-FP4 B). `2` = classic double buffer; `>= 3`
            prefetches deeper to hide more DRAM latency. The decoded-bf16 B is a
            single buffer regardless (produced and consumed within one strip).
    """
    comptime assert c.flat_rank == 2, "C must be rank-2 [M, N]"
    comptime assert a.flat_rank == 2, "A must be rank-2 [M, K]"

    # bf16 -> f32 tensor-core MMA (16x8x16 on NVIDIA).
    comptime accum_type = get_accum_type[DType.bfloat16]()
    comptime mma_shape = get_mma_shape[DType.bfloat16, accum_type]()
    comptime MMA_M = mma_shape[0]
    comptime MMA_N = mma_shape[1]
    comptime MMA_K = mma_shape[2]

    comptime assert BK % MMA_K == 0, "BK must be a multiple of MMA_K"
    comptime assert WM % MMA_M == 0, "WM must be a multiple of MMA_M"
    comptime assert WN % MMA_N == 0, "WN must be a multiple of MMA_N"
    comptime assert BM % WM == 0, "BM must be a multiple of WM"
    comptime assert BN % WN == 0, "BN must be a multiple of WN"

    comptime num_m_mmas = WM // MMA_M
    comptime num_n_mmas = WN // MMA_N
    comptime num_k_mmas = BK // MMA_K
    comptime num_warps_m = BM // WM
    comptime num_warps_n = BN // WN
    comptime num_threads = num_warps_m * num_warps_n * WARP_SIZE

    comptime frag_sizes = get_fragment_size[mma_shape]()
    comptime a_frag_size = frag_sizes[0]
    comptime b_frag_size = frag_sizes[1]
    comptime c_frag_size = frag_sizes[2]

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]())

    var tid = Int(thread_idx.x)
    var warp_id = tid // WARP_SIZE
    var lane = Int(lane_id())
    var warp_y = warp_id // num_warps_n
    var warp_x = warp_id % num_warps_n

    var block_row = Int(block_idx.y) * BM
    var block_col = Int(block_idx.x) * BN

    # `NUM_STAGES`-deep SMEM cp.async software pipeline: while the current K-strip
    # decodes + MMAs, the NEXT `NUM_STAGES-1` strips' A (bf16) and packed FP4
    # weight bytes stream in via cp.async. `a_smem` stages bf16 A (also the MMA A
    # source); `bp_smem` stages the RAW packed FP4 bytes (cp.async target, since
    # cp.async cannot decode) -- both `NUM_STAGES`-buffered, stage `s` at
    # `*.tile[...](s, 0)`. `b_smem` holds the decoded-bf16 B (MMA B source) and is
    # a SINGLE buffer: it is produced by `_decode_buf` and consumed by the MMA
    # within the same strip (barriers fence reuse), so it never needs staging.
    # All plain row-major (NO ldmatrix swizzle) so the un-swizzled per-tile
    # `load_a` / `load_b` fragment loaders read them directly.
    var a_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(NUM_STAGES * BM, BK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var bp_smem = LayoutTensor[
        DType.uint8,
        Layout.row_major(NUM_STAGES * BN, BK // 2),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()
    var b_smem = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(BN, BK),
        MutAnyOrigin,
        address_space = AddressSpace.SHARED,
    ].stack_allocation()

    var tc = TensorCore[
        accum_type, DType.bfloat16, mma_shape, transpose_b=True
    ]()

    # f32 accumulators: one MMA C-fragment per (m_mma, n_mma), persisted across
    # the whole K loop. Fragment `(m_mma, n_mma)` lives at row
    # `n_mma * num_m_mmas + m_mma` -- the index order `TensorCore.mma` writes.
    var c_reg = (
        LayoutTensor[
            accum_type,
            Layout.row_major(num_m_mmas * num_n_mmas, c_frag_size),
            MutAnyOrigin,
            address_space = AddressSpace.LOCAL,
        ]
        .stack_allocation()
        .fill(0)
    )

    # Per-K-sub-step register scratch for the A and B fragments.
    var a_reg = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(num_m_mmas, a_frag_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()
    var b_reg = LayoutTensor[
        DType.bfloat16,
        Layout.row_major(num_n_mmas, b_frag_size),
        MutAnyOrigin,
        address_space = AddressSpace.LOCAL,
    ].stack_allocation()

    # Vectorized cooperative-copy geometry: each thread moves ONE `RUN`-wide
    # contiguous run along K. `RUN == NVFP4_SF_VECTOR_SIZE` (16) so a run is
    # exactly one FP8 block -> one scale load per run, and `BK % RUN == 0` (BK is
    # a multiple of MMA_K = 16). Adjacent threads own adjacent K runs -> the
    # global loads are coalesced.
    comptime RUN = NVFP4_SF_VECTOR_SIZE
    comptime assert BK % RUN == 0, "BK must be a multiple of the NVFP4 block (16)"
    comptime runs_per_row = BK // RUN
    comptime a_runs = BM * runs_per_row
    comptime b_runs = BN * runs_per_row

    var num_k_tiles = ceildiv(K, BK)

    # ---- cp.async load of one K-strip into SMEM buffer `buf`. ----
    # A (bf16) streams straight into `a_smem[buf]` (MMA-ready); the packed FP4
    # weight bytes stream into `bp_smem[buf]` (decoded later, in `_decode_buf`).
    # Each thread owns one RUN-wide run per row (coalesced). OOB M/N rows and the
    # K-tail use cp.async zero-fill (`src_size` < copy size), so no OOB DRAM read
    # and padded lanes decode/accumulate a clean zero. Caller commits the group.
    @always_inline
    @parameter
    def _load_strip(buf: Int, kt: Int):
        var k0 = kt * BK
        var gmem_a = a.ptr.address_space_cast[AddressSpace.GLOBAL]()
        var gmem_bp = b_packed.ptr.address_space_cast[AddressSpace.GLOBAL]()
        var a_buf = a_smem.tile[BM, BK](buf, 0)
        var bp_buf = bp_smem.tile[BN, BK // 2](buf, 0)

        # A: one RUN-wide bf16 run == RUN//8 x 16-byte cp.async copies.
        for r in range(tid, a_runs, num_threads):
            var row = r // runs_per_row
            var col = (r % runs_per_row) * RUN
            var m_abs = block_row + row
            comptime for h in range(RUN // 8):
                var col_h = col + h * 8
                var kh = k0 + col_h
                var dst = a_buf.ptr + (row * BK + col_h)
                var src = gmem_a + (m_abs * K + kh)
                if m_abs < M and kh + 8 <= K:
                    async_copy[16](src, dst)
                else:
                    var valid = min(8, K - kh) if (
                        m_abs < M and kh < K
                    ) else 0
                    async_copy[16, fill = Scalar[DType.bfloat16](0)](
                        src, dst, src_size=Int32(valid * 2)
                    )

        # B: one RUN-wide run == one (RUN//2)-byte packed cp.async copy.
        for r in range(tid, b_runs, num_threads):
            var row = r // runs_per_row
            var col = (r % runs_per_row) * RUN
            var n_abs = block_col + row
            var k_abs = k0 + col
            var dst = bp_buf.ptr + (row * (BK // 2) + col // 2)
            var src = gmem_bp + (n_abs * (K // 2) + k_abs // 2)
            if n_abs < N and k_abs + RUN <= K:
                async_copy[RUN // 2](src, dst)
            else:
                var valid_nib = min(RUN, K - k_abs) if (
                    n_abs < N and k_abs < K
                ) else 0
                async_copy[RUN // 2, fill = Scalar[DType.uint8](0)](
                    src, dst, src_size=Int32((valid_nib + 1) // 2)
                )

    # ---- Decode packed FP4 SMEM buffer `buf` -> decoded-bf16 `b_smem[buf]`. ----
    # Each thread decodes its own RUN-wide run (loaded by its own cp.async, so no
    # cross-thread read here): one (RUN//2)-byte SMEM load, RUN nibbles expanded
    # branch-free, ONE FP8 block scale (RUN == one 16-block), decoded with
    # `decode_e2m1_to_f32[RUN]` and a per-run f32 scale multiply -- bit-identical
    # to the Phase C scalar oracle. OOB N-rows zero; K-tail per-element.
    @always_inline
    @parameter
    def _decode_buf(buf: Int, kt: Int):
        var k0 = kt * BK
        var bp_buf = bp_smem.tile[BN, BK // 2](buf, 0)
        var b_buf = b_smem  # single decoded-B buffer (see SMEM comment above)
        for r in range(tid, b_runs, num_threads):
            var row = r // runs_per_row
            var col = (r % runs_per_row) * RUN
            var n_abs = block_col + row
            var k_abs = k0 + col
            if n_abs < N and k_abs + RUN <= K:
                var bytes = bp_buf.load[width = RUN // 2](row, col // 2)
                var nib = SIMD[DType.uint16, RUN](0)
                comptime for j in range(RUN // 2):
                    var bj = UInt16(bytes[j])
                    nib[2 * j] = bj & UInt16(0xF)
                    nib[2 * j + 1] = (bj >> UInt16(4)) & UInt16(0xF)
                var scale_abs = abs(
                    b_scales[n_abs, k_abs // NVFP4_SF_VECTOR_SIZE][0].cast[
                        DType.float32
                    ]()
                )
                var dec = (decode_e2m1_to_f32(nib) * scale_abs).cast[
                    DType.bfloat16
                ]()
                b_buf.store[width=RUN](row, col, dec)
            elif n_abs >= N:
                b_buf.store[width=RUN](row, col, SIMD[DType.bfloat16, RUN](0))
            else:
                comptime for e in range(RUN):
                    var kk = k_abs + e
                    if kk < K:
                        var byte = bp_buf[row, (col + e) // 2][0]
                        var shift = UInt8(4) if (kk & 1) == 1 else UInt8(0)
                        var nibe = UInt16((byte >> shift) & UInt8(0xF))
                        var sc = abs(
                            b_scales[n_abs, kk // NVFP4_SF_VECTOR_SIZE][0].cast[
                                DType.float32
                            ]()
                        )
                        b_buf[row, col + e] = (
                            decode_e2m1_to_f32(SIMD[DType.uint16, 1](nibe)) * sc
                        ).cast[DType.bfloat16]()[0]
                    else:
                        b_buf[row, col + e] = Scalar[DType.bfloat16](0)

    var a_reg_v = a_reg.vectorize[1, a_frag_size]()
    var b_reg_v = b_reg.vectorize[1, b_frag_size]()
    var c_reg_v = c_reg.vectorize[1, c_frag_size]()

    # ---- Split-K: this z-block owns a contiguous slice of the K-strips. ----
    # SPLIT_K divides `num_k_tiles` (launcher-guaranteed), so every z-block does
    # exactly `strips_per_split` strips starting at `kt_start`; SPLIT_K==1 => the
    # whole K range (kt_start=0). Only the OWNED slice is accumulated here; the
    # cross-z sum happens in the reduction kernel.
    var strips_per_split = num_k_tiles // SPLIT_K
    var kt_start = Int(block_idx.z) * strips_per_split

    # ---- Software-pipelined mainloop (`NUM_STAGES`-deep cp.async). ----
    # Prologue: issue the first `NUM_STAGES-1` strips' async loads (one commit
    # group each). Each iter: prefetch the strip `NUM_STAGES-1` ahead, then wait
    # until the CURRENT strip's group is the oldest still pending (keep the newer
    # prefetches in flight; `remaining` shrinks as the slice drains), decode it,
    # MMA it. Two barriers per iter (after decode: A + decoded-B ready across the
    # block; after MMA: readers done before the single b_smem / this a_smem stage
    # is reused). `i` is the local strip index within this z-block's slice.
    comptime for s in range(NUM_STAGES - 1):
        if s < strips_per_split:
            _load_strip(s, kt_start + s)
            async_copy_commit_group()

    for i in range(strips_per_split):
        var kt = kt_start + i
        var cur = i % NUM_STAGES

        var pf = i + (NUM_STAGES - 1)
        if pf < strips_per_split:
            _load_strip(pf % NUM_STAGES, kt_start + pf)
            async_copy_commit_group()
        # Drain down to the strips still legitimately in flight ahead of `i`.
        # `cp.async.wait_group` needs an IMMEDIATE operand, so enumerate the
        # comptime possibilities and let the runtime `remaining` select one.
        var remaining = min(NUM_STAGES - 1, strips_per_split - 1 - i)
        comptime for rem in range(NUM_STAGES):
            if remaining == rem:
                async_copy_wait_group(Int32(rem))

        _decode_buf(cur, kt)
        barrier()

        # ---- MMA over this K-strip: bf16 A from stage `cur`, decoded-bf16 B
        # from the single b_smem.
        var a_warp = a_smem.tile[BM, BK](cur, 0).tile[WM, BK](warp_y, 0)
        var b_warp = b_smem.tile[WN, BK](warp_x, 0)

        comptime for k_mma in range(num_k_mmas):
            comptime for m_mma in range(num_m_mmas):
                var a_sub = a_warp.tile[MMA_M, MMA_K](m_mma, k_mma)
                a_reg_v[m_mma, 0] = rebind[a_reg_v.element_type](
                    load_to_simd(tc.load_a(a_sub))
                )
            comptime for n_mma in range(num_n_mmas):
                # transpose_b=True: the B mma tile is [MMA_N, MMA_K].
                var b_sub = b_warp.tile[MMA_N, MMA_K](n_mma, k_mma)
                b_reg_v[n_mma, 0] = rebind[b_reg_v.element_type](
                    load_to_simd(tc.load_b(b_sub))
                )
            # Accumulate all num_m_mmas x num_n_mmas MMAs for this K-sub-step.
            tc.mma(a_reg_v, b_reg_v, c_reg_v)

        # Fence the MMA readers before this buffer is overwritten two iters on.
        barrier()

    # ---- Epilogue: store f32 accumulators to C (bounds-guarded). ----
    # NVIDIA m16n8 f32 C-fragment layout: lane -> (g = lane // 4, t = lane % 4);
    # the 4 held values map to (row = g + (i//2)*8, col = 2*t + (i%2)) within the
    # 16x8 MMA output tile. This is the exact inverse of `TensorCore.store_d`'s
    # `distribute[row_major(8, 4)]` mapping, hand-rolled here so partial M/N edge
    # tiles can be predicated per element (store_d has no bounds check).
    # For SPLIT_K > 1 the block writes its f32 partial into workspace partition
    # `z` (`work_space + z*M*N`, row-major `[M, N]`); the reduction kernel sums
    # the SPLIT_K partitions and casts to `c`. For SPLIT_K == 1 the f32 accum is
    # cast straight to `c` (no workspace).
    # Plain-Int workspace-partition base (harmless 0 when SPLIT_K == 1; the null
    # `work_space` is only ever INDEXED in the SPLIT_K > 1 branch below).
    var ws_z_base = Int(block_idx.z) * M * N
    var g = lane // 4
    var t = lane % 4
    comptime for m_mma in range(num_m_mmas):
        comptime for n_mma in range(num_n_mmas):
            comptime idx = n_mma * num_m_mmas + m_mma
            var c_vals = load_to_simd(c_reg.tile[1, c_frag_size](idx, 0))
            comptime for i in range(c_frag_size):
                var row_in_tile = m_mma * MMA_M + g + (i // 2) * 8
                var col_in_tile = n_mma * MMA_N + t * 2 + (i % 2)
                var m_abs = block_row + warp_y * WM + row_in_tile
                var n_abs = block_col + warp_x * WN + col_in_tile
                if m_abs < M and n_abs < N:
                    comptime if SPLIT_K == 1:
                        c.store[width=1](
                            Coord(m_abs, n_abs),
                            SIMD[c_type, 1](c_vals[i].cast[c_type]()),
                        )
                    else:
                        work_space[ws_z_base + m_abs * N + n_abs] = c_vals[
                            i
                        ].cast[DType.float32]()


@__name(t"nvfp4_split_k_reduce_{c_type}")
def nvfp4_split_k_reduce_kernel[
    c_type: DType,
    c_layout: TensorLayout,
    SPLIT_K: Int,
    VEC: Int,
](
    c: TileTensor[c_type, c_layout, MutAnyOrigin],
    work_space: UnsafePointer[Scalar[DType.float32], MutAnyOrigin],
    M: Int,
    N: Int,
):
    """Sum the `SPLIT_K` f32 partials in `work_space` `[SPLIT_K, M, N]` -> `c`.

    Each thread reduces one `VEC`-wide contiguous run of one output row across
    the SPLIT_K partitions and stores the cast result. `N` is a multiple of
    `VEC`, so a run never straddles a row; the 1-D grid covers
    `ceildiv(M*N, VEC)` threads.

    Parameters:
        c_type: Output element type.
        c_layout: `TensorLayout` of `c`.
        SPLIT_K: Number of K-partitions to sum (partition `z` at `+ z*M*N`).
        VEC: Contiguous output elements reduced per thread (must divide `N`).
    """
    var total = M * N
    var base = (
        Int(block_idx.x) * Int(block_dim.x) + Int(thread_idx.x)
    ) * VEC
    if base >= total:
        return
    var m = base // N
    var n = base % N
    var acc = (work_space + base).load[width=VEC]()
    comptime for z in range(1, SPLIT_K):
        acc += (work_space + (z * total + base)).load[width=VEC]()
    c.store[width=VEC](Coord(m, n), acc.cast[c_type]())


def nvfp4_w4a16_fused_matmul_cuda(
    c: TileTensor[mut=True, ...],
    a: TileTensor[DType.bfloat16, ...],
    b_packed: TileTensor[DType.uint8, ...],
    b_scales: TileTensor[DType.float8_e4m3fn, ...],
    ctx: DeviceContext,
) raises:
    """Weight-only NVFP4 FUSED matmul: `out = a @ dequant(b)^T`, no materialize.

    Same operand contract as the Phase C `nvfp4_w4a16_matmul_cuda` (bf16
    activation / uint8 packed E2M1 weight / fp8-e4m3 block-16 scales), but the
    packed FP4 weight is decoded to bf16 in shared memory inside the GEMM
    mainloop (read 4-bit from DRAM, never materialized to a dense bf16 buffer).

    Args:
        c: Output `[M, N]` (bfloat16 on the supported FLUX.2 path).
        a: Activations `[M, K]` in bfloat16.
        b_packed: Packed NVFP4 weights `[N, K // 2]` in uint8 (two E2M1 nibbles
            per byte, low nibble first / even K).
        b_scales: FP8-E4M3 block scales `[N, K // 16]` (block size 16 along K).
        ctx: Device context.
    """
    # 4 warps/block (128 threads): BM/WM = 2 M-warps x BN/WN = 2 N-warps.
    # BK=32 kept (vs 64/128): at small M the kernel is latency/occupancy-bound,
    # not K-strip-count-bound; BK=64 doubles SMEM (18->37 KB) which drops
    # occupancy 3->1 block/SM and measured SLOWER everywhere, and BK=128
    # (~72 KB) exceeds the 64 KB/SM budget. The K-reduction is instead
    # parallelized across blocks via split-K (below).
    comptime BM = 64
    comptime BN = 64
    comptime BK = 32
    comptime WM = 32
    comptime WN = 32
    # cp.async pipeline depth. Kept at 2 (double buffer): a generalized 3-stage
    # was measured NEUTRAL-to-worse -- at small M this kernel is decode-COMPUTE-
    # bound, not cp.async-latency-bound, so deeper prefetch buys nothing. Paired
    # with the single-buffered decoded-B, 2 stages fit in ~14 KB SMEM -> 4 resident
    # blocks/SM (vs 3 when B was double-buffered), which helps at large M.
    comptime NUM_STAGES = 2

    # N (= c's free dim) and K (= a's contraction dim) are static model dims;
    # only M (tokens) is dynamic. Use the static extents to shape the grid.
    comptime static_N = type_of(c).static_shape[1]
    comptime static_K = type_of(a).static_shape[1]
    comptime assert (
        static_K % NVFP4_SF_VECTOR_SIZE == 0
    ), "K must be a multiple of the NVFP4 block size (16)"

    var M = Int(c.dim[0]())

    comptime num_threads = (BM // WM) * (BN // WN) * WARP_SIZE

    # grid.x -> N tiles, grid.y -> M tiles, grid.z -> K-split (split-K).
    var grid_n = ceildiv(static_N, BN)
    var grid_m = ceildiv(M, BM)

    # ---- Split-K: parallelize the K reduction across `block.z` to fill the GPU
    # AND shorten each block's serial K loop at small M. Without it, e.g.
    # N=3072/BN=64 = 48 blocks leaves ~140 of the 188 SMs idle while each block
    # walks all 96 K-strips -- the small-M latency wall. Target ~4x SM-count total
    # blocks (the kernel reaches ~4 resident blocks/SM, so 4x fills the machine
    # in one wave; 2x measured ~25% slower at M in {128,256,512}, and >4x only
    # adds reduction cost with no critical-path gain). Snap the needed factor
    # down to a power of two in {1,2,4,8,16} that DIVIDES the (static) K-strip
    # count `num_k_tiles`. At large M the base grid already oversubscribes the
    # GPU, so `desired` collapses to 1 (no workspace, no reduction -- the
    # original direct-to-C path).
    comptime num_k_tiles = ceildiv(static_K, BK)
    comptime SM_COUNT = 188
    comptime TARGET_BLOCKS = 4 * SM_COUNT
    var base_blocks = grid_n * grid_m
    var desired = ceildiv(TARGET_BLOCKS, base_blocks)

    @parameter
    def _run[SPLIT_K: Int]() raises:
        comptime kernel = nvfp4_w4a16_fused_gemm_kernel[
            type_of(c).dtype,
            type_of(c).LayoutType,
            type_of(a).LayoutType,
            type_of(b_packed).LayoutType,
            type_of(b_scales).LayoutType,
            BM,
            BN,
            BK,
            WM,
            WN,
            SPLIT_K,
            NUM_STAGES,
        ]
        comptime if SPLIT_K == 1:
            # Direct-to-C: no workspace, no reduction pass.
            ctx.enqueue_function[kernel](
                c,
                a.as_immut(),
                b_packed.as_immut(),
                b_scales.as_immut(),
                UnsafePointer[Scalar[DType.float32], MutAnyOrigin](
                    unsafe_from_address=Int(0)
                ),
                grid_dim=(grid_n, grid_m, 1),
                block_dim=(num_threads),
            )
        else:
            # f32 partials `[SPLIT_K, M, N]`; summed + cast to C, then freed.
            var work_space = ctx.enqueue_create_buffer[DType.float32](
                SPLIT_K * M * static_N
            )
            ctx.enqueue_function[kernel](
                c,
                a.as_immut(),
                b_packed.as_immut(),
                b_scales.as_immut(),
                work_space.unsafe_ptr(),
                grid_dim=(grid_n, grid_m, SPLIT_K),
                block_dim=(num_threads),
            )
            comptime VEC = 8
            comptime reduce_kernel = nvfp4_split_k_reduce_kernel[
                type_of(c).dtype,
                type_of(c).LayoutType,
                SPLIT_K,
                VEC,
            ]
            var total = M * static_N
            comptime red_block = 256
            var red_grid = ceildiv(total, VEC * red_block)
            ctx.enqueue_function[reduce_kernel](
                c,
                work_space.unsafe_ptr(),
                M,
                static_N,
                grid_dim=(red_grid),
                block_dim=(red_block),
            )
            _ = work_space^

    # Dispatch: largest valid split (power of two dividing `num_k_tiles`) up to
    # `desired`. The `comptime if` guards keep only feasible instances compiled.
    comptime if num_k_tiles % 16 == 0:
        if desired >= 16:
            _run[16]()
            return
    comptime if num_k_tiles % 8 == 0:
        if desired >= 8:
            _run[8]()
            return
    comptime if num_k_tiles % 4 == 0:
        if desired >= 4:
            _run[4]()
            return
    comptime if num_k_tiles % 2 == 0:
        if desired >= 2:
            _run[2]()
            return
    _run[1]()
