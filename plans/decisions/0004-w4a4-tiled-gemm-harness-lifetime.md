# ADR 0004 — D2/M2: SMEM-tiled W4A4 GEMM wired as the op path; the "codegen bug" was a test-harness DeviceBuffer-lifetime bug

Date: 2026-07-06
Status: Accepted (tiled kernel validated + deployed in the op; perf ladder continues)

## Context

The correctness-first W4A4 GEMM (one warp per [16,8] tile, global reads per strip,
no reuse) measured a FLAT ~29 TF/s — 3.3× (M=64) to 13.8× (M=4096) slower than bf16
(L2-flushed, device-timed, real Klein shapes; `bench_nvfp4_w4a4.mojo`). The win
requires a tiled GEMM. The first tiled attempt appeared to have an "impossible"
correctness bug: ~¼ of output cells garbage, provably-correct operands, lane 0
correct, and a debug `print` flipped pass/fail.

## Decision

1. Ship `_gemm_kernel_tiled` / `nvfp4_w4a4_matmul_cuda_tiled`
   (`max/kernels/src/linalg/matmul/gpu/nvfp4_w4a4_cuda.mojo`): threadblock [64,64]
   C-tile, per-BK=64 K-tile cooperative staging of packed-fp4 A+B and fp8 block
   scales into two typed u32 SMEM LayoutTensors (A rows then B rows), 4 warps ×
   [32,32] warp tiles of `mma.sync…mxf4nvf4.block_scale` MMAs reading fragments
   from SMEM. **All SMEM/accumulator access goes THROUGH the LayoutTensor values**
   (never an extracted raw pointer) — the idiom the validated fused kernel uses.
2. Route the op `mo.matmul.block.scaled.cuda.w4a4` to the tiled launcher
   (`builtin_kernels/linalg.mojo`). The naive launcher stays exported as the
   correctness oracle for the 3-way test.

## The bug that wasn't (root cause worth remembering)

The tiled kernel was **never wrong**. The standalone test harness dropped its input
`DeviceBuffer`s at their last program-order use — `TileTensor(buf.unsafe_ptr(), …)`
— which is BEFORE the launchers enqueue kernels. Mojo's ASAP destruction enqueued
stream-ordered frees ahead of the kernels; the stream-ordered allocator recycled
that memory into the launchers' internal transient quant buffers, whose kernel
overwrote the still-unread inputs. Symptoms perfectly mimic a miscompile
(shape/print/order-sensitive garbage; two kernels agreeing bit-exactly on wrong
values; framework tests passing while every standalone harness fails). The same
root cause had already produced the abandoned `nvfp4_w4a4_launch_test.mojo`
("unreliable references") a session earlier.

**Rule:** in standalone GPU tests/benches, end with `_ = buf^` keep-alives for
EVERY DeviceBuffer (after the final `synchronize()`). Inside a launcher,
`_ = transient^` right after its own enqueues is safe (free lands after those
kernels in-stream). See `nvfp4_w4a4_tiled_test.mojo`'s module docstring.

**Debug method that cracked it (artifact invariants, not prints):** a cell whose
inputs are identical across two shapes must be identical — it wasn't ⇒ the kernel
read different MEMORY, not different math. Host reference must be built from
device-produced artifacts (device quant bytes copied back, dot on host), because
two GPU kernels cross-checked against each other cannot arbitrate shared input
clobber.

## Validation

- 3-way test (`nvfp4_w4a4_tiled_test.mojo`): tiled ≡ naive **bit-exact (maxd 0.0)**
  on full/edge-M, multi-K-tile, tall-N; both match device-quant host ref to bf16
  out-rounding; device quant == host recipe **byte-exact**.
- Framework gate re-PASSED with the tiled op:
  `//max/tests/integration/nn:cuda_nvfp4_tests -k w4a4`.
- End-to-end: 1024² true-FP4 renders **2 s each (was 9–10 s naive)** and all 4
  PNGs **byte-identical** to the naive-kernel renders
  (`/root/w4a4_naive_kernel_renders/` on max-build).

## Measured position vs bf16 (RTX PRO 6000, L2-flushed / scheduler-log ms)

- GEMM: tiled 7–9× faster than naive at M=4096; up to 276 TF/s (bf16 ~380–410).
  vs bf16(flush): 1.49–1.89× at M=4096; **0.95× (WIN) at (36864,4096) M=64**.
  Small-M caveat: at N=4096 M=64 the 64-block grid underfills 188 SMs (tiled 2.1×
  slower than naive there).
- Render: bf16 1.49 s vs true-FP4 1.96 s mean → **1.32× from parity** end-to-end.

## Constraints

- Tiled GEMM requires K % BK(=64) == 0 and N % BN(=64) == 0; M dynamic (zero-fill
  staging + guarded epilogue). All 110 Klein NVFP4 Linears satisfy these
  (K ∈ {4096, 12288, 16384}; N ∈ {4096, 12288, 24576, 36864}).

## Next (task #22)

cp.async double-buffering of the K-loop, 128-wide tiles + swizzle, hoist/fuse the
per-call activation-quant alloc, small-M occupancy (split-K/N or smaller BM), then
re-run the A/B. Alternatives considered and rejected for now: ldmatrix/wgmma-style
restructure (sm_90 templates use a different MMA model; incremental rungs on the
proven kernel are lower-risk).
