# ADR 0002 — sm_120 NVFP4 fused GEMM: split-K + single-buffered-B close the small-M gap

Date: 2026-07-05
Status: Accepted
Context: Phase D-1 fused kernel (`nvfp4_w4a16_fused_cuda.mojo`) was CORRECT but
~8-14x slower than a realistic (DRAM-fed) bf16 GEMM at small M, and even slower
than the Phase C materialize path for square shapes. Root cause at small M:
occupancy/latency, not bandwidth. Goal of this round: close the gap with a
bounded, prioritized set of optimizations, oracle test staying green.

## Measurement fix (prerequisite)
The bench timed one weight back-to-back, so it stayed L2-resident and the
DRAM-bound baselines looked fake-fast (bf16 M=1 = 3.6 us = L2 BW). Fixed with an
L2 flush: rotate each timed iter over `R` weight copies whose total bytes swamp
L2 (256 MB packed target -> R=15-57), via `execution_time_iter`. Report bf16 both
ways: `bf16(resid)` (optimistic) and `bf16(flush)` (realistic; the number to
beat). All fused/materialize numbers are on rotated (realistic) weights.

## Decisions
1. **Keep BK=32.** Tried BK=64 (task step 1). It was SLOWER everywhere
   (M=64 N=K=3072: 184->192 us; M=2048: 765->1157 us): SMEM 18->37 KB drops
   occupancy 3->1 block/SM, and at small M the wall is latency/occupancy, not
   K-strip count, so halving strips while doubling per-strip work is a wash.
   BK=128 (~72 KB) exceeds the 64 KB/SM budget outright. **Rejected BK>32.**
2. **Split-K, on by default, M/shape-dependent (THE lever).** Grid gains a
   `block.z` K-split; each z-block accumulates its contiguous K-slice into an
   f32 workspace `[SPLIT_K,M,N]`; a vectorized reduction kernel sums + casts to
   C; all internal to the launcher (op/Python contract unchanged; SPLIT_K==1 =
   original direct-to-C path, no workspace/reduction). Heuristic: pick the
   largest `SPLIT_K in {1,2,4,8,16}` dividing the (static) K-strip count so
   total blocks reach **~4x SM count** (188 SMs -> target 752). 4x (not 2x) is
   right because single-buffered B (below) yields ~4 resident blocks/SM;
   2x left ~25% on the table at M in {128,256,512}. Higher (6-8x) hits the
   2-wave regime with no critical-path gain, only more reduction cost.
3. **Single-buffer the decoded-B SMEM.** It is produced by the decode and
   consumed by the MMA within ONE strip (barriers fence reuse), so it never
   needed staging. Dropping it from double->single frees SMEM (18->14 KB at
   2-stage) -> 4 blocks/SM -> measurably faster at large M (N=4096 M=2048:
   1335->1238 us, ~7%) and neutral at small M. Kept.
4. **Keep NUM_STAGES=2 (reject 3-stage).** Tried a generalized N-deep cp.async
   pipeline (task step 3). 3 stages was neutral-to-slightly-worse everywhere
   (M=64 N=K=3072: 47.2 vs 47.5 us). At small M we are decode-COMPUTE-bound, not
   cp.async-latency-bound, so deeper prefetch does nothing. The generalized-stage
   machinery is retained (set to 2) since it also carries the single-B change.

## Result (L2-flushed; fused vs realistic bf16(flush), µs)
| shape (N,K)      | M   | before | after | fused/bf16f before->after |
|------------------|-----|--------|-------|---------------------------|
| 3072,3072        | 64  | 184    | 41    | 10.4 -> 2.30              |
| 3072,3072        | 128 | 182    | 68    | 9.8  -> 3.65             |
| 3072,3072        | 256 | 210    | 120   | 8.5  -> 4.88             |
| 3072,12288       | 64  | 729    | 123   | 13.4 -> 2.27             |
| 12288,3072       | 128 | 310    | 231   | 5.3  -> 3.95             |
fused now BEATS the materialize path everywhere at M<=256 (fused/mat 0.19-0.73;
was >1 for square shapes).

## Honest verdict
Gap to bf16 CLOSED ~3-4x (from 8-14x to 2.3-5.7x in M in {64,128,256}) but NOT
beaten. Best case is M=64 at ~2.3x realistic bf16. Remaining wall is the
scalar-ALU FP4->bf16 decode (M-independent: M=1/8/32/64 all ~equal since same
grid; effective ~29-40 TFLOP/s vs bf16 ~68-89). Beating bf16 needs cheaper
decode (ILP/vectorization) or native FP4 tensor cores (D2) -- both out of scope
here (swizzle / warp-spec / dedicated GEMV deferred to next round).

## Correctness
Oracle `-k fused` (vs materialize, SPLIT_K=16 path) PASSED after every step.
Independent CPU-reference unit test (clean SPLIT_K=16, ragged-mn SPLIT_K=2,
k-tail SPLIT_K=2 partial-strip) PASSED. No throttle, GPU uncontended.
