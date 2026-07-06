# 0006 — W4A4 activation-quant placement: two-pass is optimal (fusion measured 3.5× worse)

Status: Closed (2026-07-06) — keep the two-pass structure; both alternatives measured.
Context: post-R1/R2/R3 the Klein render = 855 ms with the W4A4 GEMM path ~425 ms (~50% of
GPU work), of which the standalone activation-quant pass (one `_quant_act_kernel` launch +
packed-A DRAM round-trip per matmul, ~552/render) was estimated at ~35–40 ms. Two attempts
to recover it:

## R4 Step 1 — quant as its own graph op + identity-memo dedup: NO WIN (855→854 ms)

Split `mo.matmul.block.scaled.cuda.w4a4` into `mo.quant.act.fp4.cuda` +
`mo.matmul.block.scaled.cuda.w4a4.prequant` (same kernels), with a flatten-memo in
`nn/linear.py` and a quant-memo in `_cuda_w4a4_matmul` (Graph-attached, object-identity
keyed) so shared activations (q/k/v, modulation) quantize once. Also moved ~1104/render
`enqueue_create_buffer` scratch allocs to graph-managed tensors. **Measured: noise.** The
exec is GPU-bound and quant is small/overlapped; dedupable (shared-activation) quant is a
minor slice. KEPT anyway: cleaner structure, no regression, pixel-identical.

## R4 Step 2 — quant fused into the GEMM SMEM prologue: 3.5× WORSE (855→3001 ms)

`_gemm_kernel_tiled_fq` + `nvfp4_w4a4_matmul_cuda_tiled_fusedq`, gated
`MODULAR_NVFP4_FUSEDQ=1` (Python emits the all-in-one op; the Mojo op routes on the env).
Register-direct per-(row, 64-elem-strip) quant writing the exact packed/scale u32 image
into the existing `pk_s`/`sc_s` SMEM; B keeps cp.async; SMEM/BK unchanged.
**Bit-identical to the two-pass path on every test shape (fusedq-vs-tiled maxd 0.0,
including s2 folds and the N%128 fallback) and pixel-identical renders — but the render
went 855 → 3001 ms.**

**Root cause (design, not bug):** the GEMM grid re-runs the A-tile quant in EVERY
block-column: quant work ×(N/BN) = ×192–432 for the big Klein GEMMs, executed as scalar
`_round_e2m1_nibble` chains on warp-stalling synchronous global loads at prefetch time.
The two-pass design quantizes once (M×K) and streams 0.625 B/elem packed data through
cp.async — that asymmetry IS the design; fusing it away multiplies the work by the grid
width.

## Decision

- Two-pass (quant op + prequant GEMM) stays the production path (855 ms).
- The fusedq kernel stays in-tree, env-gated OFF, as documented negative-result machinery.
  **Do not enable `MODULAR_NVFP4_FUSEDQ`** at Klein-like N/BN grid widths.
- The GEMM-path lever is CLOSED: remaining recoverable there was the quant pass, and both
  placements are now measured. Further render wins live elsewhere (elementwise tail,
  VAE-conv gate, PNG level 0/async, serving floor ~750–780 ms).

## Lessons

- "Cheap redundant recompute" must be multiplied by the GRID REDUNDANCY FACTOR before
  believing it: per-block-column redundancy is ×(N/BN), i.e. ×hundreds on wide GEMMs.
- A bit-exact-on-first-try kernel can still be a perf disaster — correctness gates and
  perf gates are independent; always measure in-model before judging a fusion.
- Negative results measured cheaply (one test + one render A/B, ~30 min) are how a lever
  gets *closed* rather than lingering as "maybe someday".
