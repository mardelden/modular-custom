# ADR 0003 — D2: native sm_120 FP4 tensor cores (W4A4), spike-confirmed

Date: 2026-07-05
Status: Accepted (spike GO; W4A4 accuracy gate pending)

## Context
D1 (fused in-mainloop fp4→bf16 decode + split-K) is correct and **beats the deployed
materialize NVFP4 everywhere at small M** (up to ~5×), but stays **2.3–5.7× off bf16**
(L2-flushed) — the wall is the scalar-ALU decode, and D1's ceiling is ≈ **bf16 parity**
(after decode it runs the same bf16 MMA). To make quantized *clearly faster than bf16*
we need the hardware to do the decode inline: native FP4 tensor cores.

## Decision
Build **D2 = native sm_120 FP4 GEMM** using the warp-level block-scaled instruction
`mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`.
Chose **W4A4** (fp4 activation, ~4× bf16 rate) over W4A8 for max speed, with an
image-quality gate.

## Spike result (GO)
A single instruction **assembles for `sm_120a`** via **raw inline PTX** (the
`llvm.nvvm.mma.block_scale` intrinsic is ptxas-rejected — CUTLASS #3227) and is
**numerically exact** vs a host reference (uniform, SFA-only, SFB-only, full-structured).
Validated encoding: A=4×u32/thread, B=2×u32, C=4×f32, `scale_vec::4X` (block-16),
`ue4m3` scales, `{byte,thread}` selectors `{0,0}`, full per-lane fragment/scale layout
(`nvfp4_mma_spike.mojo`).

## Key constraints discovered
- **sm_120 uses warp-level `mma.sync`, NOT tcgen05** (which is sm_100-only). TN layout
  only, cluster 1×1×1, ~2× Ada fp8 rate.
- **Native FP4 MMA is FP4×FP4** → no bf16×FP4 path → **must quantize the activation**
  (W4A4/W4A8). This is a numerics change on a weight-only-quantized model → **image
  quality must be validated** (fp4 act = 1 mantissa bit; diffusion is sensitive).
- The **activation quantizer** (`quantize_dynamic_block_scaled`, `fp4_quantization.mojo`)
  is **also SM100-gated** → D2 must add an sm_120 activation quantizer, not just the MMA.

## Sequencing (spike-first discipline)
1. ✅ MMA spike (matmul feasibility) — GO.
2. ⏳ **W4A4 image-quality pre-check** (fp4-activation fake-quant render vs known-good) —
   BEFORE building the GEMM, to avoid weeks of work on a scheme that may not hold quality.
3. Build GEMM (wire the atom: smem→reg loads, scale loads, K-loop) + sm_120 activation
   quantizer → numeric test → bench vs bf16.

## Alternatives
- **W4A8** (fp8 act via `mxf8f6f4`): safer accuracy, ~2× bf16 rate — the fallback if W4A4
  degrades quality.
- **Ship D1** (beats materialize, zero accuracy change): a real, deployable win
  independent of D2; keep as the safe baseline.
- **Stay Phase C materialize**: rejected (the small-M regression stands).

## Consequences
- Multi-week: GEMM wiring + sm_120 activation quantizer + accuracy validation + integration.
- If W4A4 quality fails → fall back to W4A8, or ship D1 and defer.
- The spike atom + encoding are the validated foundation; the rest is engineering.
