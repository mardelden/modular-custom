# Lesson: Z-Image 1024² denoise is GEMM-bound (~92%), not attention — explains why dynamic batching is ~0% ROI at production res

**Date:** 2026-07-08
**Area:** performance / diffusion

## What We Were Trying to Do

Before investing in Phase 2 (dynamic batching of concurrent requests) for
Z-Image, find out where the ~137 ms/denoise-step at 1024² actually goes and
whether a single-request optimization would beat batching's measured ~0-14%.

## What We Tried

| Approach | Result | Notes |
|----------|--------|-------|
| nsys / ncu per-kernel profile | **Blocked** | Neither installed on max-build; no CUDA toolkit (box uses vendored `/opt/nvidia-libs`). Install = yak-shave. |
| MAX `--trace-file` native profiler | **Blocked** | `max/profiler/oneshot` re-execs under nsys (`.nsys-rep`); still needs nsys. Transformer forward has **no internal Tracer scopes** → only a single "transformer" block. |
| Inspect attention impl for a naive-vs-fused win | **No win** | `z_image_modulev3/layers/attention.py` already uses fused `flash_attention_gpu` (not naive matmul+softmax). |
| **Step-diff × resolution sweep** (zero-install) | **Worked** | `(t[steps=8] − t[steps=4]) / 4` = pure per-step denoise (text-encode/VAE/PNG cancel); compare across res to split O(tokens) vs O(tokens²). |

## Root Cause / Finding

Per-step denoise cost scales **linearly with tokens**, not quadratically
(tokens = (res/16)²):

| res | tokens | per-step |
|-----|--------|----------|
| 512²  | 1024 | 34.7 ms |
| 768²  | 2304 | 73.4 ms |
| 1024² | 4096 | 138.6 ms |

4× tokens → 3.99× time. Least-squares fit `per_step = 3.11e-5·tok +
6.48e-10·tok²` ⇒ at 1024²: **~92% linear (W4A4 GEMMs + norms/RoPE/elementwise),
~8% attention (quadratic)**. Pure-linear model predicts 136.7 ms vs 138.6 ms
measured. FLOP check agrees: per block the GEMMs (qkv 3·d², out d², MLP ~8·d²)
dwarf attention (≈ seq·d) by ~12·d/seq ≈ 7.5× at 1024²/d≈2560.

**Consequence — this is *why* dynamic batching is ~0% at 1024²:** the workload
is W4A4-GEMM-bound and the tiled kernel is already saturated at M = 4096 (it
"beats bf16 at large M" — see [[w4a4-tiled-gemm-harness-lifetime]] / the NVFP4
work). Batching to M = 16384 adds no per-FLOP efficiency when already saturated;
it only helps sub-megapixel (512² M=1024 → ~14%), where the GEMM is below
saturation. Attention is only ~8%, so the "closed fp8-attention wall"
([[fp8-attention-sm120-status]]) is **irrelevant** to the main cost.

## Solution / What to do about speed

- **Do NOT expect batching (P2) to help at 1024²** — fundamental, not a wiring gap.
- **Attention optimization is a dead end here** (~8% of the step).
- The denoise is dominated by already-optimized W4A4 GEMMs; remaining *kernel*
  headroom is small (the ~few % activation-quant dedup noted in
  [[nvfp4-fused-d1-d2-status]]).
- The only **large** levers are algorithmic and are quality tradeoffs, not free:
  fewer steps (denoise is ~81% of render, linear in steps → 8→6 ≈ −20% render),
  or step-caching (TaylorSeer / first-block-cache; marginal at 8 Turbo steps).

## How to Avoid in Future

Before optimizing a diffusion transformer's attention or adding batching, run the
**step-diff × resolution sweep** first (`/root/sweep_denoise.py` on max-build). If
per-step time scales ~linearly with tokens, it's GEMM-bound → attention work and
batching are both low-ROI; look at GEMM efficiency (M-saturation) and step count
instead. Quadratic scaling would flip the conclusion (attention-bound), but only
kicks in at much higher resolutions (2048²+).
