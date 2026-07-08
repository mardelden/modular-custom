# Lesson: ComfyUI/SwarmUI "CFG" ≠ MAX `guidance_scale` for distilled models

**Date:** 2026-07-08
**Area:** pipelines / diffusion / Z-Image

## What We Were Trying to Do

Match Z-Image-Turbo (distilled, bf16) render quality on MAX to what it produces in
ComfyUI/SwarmUI, where it is sharp at **steps=8, CFG=1**.

## What We Tried

Klein-style 1024² renders of a fixed prompt on max-build (sm_120), varying steps and
guidance:

| Approach | Result | Why it failed/worked |
|----------|--------|----------------------|
| steps=8, `guidance_scale=1.0` (the naive "same as ComfyUI") | **Soft/blurry**, 3.4 s | `guidance_scale=1` in MAX still runs 2-pass CFG; distilled model under-converges at low steps |
| steps=8, `guidance_scale=3.5` (the framework default) | Blurrier, 3.4 s | Even more CFG → more softening on a model not trained for CFG |
| steps=20, `guidance_scale=1.0` | Sharp, 8.2 s | CFG *does* converge — but only with ~2.5× the steps (kills the "turbo" point) |
| Reduce resolution to 512/768 @ guidance=1 | Sharp, faster | Real effect, but a **red herring** — masked the actual cause |
| steps=8, **`guidance_scale=0.0`** | **Sharp**, 1.86 s | CFG path OFF → single pure-conditional pass; matches ComfyUI CFG=1 |
| steps=4, `guidance_scale=0.0` | Sharp, 1.06 s | Distilled model is genuinely few-step when CFG is off |

## Root Cause

MAX's Z-Image tokenizer gates classifier-free guidance on
`do_zimage_cfg = guidance_scale > 0.0`
(`max/python/max/pipelines/architectures/z_image_modulev3/tokenizer.py`). So any
`guidance_scale >= 1` keeps a **2-pass** CFG (a positive **and** a negative forward
per step). Z-Image-Turbo is distilled to run **without** CFG, so the guided result is
mushy at low step counts and only sharpens around ~20 steps.

The trap is a **parameter-semantics mismatch between frameworks**:

- In ComfyUI/SwarmUI (and most UIs) "**CFG = 1**" means guidance **disabled** — the blend
  `uncond + cfg·(cond−uncond)` collapses to `cond`, and the UI skips the uncond pass
  → single forward.
- In MAX, `guidance_scale = 1` is still `> 0`, so it **keeps** the 2-pass CFG path.

Therefore the true MAX equivalent of ComfyUI **CFG=1** is **`guidance_scale = 0`**, not 1.
The 2-pass vs 1-pass difference is directly visible in timing: guidance>0 ≈ 3.4 s vs
guidance=0 ≈ 1.86 s at 1024²/8-steps (≈ half — one forward instead of two).

## Solution

Serve/request Z-Image-Turbo on MAX with **`guidance_scale = 0`** and **steps ≈ 8** (4 is
also clean). At guidance=0, 1024² is sharp — no need to lower resolution or raise steps.
Recommended model defaults: `guidance_scale=0`, `default_num_inference_steps=8`, size
request-driven.

## How to Avoid in Future

- When a **distilled/turbo** model looks sharp in ComfyUI/SwarmUI at low steps but **soft
  in MAX at the "same" settings**, suspect CFG is on. Set `guidance_scale=0` first, before
  blaming resolution or step count.
- Don't trust "same settings" across frameworks — the **parameter semantics differ**. UI
  "CFG=1" = guidance off = MAX `guidance_scale=0`.
- Use **timing as a probe**: if a render roughly halves when you drop guidance, you were
  running a 2-pass CFG; that confirms the model wants CFG off.
- "Blurry at high resolution" can be a **symptom of CFG under-converging at low steps**,
  not a true resolution limit — toggle guidance to 0 to disambiguate before concluding the
  model needs a smaller canvas.
- MAX gates CFG on `guidance_scale > 0.0` (a hard on/off at 0), not on `>= 1`; there is no
  "neutral 1.0" — 1.0 is fully on.
