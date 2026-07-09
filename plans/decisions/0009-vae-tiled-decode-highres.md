# Decision: Shared spatial tiled VAE decode to enable 4K image generation

**Status:** Accepted · **Date:** 2026-07-09 · **Area:** VAE / high-res

## Context

4096² generation OOM'd in the VAE decode: the latent unpacks to spatial NCHW
`[1,16,512,512]` and the first conv/GroupNorm intermediate alone needs ~47 GB
(`region_432`, `CUDA_ERROR_OUT_OF_MEMORY`) — a single contiguous block the
allocator couldn't get alongside the ~15 GB of resident weights. Profiling
(`0008`) had already shown that ≥2K is attention-bound in the *denoise*, but the
VAE was the hard wall preventing 4K from running at all. No spatial tiling
existed anywhere (only batch-axis and Wan temporal chunking).

## Decision

Add a **shared, decoder-agnostic spatial tiling helper**
(`autoencoders_modulev3/tiling.py::tiled_decode`) and wire it into Z-Image's
`decode_latents`, gated on latent edge size.

- **Why it works on one compiled graph:** the modulev3 `Decoder.input_types`
  declares `latent_height`/`latent_width` as **symbolic** dims (`vae.py:782`), so
  any tile size runs on the *same* compiled decoder — no per-tile recompile.
  `decode_latents` is eager Python and `self.vae.decode` is a *separate* graph,
  so each tile materializes and **frees its conv intermediates before the next**
  — that is what bounds peak memory.
- **Algorithm:** even-spaced overlapping H/W tiles → decode each → **feather-blend**
  with a separable raised-cosine (Hann) window, **accumulate-and-divide**
  (`sum(w·tile)/sum(w)`), correct for arbitrary overlap, full weight at the true
  image border (no dark frame). Pure-functional `F.pad`+sum accumulation (avoids
  `__setitem__` mutation-ordering risk). Window baked as float32 numpy →
  `F.constant` → `.cast(bf16)` (F.constant requires value dtype == requested).
- **Gating:** auto-enable when `max(latent_h,latent_w) > MODULAR_VAE_TILE_THRESHOLD`
  (default 320 latent ≈ 2560 img). Below it, the call is a literal
  `self.vae.decode(latents)` → **byte-identical**, so ≤2K is untouched.
  `MODULAR_VAE_ENABLE_TILING` forces on/off; `MODULAR_VAE_TILE_SIZE`/`_OVERLAP`
  tune geometry (latent px). Defaults 256/32/320.

## Result (validated on max-build, sm_120, NVFP4 W4A4)

- **4096² renders** (was OOM): 91.5 s, peak **89 GB** (the *denoise* is now the
  ceiling; each VAE tile ≈12 GB), 27 MB PNG. **No seams** — border diff ratio
  0.88–1.10× median (visual + quantitative).
- **≤2K unchanged**: 1024² 1.72 s, 2048² 9.32 s (tiling auto-off, plain decode).

## Consequences / follow-ups

- The helper is decoder-agnostic (`decode_fn` + `upsample` only). Reuse for the
  other image VAEs, easiest→hardest: **Ideogram4** (same modulev3 `Decoder`) →
  **Qwen-Image** (own decoder, same algorithm) → **Klein** (fused graph
  unpatchifies packed→spatial *inside* the graph, uses `Buffer`s → needs a
  spatial entry point) → **Wan** (framewise temporal; spatial tiling composes on
  the inner per-frame decode). Re-export `tiled_decode` from
  `autoencoders/__init__.py` once ≥2 pipelines use it.
- 4K is now **denoise-memory-bound (~89 GB)**, not VAE-bound — pushing past 4K
  (or 4K with num_images>1) needs denoise/attention memory work next, not more
  VAE tiling.
