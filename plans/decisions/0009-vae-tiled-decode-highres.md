# Decision: Shared spatial tiled VAE decode to enable 4K image generation

**Status:** Parked (code kept, opt-in only; blend revised; disabled-by-default for Z-Image — see 2026-07-09 + 2026-07-08 updates) · **Date:** 2026-07-09 · **Area:** VAE / high-res

> **Update 2026-07-09 (c) — PARKED.** 4K-via-tiling on our models is shelved.
> Rationale: Klein/Z-Image use an **8× VAE**, so 4K unavoidably needs tiling
> (Z-Image = tiled decode; Klein's *fused* VAE OOMs at a single 128GB alloc and
> can't use this helper at all → 4K there is a client-side tiled img2img upscale,
> see [[klein-4k-tiled-img2img-upscale]]). The proper answer to *native* 4K is a
> different architecture with an aggressive-compression VAE — **Sana** (DC-AE
> 32× → 4K latent 128×128, no tiling), **PixArt-Σ**, or **UltraFlux** — not more
> tiling on an 8× VAE. The code stays in place and inactive (Z-Image opt-in via
> `MODULAR_VAE_ENABLE_TILING=1`; the helper is decoder-agnostic for future
> reuse); we're simply not investing further in the tiling path. Reopen only if
> a concrete need arises (e.g. wiring the helper into Ideogram4/Qwen-Image).

> **Update 2026-07-08 (b) — tiling is now OFF by default for Z-Image.** The
> auto-enable-at-threshold gating below was removed. Rationale: Z-Image is
> ~2K-native (Tongyi's whole family caps recommended output at 2048²), and true
> 4K needs a tiled *img2img upscale* (native-resolution tiles refined by the
> model), not merely a tiled *decode* of a 4K latent — so auto-tiling the decode
> is not a useful default (a 4K txt2img latent just yields a small subject on a
> big canvas). The helper + wiring are kept; tiling is opt-in via
> `MODULAR_VAE_ENABLE_TILING=1`. `MODULAR_VAE_TILE_THRESHOLD` was dropped.
> Default Z-Image decode is now plain `self.vae.decode` (byte-identical) at all
> sizes; a 4K request OOMs unless tiling is force-enabled. Verified: 1024²/2048²
> default = mean 0.0000 vs untiled.

> **Update 2026-07-08 (a) — blend moved to the host.** The device-side feather-blend
> described below (`F.pad` each weighted tile + accumulate + divide on device,
> baked `F.constant` windows) was **broken**: the compiled decoder reuses its
> output buffer across calls, so the lazy accumulation read the *last* tile for
> every tile (black/seamed output, mean diff ~77). The blend now runs on the
> **host** — each decoded tile is snapshotted with
> `np.from_dlpack(img_tile.cast(f32).to(CPU()))`, accumulated in numpy, and the
> result uploaded once via `F.constant`. Same tiling geometry, same gating, same
> accumulate-and-divide feather; only the accumulation substrate changed. Root
> cause + full debug trail in **lesson 0010**. Validated (below) with the host
> blend: 1024²/2048² forced-tile ≈ untiled (mean ~1.1, seamless), 4096² seamless
> at both tile=64 and default tile=256; ≤2K default gate byte-identical (mean 0).

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
