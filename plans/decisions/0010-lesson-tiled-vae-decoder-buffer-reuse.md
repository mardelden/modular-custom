# Lesson: Compiled decoder reuses its output buffer → tile the blend on the host

**Date:** 2026-07-08 · **Area:** VAE / high-res / experimental-tensor runtime

## What We Were Trying to Do

Feather-blend the per-tile outputs of the spatial tiled VAE decode
(`autoencoders_modulev3/tiling.py::tiled_decode`, ADR 0009) into one seamless
high-res image.

## What We Tried

| Approach | Result | Why it failed/worked |
|---|---|---|
| Device-side lazy accumulate: `out_sum += F.pad(img_tile * win)` over tiles, realize at end | **Failed** — black/seamed output; mean pixel diff **77** at 1024² (worse with more/smaller tiles) | Every tile's `img_tile` aliases the *same* reused decoder output buffer; the lazy accumulator holds no realized copy, so the final realize reads the **last** tile's pixels for all tiles |
| Realize the accumulator each iteration (`out_sum._sync_realize()`) | Failed — still mean 77 | Realizing the accumulator still resolves `img_tile` against the (now-clobbered) shared buffer |
| Realize each tile in place (`img_tile._sync_realize()`) | Failed — still mean 77 | The decode output is *already* "real"; it just points at the reused buffer. Realizing in place copies nothing off it |
| Copy each tile to host immediately (`np.from_dlpack(img_tile.cast(f32).to(CPU()))`), accumulate the blend in numpy, upload once with `F.constant` | **Worked** — mean **1.19** at 1024² (feather rounding only), byte-identical gate ≤2K, seamless 4K | The device→host copy captures each tile's content *before* the next decode overwrites the buffer; blend never touches the aliased buffer again |

## Root Cause

The compiled modulev3 `Decoder` graph (a *separate* graph from the eager
`decode_latents`) **reuses one output buffer across successive `vae.decode`
calls**. In eager experimental-tensor code, a decode result is a "realized"
tensor that *points at that shared buffer* — it is not a private copy. Folding it
into a lazy device-side graph (`F.pad`/`+`) keeps only a reference to the
buffer, not its bytes, so by the time the whole graph realizes, the buffer holds
the **last** tile. The batch-chunk decode loop elsewhere "gets away with it"
because `F.concat` consumes each realized tile into distinct output regions
within one realize, not a deferred cross-iteration accumulation.

The tell: a *debug read* that copied the tile to host (`.cast(f32).to(CPU())`)
made the output correct — proving the tiles decode fine and the bug is
buffer aliasing in the blend, not the decode or the blend math.

## Solution

Blend on the **host** (`tiling.py`): decode each tile on GPU (one at a time,
bounding peak memory), immediately `np.from_dlpack(img_tile.cast(DType.float32)
.to(CPU()))` to snapshot it, accumulate `out_sum`/`wt_sum` (separable
raised-cosine Hann feather, accumulate-and-divide) in numpy, then upload the
final `out_sum/wt_sum` once via `F.constant(..., float32).cast(out_dtype)`. Host
arrays are cheap (4K image ≈ 800 MB f32) and decode stays on GPU. This also
dropped the F.constant-window / bf16-accumulation machinery the device version
needed.

## How to Avoid in Future

- **Never accumulate compiled-graph outputs across an eager loop lazily.** A
  realized tensor from a compiled component may alias a reused output buffer;
  treat each call's result as valid only until the next call. Snapshot it (host
  copy, or a forced fresh-buffer copy) before reusing the callable.
- `_sync_realize()` does **not** copy off a reused buffer — it only forces
  materialization of an already-lazy value. It is not a defense against aliasing.
- When tiled/looped device output looks like "only the last piece survived,"
  suspect output-buffer reuse before suspecting the blend math. Confirm with a
  per-iteration host read (if that fixes it, it's aliasing).
