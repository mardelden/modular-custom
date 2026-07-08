# Plan: Batching + multi-image variations for Z-Image (z_image_modulev3)

**Status:** Phase 1 Implemented (2026-07-08); Phase 2 Proposed (pending go/no-go)
**Date:** 2026-07-08

## Progress

- **Phase 1 — DONE & validated on max-build** (branch `feat/zimage-batching`,
  commit `3dc38cf8f1`). One-spot fix as predicted: the per-step timestep is
  tiled to `[batch_size]` when `num_images_per_prompt > 1` (new
  `_batched_timestep_tensors` in `pipeline_z_image.py`; `batch_size == 1` path
  byte-identical). Source-serve renders: `num_images=1` → 1 img/1.71 s;
  `num_images=2` → 2 **distinct** imgs/2.75 s; `num_images=4` → 4 **distinct**
  imgs/5.67 s; **no HTTP 500**. Images: `/Users/mardel/zimage-batch-renders/`
  (eyeballed n=2: two different sharp photorealistic foxes = real variations).
  Timing is sub-linear in N (fixed overhead amortized; denoise is compute-bound
  so the per-image compute still scales, matching the Klein finding).
- **Phase 2 — NOT started, deliberately.** Confirmed it is the bigger, modest-ROI
  change the plan flagged, with cross-prompt correctness risk — see the updated
  Phase 2 notes below. Left for an explicit go/no-go.

## Context

FLUX.2-Klein has two batching features that Z-Image (a separate `modulev3` pipeline)
never got, so both are broken on Z-Image:

1. **Variations** (`num_images>1` — N images in one request). Empirically **crashes**:
   `num_images=2` → HTTP 500, `ValueError: symbolic dimension 'batch_size' for input 2
   does not match prior uses`. The compiled-model args show the cause exactly:
   `hidden_states [2,4096,64]` ✓, `encoder_hidden [2,15,2560]` ✓ (embeds already
   broadcast at `pipeline_z_image.py:585-596`), but **`timestep [1]`** ✗ — the per-step
   timestep is never broadened to `num_images`. IDs are `[seq,3]` (batchless, correct).
   So variations is a **one-spot bug**, not a from-scratch build.

2. **Dynamic batching** (scheduler groups concurrent compatible requests, `MODULAR_
   PIXEL_MAX_BATCH_SIZE`). Not wired for Z-Image: its `prepare_inputs` takes a **single**
   `PixelContext` (Klein takes `list[...]`), it has no `supports_dynamic_batching` flag,
   and the **Qwen3 text encoder hard-asserts `batch_size==1`** (`qwen3_modulev3/
   text_encoder/model.py:165-191`, with a "TODO: lift this… if the Klein text-encoder"),
   which blocks different-prompt batches. Klein's win here was **modest** (compute-bound;
   parallel requests largely serialize, ~1.5× only sub-megapixel).

The two differ sharply in effort/ROI, so this is **phased** — P1 is a quick, high-value
feature fix; P2 is a bigger, modest-ROI throughput change gated on a text-encoder recompile.

## Approach

### Phase 1 — Variations (`num_images>1`)  [small, recommended]

Broadcast the per-step timestep to `num_images`. Everything else (latents from the
tokenizer, embeds broadcast, symbolic `batch_size` in `z_image.py:_base_input_types`,
symbolic-batch VAE decode) already handles the batch — the timestep is the only miss.

- **`z_image_modulev3/pipeline_z_image.py`** — the denoise loop builds the per-step
  timestep passed as `run_transformer(..., timestep=...)` (`run_transformer` at line 419;
  timestep built via `_prepare_timestep_broadcast` at line 945). Make the per-step tensor
  shape `[num_images_per_prompt]` (tile/broadcast the scalar), keyed by num_images in the
  cache. `num_images_per_prompt` is already on `ZImageModelInputs` (line 150).
- Verify the **output path** emits N images: `decode_latents` returns `[num_images,H,W,3]`
  uint8 and the diffusion pipeline splits per-request (Klein: `diffusion/pipeline.py:337`).
  Confirm Z-Image's decode + response builder return all N (fix if it takes only `[0]`).

### Phase 2 — Dynamic batching (concurrent requests)  [bigger, modest ROI]

**Confirmed scope after investigation (2026-07-08):** this is genuinely multi-part
and correctness-sensitive; recommend an explicit go/no-go before starting.

- **Executor restructuring is required, not optional.** Z-Image currently uses the
  base pipeline's `_pipeline_model` path, whose `prepare_batch` takes only
  `flat_batch[0][1]` (a single context — `diffusion/pipeline.py:404-406`) and
  hard-raises for >1 different requests unless the pipeline is an *executor* that
  advertises `supports_dynamic_batching` (`prepare_batch:386-394`). So Z-Image must
  either move to the executor path or have the base extended — mirroring Klein's
  ~130-line concat in `flux2_klein_executor.py:403-539`.
- **Qwen3 text-encoder batch-lift is the real blocker** (`qwen3_modulev3/
  text_encoder/model.py:155-205`). `__call__` squeezes 2D tokens and *requires*
  `batch_size==1` (167-170, 188-193); `attention_bias_from_attention_mask_array`
  builds a `[1,1,seq,seq]` additive mask. Batching different concurrent prompts
  needs: (a) 2D `[batch,seq]` tokens accepted, (b) a **batched** additive causal+
  pad mask `[batch,1,seq,seq]`, (c) **uniform per-batch token padding** (different
  prompts ⇒ different lengths; Klein pads text to 512), (d) the compiled Qwen3
  graph accepting batched inputs (**recompile**). Get the mask/padding wrong and
  prompts cross-contaminate — so this needs a numeric equivalence gate (batched
  encode vs per-request encode, per row) before trusting any render.
- **ROI is modest** (compute-bound; ~1.5× only sub-megapixel, per the Klein
  result [[klein-dynamic-batching]]). There is **no partial P2 that adds value**:
  same-prompt batching is already covered by P1 (`num_images`); only *different*-
  prompt batching is left, and that can't work without the text-encoder batch-lift.

Mirror Klein's mechanism (`flux2/flux2_klein_executor.py` + `serve/scheduler/`):

- **`pipeline_z_image.py`** — change `prepare_inputs(context)` → `prepare_inputs(contexts:
  list[PixelContext])`; validate compatibility (same H/W/steps/num_images/sigmas) and
  concatenate latents/timesteps/guidance across contexts (Klein `flux2_klein_executor.py:
  403-539`). Add `supports_dynamic_batching = True` so `diffusion/pipeline.py:max_batch_size`
  (206-228) returns `MODULAR_PIXEL_MAX_BATCH_SIZE`.
- **`qwen3_modulev3/text_encoder/model.py:165-191`** — lift the `batch_size==1` assert:
  make the encoder graph accept a batched additive mask + 2D/3D tokens (batch>1) and
  recompile. **This is the hard part** (different concurrent prompts ⇒ a real text-batch).
- The scheduler side is generic: `serve/scheduler/__init__.py:batch_key` +
  `one_shot_scheduler.py:_next_group` already group PIXEL_GENERATION requests; they just
  need the executor to accept a list (above). No scheduler change expected.

## Files to modify

| File | Phase | Change |
|------|-------|--------|
| `z_image_modulev3/pipeline_z_image.py` | 1 | per-step timestep → `[num_images]`; confirm N-image output split |
| `z_image_modulev3/pipeline_z_image.py` | 2 | `prepare_inputs(list)` + concat + `supports_dynamic_batching` |
| `qwen3_modulev3/text_encoder/model.py` | 2 | lift `batch_size==1`; batched mask + recompile |

## Verification

- **P1:** `POST /v1/responses` with `provider_options.image.num_images=2` (and 4) →
  returns that many **distinct** 1024² PNGs (different seeds per image), no 500. Eyeball
  they're varied + sharp. Single-image path unchanged (regression check).
- **P2:** serve with `MODULAR_PIXEL_MAX_BATCH_SIZE=4`; fire 4 concurrent different-prompt
  requests → the `OneShotScheduler` log shows **one** `Starting 4 request(s)` group (not 4
  serial), all 4 return correct images; measure wall-time vs serial (expect modest, sub-MP).
- Build/serve on max-build from source (`bazelw run`), gate with the render helpers.

## Risks / notes

- P1 is low-risk and localized; likely the whole "variations" fix.
- P2's text-encoder batch-lift needs a graph recompile + careful mask handling; ROI is
  modest (compute-bound), so it's worth confirming the throughput win is wanted before
  investing. Recommend landing **P1 first**, then deciding on P2.
- Do this on a branch off `main`; don't touch live max-serve; commit only when asked.
