# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

Scope: `max/python/max/pipelines/` — the pipeline layer that turns a request
into model execution. This document focuses on the **pixel-generation
(diffusion) pipelines** (FLUX2, Klein, Ideogram4, Qwen-Image, Wan, Z-Image),
because that subsystem's control flow is spread across a tokenizer, a context
object, an executor, and several compiled graph components, and is not obvious
from any single file. For general MAX SDK build/test guidance see
`max/CLAUDE.md`; for repo-wide commands see the root `CLAUDE.md`.

## Two pipeline families

Every model is registered as a `SupportedArchitecture` (see each
`architectures/<name>/arch.py`) and discovered through `lib/registry.py`. The
`task` field splits the world in two:

- **`PipelineTask.TEXT_GENERATION`** — LLMs. KV-cache, sampling, paged
  attention. Not covered here.
- **`PipelineTask.PIXEL_GENERATION`** — diffusion image/video models:
  `flux2`, `flux2_modulev3`, `ideogram4`, `qwen_image`, `qwen_image_edit`,
  `wan`, `z_image_modulev3`. No KV-cache; the "loop" is the denoising loop.

A `SupportedArchitecture` wires together three collaborators that you will
almost always touch together:

| Field           | Role                                                    |
| --------------- | ------------------------------------------------------- |
| `tokenizer`     | request → `PixelContext` (CPU-side prep, host tensors)  |
| `pipeline_model`| the **executor**: `prepare_inputs` + `execute` on device|
| `context_type`  | `PixelContext` — the data carrier between the two       |

## Pixel-generation data flow

```
OpenResponsesRequest
   │  Flux2Tokenizer.new_context()            (architectures/flux2/tokenizer.py,
   │    - tokenize prompt (+ negative)         lib/pixel_tokenizer.py)
   │    - scheduler → (timesteps, sigmas)
   │    - _prepare_latents → noise + latent_image_ids
   │    - _build_text_ids (padded length)
   ▼
PixelContext                                   (context/… ; carries numpy host
   │                                            tensors: tokens, mask, sigmas,
   │                                            latents, latent_image_ids,
   │                                            text_ids, guidance_scale, …)
   │  Executor.prepare_inputs(contexts)         → *ExecutorInputs (TensorStruct)
   │    - patchify+pack latents, build attn bias, .to(device)
   ▼
Executor.execute(inputs)                        (architectures/flux2/
   │    1. text encoder  → prompt_embeds         flux2_klein_executor.py)
   │    2. (img2img) image encoder → latents
   │    3. denoising loop (see below)
   │    4. VAE decode → uint8 image
   ▼
Flux2ExecutorOutputs(images=…)  → (B, H, W, 3) uint8 on CPU
```

The executor is a `PipelineExecutor[PixelContext, *Inputs, *Outputs]`.
`prepare_inputs` is host-side and batch-aware in signature but current
FLUX2/Klein executors assert `batch_size == 1`.

## Compiled-component pattern (the key structural idiom)

An executor is **not** one graph. It composes several `CompiledComponent`
subclasses (`lib/compiled_component.py`), each owning one compiled graph:
`ImageEncoder`, `VaeDecoder`, `DenoiseCompute`, `DenoisePredict`,
`CfgCombineComponent`, plus a text encoder.

Two compile modes:

- **Eager**: a component calls `_load_graph(graph)` in its own `__init__` and
  compiles immediately.
- **Deferred (preferred for FLUX2 Klein)**: all components are built into one
  shared `max.graph.Module` (`graphs_module`), then compiled together in a
  single `session.load_all(module, weights_registry=…)` call, after which the
  executor calls `_attach_compiled_model(models)` on each. This is why
  `Flux2KleinExecutor.__init__` merges every component's `_pending_weights`
  into one registry and checks for **weight-key collisions** — components
  compiled together share one namespace, so FQNs must be unique (e.g. the VAE
  decoder prefixes its BatchNorm stats with `decoder_`).

When adding a component to a deferred executor: give its graph a unique name,
prefix its weights, add it to the `components` list, and it will be compiled
and attached automatically.

## The denoising loop (FLUX2 Klein)

`_run_denoising_loop` runs, per step `i`:

1. `noise_pred = denoise_compute(latents, …, timestep=sigmas[i], guidance)` —
   the transformer forward, returns raw `noise_pred`.
2. If CFG: run it again with the negative prompt, then
   `cfg_combine(pos, neg, guidance_scale)` = `neg + scale·(pos − neg)`.
3. `latents = denoise_predict(latents, noise_pred, dt=sigmas[i+1]−sigmas[i])`
   — the Euler step `latents + dt·noise_pred` (flow-matching).

The `denoise_compute` / `denoise_predict` **split** (transformer vs. Euler
step as separate graphs) exists so **TaylorSeer** caching can skip the
transformer on some steps and reuse a Taylor-predicted `noise_pred` while
still applying the scheduler step every iteration. TaylorSeer state is
per-CFG-stream (`state_pos`, `state_neg`) and gated by
`DenoisingCacheConfig`; defaults live on the executor
(`_DEFAULT_TAYLORSEER_*`). See `diffusion/cache.py`, `diffusion/taylorseer.py`.

## Scheduler ↔ executor contract (subtle; split across two files)

`FlowMatchEulerDiscreteScheduler.retrieve_timesteps_and_sigmas`
(`diffusion/schedulers/scheduling_flow_match_euler_discrete.py`) returns a
`sigmas` array of length `num_steps + 1` ending in `0.0`. The executor
(`_prepare_scheduler`) then derives:

```python
timesteps = sigmas[:-1]            # value fed to the transformer as "timestep"
dts       = sigmas[1:] - sigmas[:-1]   # per-step Euler delta (negative)
```

Key facts that bite:

- The transformer expects the **raw sigma in [0,1]**; it multiplies timestep
  **and** guidance by `1000.0` *internally* (`flux2.py`), so pass unscaled
  sigma / guidance_scale, kept in float32 until inside the graph.
- `use_empirical_mu=True` is forced on for FLUX2 (Klein) via the tokenizer's
  `scheduler_config_overrides`; the empirical-μ formula is BFL's
  `flux2/sampling.py`. `use_dynamic_shifting` comes from the model's HF
  scheduler config — if it flips, the whole sigma curve changes.
- `image_seq_len` (packed tokens = `(latent_h/2)·(latent_w/2)`) feeds μ, so a
  resolution change shifts the schedule.

## FLUX2 / Klein specifics & invariants

- **Text is padded to `FLUX2_TEXT_SEQ_LEN = 512`** (`architectures/flux2/
  arch.py`). The denoiser was trained with the full padded text sequence
  present in unmasked joint attention — the pad tokens act as **register /
  scratch-pad** tokens. `prompt_embeds` length and `text_ids` length must
  match or the transformer's rope/attention concat mismatches. Under-padding
  text is a known cause of weak, prompt-ignoring output.
- **RoPE ids are 4-axis `[t, h, w, l]`**, `axes_dims_rope = (32,32,32,32)`,
  `rope_theta = 2000` (not FLUX.1's 10000). Image tokens vary on `h,w`; text
  tokens vary on `l`. `latent_image_ids` are built over the **packed** grid
  (`latent_h/2 × latent_w/2`).
- **CFG structure** mirrors the V3 Klein pipeline: positive forward,
  optional negative forward, optional `cfg_combine` blend, then one Euler
  step. CFG is active iff a negative prompt is present **and**
  `guidance_scale > 1.0` **and** the checkpoint is not distilled.
- **Distilled Klein** (`manifest.metadata["is_distilled"]`) runs exactly
  `_DISTILLED_KLEIN_NUM_STEPS = 4` steps and **disables CFG** regardless of
  request; other step counts raise / are warned. The transformer "guidance"
  embedder is auto-disabled when its weights are absent.
- **VAE latent denorm is per-channel BatchNorm** (`x·√(var+eps)+mean`), not a
  scalar scaling/shift factor. All-zero BN stats collapse output to a
  washed-out near-grey image, so `VaeDecoder` validates them at load. Packing
  (`_patchify_and_pack`) and the VAE's unpatchify must stay exact inverses
  (channel layout `[c, a, b]` over 2×2 patches).

## `flux2` vs `flux2_modulev3` (don't confuse them)

There are two FLUX2 implementations:

- `architectures/flux2/` — **Graph API / ModuleV2** stack
  (`max.graph`, `max.nn.layer.Module`). This is the `Flux2Executor` /
  `Flux2KleinExecutor` path registered as `Flux2Pipeline` /
  `Flux2KleinPipeline`.
- `architectures/flux2_modulev3/` — the **experimental-tensor / modulev3**
  stack (`max.experimental.tensor`, operator-syntax style per
  `max/CLAUDE.md`). The Graph-API components' docstrings reference this and an
  internal `pipeline_flux2_klein.py` as the behavioural reference, but that
  reference file is not part of this open-source subset.

When editing "the FLUX2 pipeline", confirm which stack the served model uses
(`arch.py` `pipeline_model=`) before changing math — the two are independent.
