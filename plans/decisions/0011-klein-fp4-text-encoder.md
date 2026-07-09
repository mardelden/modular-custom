# Decision: Allow a quantized (NVFP4) Qwen3 text encoder for FLUX.2-Klein

**Status:** Implemented (branch `feat/klein-fp4-text-encoder`) · **Date:** 2026-07-09 · **Area:** FLUX.2 / quantization / VRAM

## Context

Klein 9B NVFP4's transformer is already fp4 (5.4 GB), but its **Qwen3-8B text
encoder was pinned to bf16 (16 GB)** — 3× the transformer and the dominant
resident-weight cost (~21.5 GB total). The Klein executor deliberately forced the
text encoder (and VAE) to bf16 whenever the transformer was NVFP4
(`flux2_klein_executor.py`), and the `Qwen3TextEncoderKlein` model built only
plain (unquantized) `Linear` layers. BFL themselves run the encoder FP8 by
default, and `nvidia/Qwen3-8B-NVFP4` (6 GB) is a stock-Qwen3 quantization — so a
quantized encoder is legitimate, it just wasn't wired.

## Decision

Make the Qwen3 text-encoder variant quantization-capable and let the executor
honor an explicitly-configured encoder encoding. **Additive**: `quant_config=None`
keeps the bf16 path byte-identical; only an explicit
`--model-override text_encoder.{model_path,quantization_encoding}` opts in.

- **Reuse, don't reinvent.** `parse_quant_config` already supports modelopt
  fp4/fp8; the shared `nn/linear.py::Linear` already stores fp4 weights as uint8
  + scales. The fix is to *thread* a `QuantConfig` into the encoder's
  `EncoderAttention`/`Qwen3MLP` `Linear`s and parse it in the model.
- **Compute dtype stays bf16 for fp4** — embeddings, norms and activations stay
  bf16; only the projection *weights* go fp4 (mirrors the transformer's
  `_model_dtype = bf16 if fp4`).
- **Drop KV-cache scales.** Modelopt checkpoints ship `*.k_proj.k_scale` /
  `*.v_proj.v_scale` (`kv_cache_scheme`); the encoder-only module has no KV
  cache, so skip them like `norm.weight`/`lm_head.weight` (this was the *only*
  weight-adapter change needed — no nibble/scale-layout transforms).
- **Executor** honors `text_encoder_entry.quantization_encoding` before the bf16
  guard; the VAE stays bf16 (hardcoded separately in `vae_decoder.py`).

## Result (validated on max-build, sm_120)

- **Loads + renders** (1024 txt2img, 2048 img2img) with the NVFP4 encoder; the
  modelopt checkpoint dropped in cleanly after the k_scale/v_scale skip.
- **Quality identical to bf16** — same prompt/seed, mean pixel diff 8.89,
  visually indistinguishable → Klein's encoder is stock Qwen3 and fp4 doesn't
  hurt prompt adherence/composition.
- **Resident weights ~21.5 GB → ~11.5 GB** (fp4 encoder 6 GB vs 16 GB bf16).

## Consequences / caveats

- The ~10 GB saving is **weight footprint / activation headroom**, not a smaller
  reserved pool: the MemoryManager floors at ~34 GB for this workload (bf16 and
  fp4 both load at `MM=15%` showing 34 GB). See ADR 0009-adjacent memory
  `gpu-mem-pool-not-load`.
- Only **fp4** is cleanly supported today; **fp8** would need compute-vs-weight
  dtype separation (fp8 weights but bf16 embeddings/norms) — deferred (fp4
  quality was sufficient).
- Kept on an isolated branch; not merged to `feat/zimage-batching`/`main`.
