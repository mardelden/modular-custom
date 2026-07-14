# Plan: Whisper v2 — KV-cached decode → micro-batching → `max serve`

**Status:** In progress — A + B done & GPU-validated; C partially wired.
**Date:** 2026-07-13
**Branch:** `feat/whisper-word-ts` (worktree `modular-whisper`, continuing from v1 @ `0c7adadf13`)
**Order (user-fixed):** (A) KV cache → (B) micro-batching → (C) max serve → (D) validation/deploy.

## Progress / RESUME POINT (HEAD `049fe9aa58`)

- **Phase A (KV cache)** — DONE, GPU-validated on large-v3: cached==no-cache, **2.33× decode speedup**. Commit `3095a00c11`.
- **Phase B (micro-batching)** — DONE, GPU-validated: batch parity + **2.2× @ B=16**. Commit `177842363a`.
- **Phase C (max serve)** — IN PROGRESS. Committed + import-verified so far: `PipelineTask.SPEECH_TO_TEXT`/`InputModality.AUDIO`, `SpeechToTextContext`, `SpeechToTextOutput`+`TranscribedWord`, `SpeechToTextInputs`, `SpeechToTextPipeline` (executor-path), all package exports (`20e48adf19`, `049fe9aa58`).
- **REMAINING (build against the LIVE serve loop; box confirmed ready):**
  1. **`WhisperExecutor`** (`architectures/whisper/executor.py`) — adapt `transcribe.py` (encoder + cross_kv + cached-decode loop + per-row align) into `PipelineExecutor[SpeechToTextContext, WhisperExecInputs(TensorStruct), Any]`; `supports_dynamic_batching=True`; returns a plain `WhisperExecResult(texts, tokens, words)`; builds graphs in `__init__` from `manifest["main"]` (factor the weight-load helpers out of `transcribe.py`).
  2. **`WhisperServeTokenizer`** (`architectures/whisper/serve_tokenizer.py`) — `PipelineTokenizer`; `new_context(request)`: audio bytes → mel + `num_content_frames` + SOT prompt → `SpeechToTextContext`; ≤30.5s else `InputError`→400. Extend `audio.load_audio` to accept bytes/BytesIO.
  3. **`arch.py` + `WhisperArchConfig`** (`get_max_seq_len()->448`) + lazy entry in `architectures/__init__.py` (`_LazyArch("WhisperForConditionalGeneration", ".whisper.arch", "whisper_arch")`).
  4. **Registry** (`lib/registry.py`): `get_pipeline_for_task` SPEECH_TO_TEXT→`SpeechToTextPipeline`; early SPEECH_TO_TEXT branch in `retrieve_factory` (try full `resolve()`, fall back to `models.resolve()`).
  5. **Serve plumbing**: `serve/scheduler/__init__.py` branch → `OneShotScheduler` w/ `batch_key=(language, word_timestamps)`; `serve/worker_interface/zmq_interface.py` response type; `serve/api_server.py` factory (`GeneralPipelineHandler`); **route** `POST /v1/audio/transcriptions` in `serve/router/openai_routes.py` (UploadFile+Form; verbose_json/json; InputError→400, cancelled→500).
  6. **BUILD.bazel**: new `speech/BUILD.bazel` (no-mypy, ignore_unresolved_imports); add speech dep to `lib` + `serve/scheduler`; whisper BUILD += context/modeling.
  7. **Boot** `max serve --model openai/whisper-large-v3 --devices gpu` on max-build (self-serve over ssh; scp in-progress files; **keep Klein down**, kill by port) and iterate on real tracebacks; then Phase D gates (serve_gate.sh + chunked_client_example.py).
- **Env ready on box**: worktree `/opt/modular-whisper`, `/root/whisper-venv` (has serve stack + `python-multipart 0.0.32`), `HF_HOME=/mnt/models/huggingface`, `PYTHONPATH=/opt/modular-whisper/max/python`, cap `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE`. **NEXT = WhisperExecutor.**

## Context

Whisper v1 on MAX is done and hardware-validated (transcript exact vs HF/faster-whisper; word timestamps median 0ms). But it's a Python class + CLI — other teams can't just point at a URL — and it's slow at scale: the decode loop **recomputes the full token prefix every step** (no KV cache) and serves **one request at a time**. The fleet chunks long audio client-side into ≤30s windows, so the server sees floods of independent chunk requests → cross-request **micro-batching** is the throughput lever, and **`max serve`** (the generic `OneShotScheduler` dynamic batching that already powers Klein pixel-gen) is where it belongs. Target endpoint: OpenAI-compatible `POST /v1/audio/transcriptions` (multipart upload; `verbose_json` + word timestamps) so teams reuse existing OpenAI clients.

Why not a thin HTTP wrapper: it would be single-stream (no batching). Whisper isn't a `max serve` task today (no SPEECH task/executor/scheduler path), so getting serve's batching means *building* the task integration — this plan. Biggest single-stream win is the KV cache (independent of batching), so it goes first.

## Verified mechanics (explored + spot-checked against shipping code)

- **Dynamic-offset cache writes ship today**: `autoencoders/autoencoder_kl_wan.py:349-407` (`_compile_write_chunk_graph`) — mutable `BufferType` (symbolic dims OK), write offset as a **CPU-resident `int64` scalar** graph input, tuple slice form `output_buf[..., (slice(start, start+ops.shape_to_tensor(chunk.shape)[i], 1), sym_dim), ...] = chunk`. Copy this idiom exactly. GPU `BufferType` state also ships in `nemotron_h/state_cache.py`; mamba `ssm_cache.py` is the per-request state-buffer management pattern. `nn/kv_cache` is paged/LLM-only → hand-roll Whisper's 448-token cache.
- **`OneShotScheduler` is generic** (`serve/scheduler/one_shot_scheduler.py:44-200`): needs `context.request_id`, a `batch_key`, and `pipeline.execute(inputs) -> dict[RequestID, Output]`; singletons are never delayed; a batch exception cancels the whole group (`result=None`) → route must 500. The pixel branch in `serve/scheduler/__init__.py:84-141` (incl. our CFG-solo `batch_key`) is the direct template.
- **`PipelineExecutor` In/Out TypeVars are `TensorStruct`-bound** (`lib/pipeline_executor.py:35-68`; fields must be Tensor/Buffer, enforced at class-def). So `execute()` returns a **plain frozen dataclass** (texts/tokens/words) — a commented deviation; the speech package uses diffusion's `no-mypy` + TYPE_CHECKING + `ignore_unresolved_imports` pattern.
- **zmq**: plain dataclasses are msgspec-native (like `TextGenerationOutput`) — no pydantic registry edits. Mel ndarray rides OOB.
- **No registry collision** for `"WhisperForConditionalGeneration"`; `python-multipart` 0.0.20 + `soundfile` already in `bazel/pip/requirements/uv.lock`.
- **Risk**: `retrieve_factory`'s `pipeline_config.resolve(arch)` runs LLM validation for archs with a `"main"` model. Branch SPEECH_TO_TEXT early after arch resolution; fall back to `models.resolve()` if full resolve trips on the whisper config.

## Phase A — KV-cached decode (batch-1) → Gate K

Static shapes throughout; f32.
- **Self-KV cache**: per-layer mutable `BufferType(f32, ["batch", 20, 448, 64])` × 32 layers × {K,V} = 64 buffer inputs (large-v3: H=20, hd=64, max_target_positions=448).
- **Cross-K/V**: precomputed **once per request** by new `build_cross_kv_graph` (encoder states → `cross_k`,`cross_v` `[32,"batch",20,1500,64]`, immutable, static per-layer index).
- **`build_decoder_cached_graph`** (prefill T_new=4 and step T_new=1 share it; symbolic `t_new`): inputs `tokens[B,t_new]`, `positions`, host-built additive `mask[1,1,t_new,448]` over the full 448 key slots (mask does the truncation — no dynamic-length reads), `cache_len int64 [] on CPU`, cross_k/v, 64 cache buffers → **last-position logits `[B,vocab]`** (slice before the tied-head matmul). Per layer: write new K/V via the wan tuple-slice idiom, `ops.buffer_load`, attend over all 448.
- **Host loop** (`transcribe.py`): keep encoder output + cross-KV **device-resident** (kills v1 per-step re-upload); allocate 64 zeroed cache buffers per transcription; keep v1 no-cache path behind `use_kv_cache=False` (not compiled unless requested). Align graph untouched.

**Files:** `whisper/decoder.py` (+`WhisperDecoderCachedSelfAttention`, cached cross-attn, `WhisperDecoderCached`; same weight FQNs → adapter unchanged), `whisper/graph.py` (+2 builders), `whisper/transcribe.py` (cached loop, tokens/s), `standalone/check_buffer_gpu.py` (**Gate A0**, run first), `standalone/parity_kvcache.py` (**Gate K**), `run_gates.sh`.

## Phase B — micro-batching (standalone) → Gate Batch

- Fix align gather batch-0 hardcode (`decoder.py` ~226) → `[:, head]` → `[n_align,"batch",T,1500]`; host indexes per row.
- `transcribe_batch(paths)`: stacked mels; **lockstep greedy** — finished rows keep feeding EOS so `cache_len` stays one scalar (no per-row cache lengths); per-row suppress/argmax host-side; stop when all done. One batched teacher-forced align pass (EOS-padded); per-row DTW.
- `standalone/bench_batch.py`: batch-N == batch-1 per clip + throughput table B∈{1,4,8,16}.

## Phase C — `max serve` integration → Gates C1/C2/C3

**New msgspec-friendly dataclasses:** `SpeechToTextContext` (mel `[128,3000]` f32, num_content_frames, duration_s, language, prompt_tokens, word_timestamps, request_id, status/is_done) in `pipelines/context/context.py`; `TranscribedWord` + `SpeechToTextOutput` (mirror `TextGenerationOutput`) in `context/outputs.py`; `SpeechToTextInputs(PipelineInputs)` in `modeling/types/pipeline_variants/speech_to_text.py`; `PipelineTask.SPEECH_TO_TEXT` in `modeling/types/task.py`.

**New `pipelines/speech/` package:** `SpeechToTextPipeline` — structural copy of `PixelGenerationPipeline` (flatten batch → executor prepare/execute → zip rows to request IDs; `max_batch_size` from `MODULAR_WHISPER_MAX_BATCH_SIZE`, default 8; diffusion BUILD pattern: `no-mypy`, `ignore_unresolved_imports`, TYPE_CHECKING).

**Whisper arch additions:** `executor.py` (`WhisperExecInputs(TensorStruct)`; `WhisperExecutor(PipelineExecutor[...])`, `supports_dynamic_batching=True`, builds 4 graphs from `manifest["main"]`, Phase-B loop inside `execute()`, returns plain `WhisperExecResult`; skip align when words off), `serve_tokenizer.py` (`TranscriptionRequest` + `WhisperServeTokenizer.new_context`: bytes→waveform via `audio.load_audio(BytesIO)`, ≤30.5s else `InputError`→400, mel+SOT→context), `arch.py` (`name="WhisperForConditionalGeneration"`, task=SPEECH_TO_TEXT, encodings {"float32"}); factor weight-loading helpers out of `transcribe.py` for reuse.

**Plumbing (small, each mirrors an existing branch):** lazy arch in `architectures/__init__.py` (module `.whisper.arch`); `lib/registry.py` `get_pipeline_for_task` + early SPEECH_TO_TEXT branch in `retrieve_factory` (resolve() fallback); `serve/scheduler/__init__.py` branch → `OneShotScheduler`, `batch_key=(language, word_timestamps)`; `serve/worker_interface/zmq_interface.py` response-type; `serve/api_server.py` pipeline-factory (`GeneralPipelineHandler`); **new route** `serve/router/openai_routes.py` `POST /audio/transcriptions` (`UploadFile`+`Form`; 25MB cap; `InputError`→400, cancelled→500; `verbose_json`={task,language,duration,text,words[]}, `json`={text}). BUILD: new `speech/BUILD.bazel`; add speech dep to `lib` + `serve/scheduler`; whisper BUILD += context/modeling. `model.py` stays parked.

## Phase D — validation + deploy handoff

- `standalone/serve_gate.sh`: start serve → health → curl-vs-CLI words diff → 16-way flood (all-200 + worker log batch>1) → teardown **by port listener** (never `pkill -f` over ssh).
- `standalone/chunked_client_example.py`: client-side long-file chunking + concurrent posts + word-offset stitching (fleet-integration example + throughput demo).
- Regression each phase: v1 gates (a)–(d) pass; pixel archs still import/resolve.
- Deploy loop unchanged: team pulls branch only; we run over ssh (`PYTHONPATH=/opt/modular-whisper/max/python` namespace pkg covers serve, `/root/whisper-venv`, `HF_HOME=/mnt/models/huggingface`). **Never run whisper serve while Klein serve is up** (shared GPU). Final ship = `packaging/build_overlay.sh` wheel — deploy-team note.

## Sub-phase gates (branch stays green after each)

| Step | Gate |
|---|---|
| A0 | dynamic-offset `buffer_store_slice` round-trip on sm_120 |
| A1 | **K**: cached path transcript+words token-identical to v1; tokens/s speedup; v1 gates pass |
| B1 | **Batch**: B∈{4,8,16} == batch-1 per clip; scaling table |
| C1 | registry smoke (`retrieve_pipeline_task`), `import python_multipart`, pixel archs import |
| C2 | serve boots; curl == CLI words; json+verbose_json; >30s→400 |
| C3 | 16-flood all-200 with observed batch>1; throughput |
| D | full `run_gates.sh` green; chunked-client demo |

## Pitfalls

- **Memory (f32, large-v3):** self-KV ≈147MB/row, cross-KV ≈492MB/row → B=8 ≈ 5.1GB caches + ~11GB weights; B=16 needs ~10GB more. bf16 = named follow-up. Don't compile the legacy no-cache graph in serve. Cap `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE` on the shared GPU.
- **Lockstep EOS** keeps `cache_len` scalar — junk K/V of finished rows only affects their own (ignored) logits.
- Flood gate needs ≥2× max_batch_size concurrency to see batch>1 (batching forms from backlog; singletons never delayed).
- Words over zmq: dataclasses only. Cancelled batch → `result=None` → route 500 (no AttributeError).
- Tiny first at every gate, then large-v3.

## Deferred

bf16 · beam/temperature fallback · language autodetect · server-side long-audio · continuous batching · per-row cache lengths · fused align-head Mojo kernel.
