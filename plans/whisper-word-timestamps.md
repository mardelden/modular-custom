# Plan: Whisper speech-to-text with word-level timestamps on MAX

**Status:** Implemented + on-hardware validated (max-build GPU, large-v3) — **all gates green**: transcript exact vs HF/faster-whisper; word timestamps median 0ms vs faster-whisper; encoder/decoder parity pass on the GPU-TF32 cosine gate.
**Date:** 2026-07-13
**Branch:** `feat/whisper-word-ts` (worktree `modular-whisper`, based on `feat/klein-fp4-text-encoder` @ `2a52ee64df`)
**Scope:** standalone ≤30s-window PoC. NO `max serve` integration this iteration.
**Line numbers below are as-of base SHA `2a52ee64df`.**

## Implementation status & results

All code is written. Validated locally on **Mac CPU** (backend-agnostic correctness only — real numerics/perf run on Metal/MLX/CUDA via the deploy team):

| Check | Result (whisper-tiny, CPU) |
|---|---|
| Gate (a) encoder vs HF | **PASS** — max_abs 8.4e-05, cos 1.000000, shapes match, 67 weights strict-loaded |
| Gate (b) decoder teacher-forced vs HF | **PASS** — logits max_abs 1.1e-04, argmax 100%; alignment probs max_abs 3.2e-06 |
| `timing.py` DTW / median / word-split | **PASS** — synthetic peaks: monotonic, brackets peaks, reconstructs phrase |
| End-to-end pipeline wiring | **PASS** — safetensors load + 3 graphs + greedy + align + timing compose; well-formed words |

**Bugs fixed during bring-up (beyond B1–B5):** (1) encoder input frame dim must be **static** (`2*max_source_positions`) or the positional add is non-inferable; (2) `timing.dtw` tie-break must match openai-whisper exactly (strict `<`, ties fall through to "advance frame") — a diagonal/up preference collapses tokens onto one frame; (3) **word-timestamp off-by-one** — cross-attention at decoder position `p` localizes the *predicted* token `seq[p+1]`, so the alignment rows must be sliced `[sot_len-1:-2]` (not `[sot_len:-1]`); the wrong slice put every timestamp ~one token (~240ms) late. All fixed + commented in code.

**On-hardware validation (max-build GPU sm_120, `openai/whisper-large-v3`, real 5.86s LibriSpeech clip):**

| Gate | Result |
|---|---|
| (c) transcript | **PASS** — exact char-for-char match vs HF **and** faster-whisper (17/17 words): "Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel." |
| (d) word timestamps vs faster-whisper | **PASS** — \|Δ\| **median 0ms**, p95 120ms (was +240ms) after the off-by-one fix; last word's end bounded at the EOT onset (5.34 vs 5.84, fw 5.08). |
| (a)/(b) numeric parity on GPU | **PASS** — via the GPU-TF32 cosine gate: encoder cos **0.999993**; decoder argmax **100%**, logit cos 1.0, align cos 0.999999. (Raw magnitude drift is TF32, not a correctness issue — proven by the exact transcript.) |
| (e) portability | runs on max-build GPU (large-v3) and Mac CPU (tiny). |

Environment note: deploy team's venv is `/root/whisper-venv/bin/python` (`import max` = branch base SHA, no ABI drift); cap the MemoryManager (`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE≈24GiB`) so it doesn't over-reserve against other GPU tenants; `HF_HOME=/mnt/models/huggingface`. faster-whisper's CT2 needs CUDA 12 (`libcublas.so.12`) which the CUDA-13 box lacks → run the fw acceptance ref on CPU.

**Open follow-ups (deferred):** only the deferred-scope items below (KV-cached decode, bf16, >30s chunking, language autodetect, serve integration, …). The GPU-TF32 gate recalibration and the last-word-end trim are **done**.

## Deploy vs validation (split responsibilities)

**Deploy team — DEPLOY ONLY (do not run the gates).** Make `feat/whisper-word-ts` runnable on max-build and hand back the two paths below; we drive validation:

1. Worktree on the box (do **NOT** switch `/opt/modular-custom`'s own branch — the Klein fp4 dev serve runs from it):
   ```bash
   git -C /opt/modular-custom worktree add /opt/modular-whisper feat/whisper-word-ts
   ```
2. A Python that imports the custom `max` — the baked venv `/root/wheeltest-baked/bin/python` or the published wheelhouse wheel (branch base ⇒ no ABI drift) — plus the parity deps: `torch`, `transformers>=5.12,<5.13`, `datasets`, `soundfile`, `numpy`. (`faster-whisper` only for the optional acceptance ref — a throwaway venv is fine.)
3. Confirm HF access (large-v3 downloadable/cached) and drop a ≤30s test clip on the box.

**Deliverable back to us:** the worktree path (`/opt/modular-whisper`), the python path, and a confirmed `import max`.

**Us — VALIDATION + iteration.** Once deployed, we run the gates and own correctness:
```bash
PY=<their python> MODEL=openai/whisper-large-v3 DEVICE=gpu AUDIO=<clip.wav> \
  PYTHONPATH=/opt/modular-whisper/max/python \
  /opt/modular-whisper/max/tests/integration/architectures/whisper/standalone/run_gates.sh
```
`PYTHONPATH=<worktree>/max/python` overlays the whisper package onto the installed `max`. We read the gate (a)–(e) PASS/FAIL + the `|Δ|` word-timing stats; on a failure we fix on the branch and ask the deploy team to **redeploy** (that's their only follow-up). Local Mac dev overlays the changed files onto a nightly-wheel venv (`https://whl.modular.com/nightly/simple/`).

## Context

Goal: word-timestamped Whisper on MAX, for hardware portability (the existing fleet is CT2/CUDA-locked faster-whisper for word timestamps + vLLM-whisper for segment-only throughput). A prior napkin analysis argued MAX was the wrong tool because (1) MAX doesn't run Whisper and (2) fast attention kernels don't materialize the cross-attention matrix that word timestamps need. Codebase exploration overturned both premises **for this repo**:

- A **Whisper encoder already exists in-tree** at `max/python/max/pipelines/architectures/whisper/`, parked mid-bring-up (its integration test is skipped: *"We decided to postpone finishing Whisper bring up. Should debug if we come back to it."*).
- Because we author the graph, capturing cross-attention softmax probs is just an extra graph **output** — T5 already demonstrates the pattern (`t5/t5.py:426-454`, `output_attentions=True`).

So the real work is: fix the parked encoder's latent bugs, add the decoder + autoregressive generation + a teacher-forced alignment pass, and port the openai-whisper DTW timing algorithm to host-side numpy.

**Decisions locked in:** primary model `openai/whisper-large-v3` (`openai/whisper-tiny` for fast bring-up); float32 first; greedy decode only; ≤30s single window (long-audio chunking deferred); A/B against faster-whisper large-v3. Implement in the worktree on top of the cumulative custom stack (klein/nvfp4/zimage) so its wheel is a superset.

## Starting state (verified)

Encoder-only arch: `encoder.py`, `graph.py`, `model.py`, `weight_adapters.py`. **Not registered** in the arch registry (`architectures/__init__.py`) — leave it unregistered (standalone PoC, no serve).

### The 5 latent bugs that stalled bring-up (all confirmed vs code + real checkpoint headers)

| # | Bug | Evidence |
|---|-----|----------|
| **B1** | `huggingface_config.n_heads` does not exist on `WhisperConfig` (only `num_attention_heads`/`encoder_attention_heads`) → `AttributeError` on first real build | `encoder.py:34` |
| **B2** | Weight adapter never strips the `model.` prefix. Real keys are `model.encoder.conv1.weight`, `model.decoder.…`. Strict `load_state_dict` fails. Never caught because the skipped test used an **empty** state dict (`test_encoder.py:105`, "TODO: Need to construct state dict") | `weight_adapters.py:21-33` |
| **B3** | MLP mapping `".fc.": ".mlp."` never matches — HF keys are `…layers.0.fc1.weight`; the substring `.fc.` doesn't occur. Needs `.fc1.→.mlp.fc1.`, `.fc2.→.mlp.fc2.` | `weight_adapters.py:29` |
| **B4** | Conv layout three-way contradiction: `graph.py:36` declares channels-first `[batch, num_mel_bins, seq]`; `encoder.py:154-173` builds `Conv1D` with default `permute=False` (expects channels-**last** + MAX `(k,in,out)` weight layout, `nn/conv.py:307-331`); the post-stem permute is commented out (`encoder.py:204`); the adapter never transposes conv weights. Any combination currently loses. | multiple |
| **B5** | `graph.py:37` pins the mel input to `DeviceRef.CPU()` regardless of execution device → wrong for GPU | `graph.py:34-38` |

**B4 fix direction (least surprise, exact HF mirroring):** keep graph input channels-first `[B, n_mels, 3000]` (exactly what `WhisperFeatureExtractor` emits), build both convs with `permute=True` (loads PyTorch `(out,in,k)` weights raw, no adapter transpose — `nn/conv.py:411-417`), and re-enable the commented post-stem permute as `ops.permute(inputs_embeds, [0, 2, 1])` before adding positional embeddings.

### Checkpoint facts (from safetensors headers — do not hardcode, read from config/checkpoint)

- **No `k_proj.bias` anywhere** (self- or cross-attention). `q_proj`/`v_proj`/`out_proj` do have biases.
- **Tied LM head**: no `proj_out.weight` key → reuse `embed_tokens` weight for the output matmul (don't invent a Linear; strict load would fail + wastes ~265MB f32).
- **Encoder positional table is sinusoidal but stored** (`model.encoder.embed_positions.weight`, `(1500,d)`) → load it, don't synthesize.
- **Decoder positional table is learned** (`model.decoder.embed_positions.weight`, `(448,d)`) → index by position.
- **large-v3 checkpoint is fp16** (adapter must `.astype(f32)`); **large-v3 `num_mel_bins=128`** vs tiny's 80; vocab differs (tiny 51865 / v3 51866) → everything reads from config.

### Building blocks already available

- `nn/conv.py` `Conv1D` (lines 307-502): `permute=True` consumes PyTorch `(out,in,k)` + channels-first I/O; supports kernel=3, stride=2, padding=1 → 3000→1500.
- Materialized-softmax attention pattern: `pixtral/vision_encoder/attention.py:106-125` (`scores = q@kᵀ; softmax(scores*scale+mask); @v`), and `t5/t5.py:426-454` returns `attn_weights` under `output_attentions`.
- Deps already pinned: `torchaudio`, `transformers>=5.12,<5.13` (`bazel/pip/requirements/pyproject.toml`).
- Standalone exec pattern: `Graph` → `InferenceSession(devices=[dev])` → `session.load(graph, weights_registry=state_dict)` → `model(inputs)` (already in `whisper/model.py:load_model` + the skipped test).

## Target architecture — 4 graphs + host orchestration

| Graph | Inputs | Outputs | Runs |
|---|---|---|---|
| **encoder** (fix existing) | mel f32 `[B, n_mels, 3000]` on exec device | encoder states f32 `[1,1500,d]` | 1× |
| **cross_kv** (new) | encoder states `[1,1500,d]` | `cross_k`,`cross_v` `[L,H,1500,hd]` — per-layer `encoder_attn` k/v projected once (avoids re-projecting encoder states every decode step) | 1× |
| **decoder_generate** (new) | `tokens i32 [1,"seq_len"]`, `positions i32 [1,"seq_len"]`, additive causal `mask f32 [1,1,"seq_len","seq_len"]` (host-built numpy), `cross_k`,`cross_v` | **last-token** logits `[1,vocab]` (slice `h[:,-1]` before the vocab matmul) | per decode step; full-prefix recompute, **no KV cache v1** |
| **decoder_align** (new) | same inputs, teacher-forced final sequence | full logits `[1,T,vocab]` + cross-attn **post-softmax** probs for the alignment heads only `[n_align,T,1500]` f32 (~few MB) | 1× after decode |

**Host orchestration** (numpy + transformers only): `WhisperFeatureExtractor` mel → run encoder → run cross_kv → greedy loop (build `positions=arange(T)`, additive causal `mask`; run generate; apply `suppress_tokens` + `begin_suppress_tokens` to logits — **required even for greedy**; argmax; stop on eos / 448) → run align on the final sequence → timing (crop to `num_content_frames//2`, per-column standardize, median-filter w=7, mean over alignment heads, DTW, `jump_times × 0.02s`, openai-style unicode-safe token→word split) → words + punctuation merge.

Greedy prompt: `[<|startoftranscript|>, <|lang|>, <|transcribe|>, <|notimestamps|>]` — all ids from `GenerationConfig` (`lang_to_id`, `task_to_id`, `no_timestamps_token_id`, `decoder_start_token_id`, `eos_token_id`, `suppress_tokens`, `begin_suppress_tokens`, `alignment_heads`). Never hardcode; tiny/v3 differ.

## File-by-file changes (all under `max/python/max/pipelines/architectures/whisper/` unless noted)

### Modified: `encoder.py`
- Fix **B1**: attention takes explicit `(d_model, n_heads)` (read `encoder_attention_heads`); `head_dim = d_model // n_heads`.
- Parameterize `MLP(d_model, ffn_dim, dtype, device)` so the decoder reuses it with `decoder_ffn_dim` (currently hardcodes `encoder_ffn_dim`). Update encoder call sites.
- Fix **B4**: `Conv1D(..., permute=True)` for `conv1`/`conv2`; after the stem, `inputs_embeds = ops.permute(inputs_embeds, [0, 2, 1])` before adding `embed_positions.weight`.
- Keep pre-LN + final `norm`; keep `ops.gelu` (exact gelu, matches HF); keep `softmax(scores*scale)` **operand order** (compiler pattern-matches fused attention — `encoder.py:51-53`).
- **Reuse verdict:** `WhisperSdpaAttention` is self-attn-only; the decoder gets its own attention classes but reuses the `wq/wk/wv/wo` naming (one adapter rename table) and reuses `MLP`, `max.nn` `LayerNorm`/`Embedding`/`Linear`.

### New: `decoder.py`
Graph API (`max.nn.layer.Module`), mirroring the encoder.
- `WhisperDecoderSelfAttention` — `wq`(bias)/`wk`(**no bias**)/`wv`(bias)/`wo`(bias). `__call__(x, mask)`: reshape to heads, `scores = q@kᵀ`, `softmax(scores*scale + mask) @ v`.
- `WhisperCrossAttention` — only `wq`(bias)+`wo`(bias); K/V injected precomputed; returns `(out, probs)`. Non-obvious part sketched below.
- `WhisperDecoderLayer` — pre-LN ×3: `attention_norm`→self-attn→res; `cross_attention_norm`→cross-attn→res; `mlp_norm`→`MLP(d_model, decoder_ffn_dim)`→res. Returns `(h, cross_probs)`.
- `WhisperDecoder` — `embed_tokens = Embedding(vocab, d)`; `embed_positions = Embedding(448, d)` **learned**; `layers = LayerList([...])` (NOT `Sequential` — can't thread multi-arg calls; FQNs come out `layers.N.…`); final `norm = LayerNorm(d, eps=1e-5)`; **tied LM head** `logits = h @ embed_tokens.weight.T`. Two build-time flavors (no runtime branch): **generate** slices `h[:, -1, :]` before the vocab matmul; **align** returns full logits + stacked alignment-head probs `[n_align,T,1500]`.
- `WhisperCrossKV` — per layer a `cross_attention` submodule holding `wk`(no bias)/`wv`(bias) so FQNs match; `__call__(encoder_states)` → `cross_k,cross_v [L,H,1500,hd]` via reshape/transpose + `ops.stack` over layers.

### Modified: `graph.py`
Keep/rename `build_graph`→`build_encoder_graph` with **B5** fixed (input on exec device; static `3000` is fine — extractor always pads). Add `build_cross_kv_graph`, `build_decoder_generate_graph`, `build_decoder_align_graph(..., alignment_heads)`. Each decoder-side module uses a **weight subset** — filter by `module.raw_state_dict().keys()` then `load_state_dict(..., strict=True)`.

### Modified: `weight_adapters.py`
- Fix encoder map (B2/B3): strip leading `model.`; add `.fc1.→.mlp.fc1.`, `.fc2.→.mlp.fc2.`; keep `decoder.→None` drop; keep conv weights **untransposed** (Conv1D `permute=True` consumes PyTorch layout).
- Add `convert_safetensor_state_dict_decoder` as an **explicit per-key function** (not ordered substring soup — `model.decoder.layers.0.encoder_attn…` contains the substring `encoder`!): strip `model.decoder.`, drop `encoder.*` by full prefix, then `self_attn_layer_norm→attention_norm`, `self_attn→attention`, `encoder_attn_layer_norm→cross_attention_norm`, `encoder_attn→cross_attention`, `final_layer_norm→mlp_norm`, root `layer_norm.→norm.` (exact prefix), `q/k/v/out_proj→wq/wk/wv/wo`, `fc1/fc2` as above.
- Both converters take `target_dtype: DType = DType.float32` and `value.data().astype(target_dtype)` (`WeightData.astype`, `graph/weights/weights.py:236`).
- No bias synthesis needed (module bias flags already match the checkpoint).

### New: `audio.py`
Host-side, **stdlib + numpy + transformers only** (package `BUILD.bazel` globs `**/*.py`; a static `torch`/`soundfile` import would trip the bazel dep checker). `load_audio(path)` via stdlib `wave` for 16kHz-mono PCM16; dynamic `importlib` fallback to soundfile/torchaudio otherwise. `extract_features(audio, model_dir)` → mel `[1,n_mels,3000]` (`padding="max_length"`) + `num_content_frames = min(len(audio)//160, 3000)`.

### New: `timing.py`
Pure numpy + tokenizer port of openai-whisper `timing.py`: `median_filter(x, 7)` (via `np.lib.stride_tricks.sliding_window_view`, reflect pad — no scipy), `dtw(cost)` DP port of `dtw_cpu`, `find_word_alignment(...)`, `merge_punctuations(...)`. DTW/word sketches below.

### New: `transcribe.py`
`WhisperTranscriber` **plain class** (NOT `PipelineModel` — its `execute` is abstract and the parked `model.py` never implemented it, so it can't even instantiate). Loads `AutoConfig`/`WhisperTokenizerFast`/`WhisperFeatureExtractor`/`GenerationConfig`; `SafetensorWeights` → 2 adapter passes; one `InferenceSession(devices=[CPU()|Accelerator()])`; `session.load` ×4 graphs. `transcribe(path, word_timestamps=True)` → `{text, words:[{word,start,end,probability}]}`. Assert ≤30s.

### New: `cli.py` + `__init__.py`
`python -m max.pipelines.architectures.whisper.cli AUDIO --model … --device cpu|gpu --language en [--no-words] [-o out.json]` → JSON. **No** arch-registry entry.

### Modified: `BUILD.bazel`
Add `requirement("numpy")`.

### Unchanged: `model.py` (parked; add a docstring pointer to `transcribe.py`).

### New: `max/tests/integration/architectures/whisper/standalone/`
`common.py`, `parity_encoder.py`, `parity_decoder.py`, `parity_transcript.py`, `parity_words.py`, `dump_faster_whisper_ref.py` — named `parity_*`/`dump_*` so the bazel `**/test_*.py` glob ignores them (torch allowed here).

## Key code sketches (the non-obvious parts)

### Cross-attention returning probs (`decoder.py`)
```python
class WhisperCrossAttention(Module):
    def __init__(self, d_model, n_heads, dtype, device):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.wq = Linear(d_model, d_model, dtype, device, has_bias=True)
        self.wo = Linear(d_model, d_model, dtype, device, has_bias=True)

    def __call__(self, x, k, v):          # x:[1,T,d]; k,v:[H,S,hd] precomputed (one layer)
        batch, seq_len = x.shape[0], x.shape[1]
        xq = ops.reshape(self.wq(x), [batch, seq_len, self.n_heads, self.head_dim]).transpose(1, 2)  # [1,H,T,hd]
        k = ops.unsqueeze(k, 0)           # [1,H,S,hd]
        v = ops.unsqueeze(v, 0)
        scale = math.sqrt(1.0 / self.head_dim)
        scores = xq @ ops.transpose(k, 2, 3)                 # [1,H,T,S]
        probs = ops.softmax(scores * scale)                  # POST-softmax, matches HF output_attentions
        out = (probs @ v).transpose(1, 2).reshape([batch, seq_len, -1])
        return self.wo(out), probs                           # caller casts probs → f32
```

### DTW (`timing.py`, numpy port of openai `dtw_cpu`)
```python
def dtw(x):                              # x = -cost, shape (N_tokens, M_frames)
    N, M = x.shape
    cost = np.full((N + 1, M + 1), np.inf); cost[0, 0] = 0.0
    trace = np.full((N + 1, M + 1), -1, dtype=np.int8); trace[0, :] = 2; trace[:, 0] = 1
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            c0, c1, c2 = cost[i-1, j-1], cost[i-1, j], cost[i, j-1]
            t = 0 if (c0 <= c1 and c0 <= c2) else (1 if c1 <= c2 else 2)
            cost[i, j] = x[i-1, j-1] + (c0 if t == 0 else c1 if t == 1 else c2)
            trace[i, j] = t
    i, j, ti, tj = N, M, [], []
    while i > 0 or j > 0:
        ti.append(i-1); tj.append(j-1); t = trace[i, j]
        if t == 0: i -= 1; j -= 1
        elif t == 1: i -= 1
        else: j -= 1
    return np.array(ti[::-1]), np.array(tj[::-1])            # 448×1500 pure-python ≈ ≤2s; vectorize later
```
`find_word_alignment`: crop `w=align_probs[:,:,:num_content_frames//2]`; standardize over token axis `(w-mean)/(std+eps)`; `median_filter(w,7)`; `matrix=w.mean(0)`; slice rows `matrix[len(sot_sequence):-1]`; `text_idx,time_idx=dtw(-matrix)`; `jump_times=time_idx[np.diff(text_idx,prepend=-1)>0]*0.02`; split via openai `split_tokens_on_unicode` (incremental decode + `�` guard — never decode tokens one-by-one, byte-BPE breaks) then `split_tokens_on_spaces`; word prob = mean of greedy per-token softmax; `merge_punctuations(words, prepended="\"'“¿([{-", appended="\"'.。,，!！?？:：”)]}、")`.

## Pitfalls checklist
1. **Symbolic dims:** reuse the same `"seq_len"` string across tokens/positions/mask; if `scores+mask` complains, use the `ops.rebind` trick from `pixtral/vision_encoder/attention.py:116-118`.
2. **Conv1D permute:** `permute=True` = PyTorch `(out,in,k)` weights AND channels-first I/O — take both or neither. k3/s2/p1 → exactly 1500 from 3000.
3. **Positional embeddings:** encoder sinusoidal-**but-stored** → load, bit-exact. Decoder **learned** → index by `positions`, never broadcast-add.
4. **Pre-LN + final `layer_norm`:** normalize before each sublayer and once at the end (decoder: before the tied LM head). eps=1e-5.
5. **Tied LM head:** no `proj_out` — reuse `embed_tokens` weight.
6. **Attention scaling:** keep `softmax(scores*scale)` operand order (fused-attn pattern-match).
7. **Suppression is not optional:** without `suppress_tokens` + `begin_suppress_tokens` greedy visibly degrades (blanks, stray timestamp tokens). Apply host-side before argmax.
8. **DTW frame crop** before normalization (trailing-silence attention pollutes alignment); matrix rows `[len(sot_sequence):-1]`.
9. **Word splitting:** unicode-safe incremental decode with `�` guard.
10. **fp16 checkpoint (v3):** adapter `.astype(f32)`.
11. **Bazel hygiene:** package globs `**/*.py` → no torch/soundfile static imports in the package dir; test glob `**/test_*.py` → standalone scripts must not be named `test_*`.

## Parity harness

Old skipped test (`max/tests/integration/architectures/whisper/test_encoder.py`): librispeech-dummy → `AutoProcessor` mel → HF `model.model.encoder(...).last_hidden_state` vs MAX graph, `assert_allclose(rtol=1e-4, atol=1e-6)`. Never ran (empty state dict, channels-last fixture, `n_heads` crash). Standalone scripts replicate this with a real state dict + the fixed adapter:

1. `parity_encoder.py` — gate (a): encoder vs HF, pass `rtol=1e-4, atol=1e-4` f32.
2. `parity_decoder.py` — gate (b): teacher-forced logits vs HF `model(input_features, decoder_input_ids=…).logits` (max-abs<1e-3, 100% per-position argmax) + alignment-head probs vs HF `output_attentions=True` (<1e-3 post-softmax).
3. `parity_transcript.py` — gate (c): MAX greedy vs HF `generate(num_beams=1, do_sample=False, language=…, task="transcribe")` token-for-token.
4. `parity_words.py --hyp … --ref … --tol-ms 80` — gate (d): primary ref HF `generate(return_timestamps="word")` (median |Δ|≤40ms); acceptance ref faster-whisper large-v3 (throwaway venv) — words equal, Δ median ≤80ms, p95 ≤160ms.

## Phases & gates (tiny first, then large-v3)

- **Phase 0** — skeleton: `__init__.py`, `standalone/` dir, `BUILD.bazel` numpy dep. (branch/worktree already created.)
- **Phase 1** — encoder revival → **gate (a)**: fix B1–B5 + `parity_encoder.py`.
- **Phase 2** — decoder + cross_kv + align → **gate (b)**: `decoder.py`, decoder adapter, two graphs, `parity_decoder.py`.
- **Phase 3** — generate graph + greedy loop → **gate (c)**: `build_decoder_generate_graph`, `transcribe.py` loop + suppression, `parity_transcript.py`.
- **Phase 4** — word timestamps → **gate (d)**: `timing.py`, `audio.py`, `cli.py`, `parity_words.py`, `dump_faster_whisper_ref.py`.
- **Phase 5** — portability + polish → **gate (e)**: full CLI on max-build GPU (large-v3, RTF) AND Mac arm64 CPU (**tiny**); `./bazelw run //:format`.

## Environments

- **Mac (CPU)**: `python3 -m venv ~/.venvs/max-whisper` → `pip install --pre modular --index-url https://dl.modular.com/public/nightly/python/simple/` + `transformers>=5.12,<5.13 torch numpy datasets soundfile` → `export PYTHONPATH=/Users/mardel/src/opensource/modular/modular-whisper/max/python`. Pin the nightly near the checkout date (native `_core` ABI vs Python-source drift).
- **max-build (GPU sm_120)**: this worktree pushes to a matching worktree on the box (`git worktree add /opt/modular-whisper feat/whisper-word-ts`) — **do NOT switch `/opt/modular-custom`'s branch; the Klein fp4 dev serve runs from it**. Run via `PYTHONPATH=/opt/modular-whisper/max/python /root/wheeltest-baked/bin/python …`, `--device gpu`.

## Deferred (out of v1)
KV-cached decoder step graph (the named perf follow-up) · bf16 · >30s chunking + timestamp-token decoding rules · language autodetect · beam search / temperature fallback · serve integration (SPEECH task enum, `/v1/audio/transcriptions`, executor) · batching.
