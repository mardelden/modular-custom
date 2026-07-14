# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Standalone Whisper transcriber (≤30s window, greedy, optional word timestamps).

Orchestrates the four host/device steps:
  1. audio -> log-mel  (host, ``audio.py`` + HF feature extractor)
  2. encoder graph      -> encoder states
  3. greedy decode loop -> transcript tokens (generate graph + host suppression)
  4. align graph + DTW  -> word timestamps (``timing.py``)

Not a ``PipelineModel`` / not registered for ``max serve`` — a plain class built
directly on ``InferenceSession``. Weights load from safetensors (no torch).
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np

from .audio import extract_features, load_audio
from .decoder import NEG_INF
from .graph import (
    build_cross_kv_graph,
    build_decoder_align_graph,
    build_decoder_cached_graph,
    build_decoder_generate_graph,
    build_encoder_graph,
)
from .timing import find_word_alignment
from .weight_adapters import _rename_decoder_key, _rename_encoder_key


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _causal_mask(seq_len: int) -> np.ndarray:
    m = np.triu(np.full((seq_len, seq_len), NEG_INF, dtype=np.float32), k=1)
    return m[None, None, :, :]


def _load_raw_state_dict(model_dir: str) -> dict[str, np.ndarray]:
    """Load all safetensors shards under ``model_dir`` as a numpy state dict."""
    from safetensors.numpy import load_file

    single = os.path.join(model_dir, "model.safetensors")
    index = os.path.join(model_dir, "model.safetensors.index.json")
    sd: dict[str, np.ndarray] = {}
    if os.path.exists(single):
        sd.update(load_file(single))
    elif os.path.exists(index):
        with open(index) as f:
            files = sorted(set(json.load(f)["weight_map"].values()))
        for name in files:
            sd.update(load_file(os.path.join(model_dir, name)))
    else:
        for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
            sd.update(load_file(path))
    if not sd:
        raise FileNotFoundError(f"No safetensors weights found in {model_dir}")
    return sd


class WhisperTranscriber:
    """Load a Whisper checkpoint and transcribe a ≤30s clip with word timestamps."""

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        language: str = "en",
        use_kv_cache: bool = True,
    ) -> None:
        from huggingface_hub import snapshot_download
        from max.driver import CPU, Accelerator, accelerator_count
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef
        from transformers import (
            AutoConfig,
            GenerationConfig,
            WhisperFeatureExtractor,
            WhisperTokenizerFast,
        )

        model_dir = snapshot_download(model_id)
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(model_dir)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_dir
        )
        self.gen = GenerationConfig.from_pretrained(model_dir)

        # Prompt + control ids (never hard-coded — read from the config).
        lang_id = self.gen.lang_to_id[f"<|{language}|>"]
        task_id = self.gen.task_to_id["transcribe"]
        self.sot = [
            self.config.decoder_start_token_id,
            lang_id,
            task_id,
            self.gen.no_timestamps_token_id,
        ]
        self.eos_id = self.config.eos_token_id
        # Suppress the generation-config tokens, plus every timestamp token
        # (we decode in <|notimestamps|> mode).
        suppress = set(int(t) for t in (self.gen.suppress_tokens or []))
        suppress.update(
            range(self.gen.no_timestamps_token_id + 1, self.config.vocab_size)
        )
        self.suppress_ids = np.array(sorted(suppress), dtype=np.int64)
        self.begin_suppress_ids = np.array(
            [int(t) for t in (self.gen.begin_suppress_tokens or [])],
            dtype=np.int64,
        )
        self.alignment_heads = [tuple(p) for p in self.gen.alignment_heads]

        # Device + session.
        if device == "gpu":
            if accelerator_count() == 0:
                raise RuntimeError("device='gpu' but no accelerator found")
            self.device = Accelerator()
        else:
            self.device = CPU()
        device_ref = DeviceRef.from_device(self.device)
        self.session = InferenceSession(devices=[self.device])

        # Weights -> encoder/decoder state dicts (float32).
        raw = _load_raw_state_dict(model_dir)
        enc_sd = self._rename(raw, _rename_encoder_key)
        dec_sd = self._rename(raw, _rename_decoder_key)

        self.use_kv_cache = use_kv_cache
        self.encoder_model = self.session.load(
            build_encoder_graph(enc_sd, self.config, DType.float32, device_ref),
            weights_registry=enc_sd,
        )
        if use_kv_cache:
            # v2: precompute cross-K/V once + a single cached step graph.
            self.cross_kv_model = self.session.load(
                build_cross_kv_graph(
                    dec_sd, self.config, DType.float32, device_ref
                ),
                weights_registry=dec_sd,
            )
            self.cached_model = self.session.load(
                build_decoder_cached_graph(
                    dec_sd, self.config, DType.float32, device_ref
                ),
                weights_registry=dec_sd,
            )
        else:
            # v1 path (kept for the parity cross-check): full-prefix recompute.
            self.generate_model = self.session.load(
                build_decoder_generate_graph(
                    dec_sd, self.config, DType.float32, device_ref
                ),
                weights_registry=dec_sd,
            )
        self.align_model = self.session.load(
            build_decoder_align_graph(
                dec_sd,
                self.config,
                DType.float32,
                device_ref,
                self.alignment_heads,
            ),
            weights_registry=dec_sd,
        )

    @staticmethod
    def _rename(raw, rename_fn) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key, arr in raw.items():
            name = rename_fn(key)
            if name is None:
                continue
            out[name] = np.asarray(arr, dtype=np.float32)
        return out

    def _buf(self, arr: np.ndarray):
        from max.driver import Buffer

        return Buffer.from_numpy(np.ascontiguousarray(arr)).to(self.device)

    def _encode(self, mel: np.ndarray) -> np.ndarray:
        out = self.encoder_model.execute(self._buf(mel))[0]
        return out.to_numpy().astype(np.float32)

    def _greedy_decode(
        self, enc: np.ndarray, max_new_tokens: int | None = None
    ) -> tuple[list[int], np.ndarray]:
        tokens = list(self.sot)
        token_probs: list[float] = []
        max_len = len(self.sot) + self.config.max_target_positions
        if max_new_tokens is not None:
            max_len = min(max_len, len(self.sot) + max_new_tokens)
        while len(tokens) < max_len:
            seq_len = len(tokens)
            tok = np.array([tokens], dtype=np.int32)
            pos = np.arange(seq_len, dtype=np.int32)[None, :]
            mask = _causal_mask(seq_len)
            logits = self.generate_model.execute(
                self._buf(tok), self._buf(pos), self._buf(mask), self._buf(enc)
            )[0].to_numpy()
            nxt = logits[0, -1].astype(np.float64)
            nxt[self.suppress_ids] = -np.inf
            if len(tokens) == len(self.sot) and self.begin_suppress_ids.size:
                nxt[self.begin_suppress_ids] = -np.inf
            token_id = int(np.argmax(nxt))
            if token_id == self.eos_id:
                break
            token_probs.append(float(_softmax(nxt)[token_id]))
            tokens.append(token_id)
        text_tokens = tokens[len(self.sot) :]
        return text_tokens, np.array(token_probs, dtype=np.float64)

    @staticmethod
    def _cached_mask(cache_len: int, t_new: int, max_t: int) -> np.ndarray:
        """Additive mask ``[1,1,t_new,max_t]``: query row i (abs pos cache_len+i)
        attends keys 0..cache_len+i; the unwritten cache tail is masked out."""
        m = np.full((t_new, max_t), NEG_INF, dtype=np.float32)
        for i in range(t_new):
            m[i, : cache_len + i + 1] = 0.0
        return m[None, None, :, :]

    def _greedy_decode_cached(
        self, enc: np.ndarray, max_new_tokens: int | None = None
    ) -> tuple[list[int], np.ndarray]:
        from max.driver import Buffer

        cfg = self.config
        n_layers = cfg.decoder_layers
        n_heads = cfg.decoder_attention_heads
        head_dim = cfg.d_model // n_heads
        max_t = cfg.max_target_positions

        # Precompute cross-K/V once; keep it + the caches device-resident.
        cross_k, cross_v = self.cross_kv_model.execute(self._buf(enc))
        zeros = np.zeros((1, n_heads, max_t, head_dim), dtype=np.float32)
        k_bufs = [
            Buffer.from_numpy(zeros.copy()).to(self.device)
            for _ in range(n_layers)
        ]
        v_bufs = [
            Buffer.from_numpy(zeros.copy()).to(self.device)
            for _ in range(n_layers)
        ]

        def run(token_list: list[int], positions: list[int], cache_len: int):
            t_new = len(token_list)
            tok = np.array([token_list], dtype=np.int32)
            pos = np.array([positions], dtype=np.int32)
            mask = self._cached_mask(cache_len, t_new, max_t)
            clen = Buffer.from_numpy(np.array(cache_len, dtype=np.int64))  # CPU
            out = self.cached_model.execute(
                self._buf(tok),
                self._buf(pos),
                self._buf(mask),
                clen,
                cross_k,
                cross_v,
                *k_bufs,
                *v_bufs,
            )[0]
            return out.to_numpy()[0, -1].astype(np.float64)  # [vocab]

        max_new = (
            max_t if max_new_tokens is None else min(max_t, max_new_tokens)
        )
        text_tokens: list[int] = []
        token_probs: list[float] = []
        # Prefill the SOT prompt (positions 0..len(sot)-1).
        logits = run(self.sot, list(range(len(self.sot))), 0)
        cache_len = len(self.sot)
        first = True
        while len(text_tokens) < max_new:
            logits[self.suppress_ids] = -np.inf
            if first and self.begin_suppress_ids.size:
                logits[self.begin_suppress_ids] = -np.inf
            first = False
            token_id = int(np.argmax(logits))
            if token_id == self.eos_id:
                break
            token_probs.append(float(_softmax(logits)[token_id]))
            text_tokens.append(token_id)
            logits = run([token_id], [cache_len], cache_len)
            cache_len += 1
        return text_tokens, np.array(token_probs, dtype=np.float64)

    def _alignment_probs(
        self, enc: np.ndarray, text_tokens: list[int]
    ) -> np.ndarray:
        seq = self.sot + text_tokens + [self.eos_id]
        seq_len = len(seq)
        tok = np.array([seq], dtype=np.int32)
        pos = np.arange(seq_len, dtype=np.int32)[None, :]
        mask = _causal_mask(seq_len)
        outs = self.align_model.execute(
            self._buf(tok), self._buf(pos), self._buf(mask), self._buf(enc)
        )
        return outs[1].to_numpy().astype(np.float64)  # [n_align, T, S]

    def transcribe(
        self,
        audio_path: str,
        word_timestamps: bool = True,
        max_new_tokens: int | None = None,
    ) -> dict:
        audio = load_audio(audio_path)
        duration = len(audio) / 16000.0
        if duration > 30.5:
            raise ValueError(
                f"audio is {duration:.1f}s; v1 handles a single ≤30s window "
                "(long-audio chunking is not implemented yet)"
            )
        mel, num_content_frames = extract_features(
            audio, self.feature_extractor
        )
        enc = self._encode(mel)
        if self.use_kv_cache:
            text_tokens, token_probs = self._greedy_decode_cached(
                enc, max_new_tokens
            )
        else:
            text_tokens, token_probs = self._greedy_decode(enc, max_new_tokens)
        text = self.tokenizer.decode(text_tokens, skip_special_tokens=True)

        result: dict = {"text": text, "words": []}
        if word_timestamps and text_tokens:
            align_probs = self._alignment_probs(enc, text_tokens)
            result["words"] = find_word_alignment(
                align_probs,
                text_tokens,
                self.tokenizer,
                num_content_frames,
                token_probs=token_probs,
                sot_len=len(self.sot),
            )
        return result

    # ------------------------------------------------------------------ #
    # v2 micro-batching: batch the (throughput-critical) decode loop; run
    # per-clip alignment with the validated batch-1 align path.
    # ------------------------------------------------------------------ #
    def _greedy_decode_cached_batch(
        self, enc: np.ndarray, max_new_tokens: int | None = None
    ) -> tuple[list[list[int]], list[np.ndarray]]:
        from max.driver import Buffer

        cfg = self.config
        batch = enc.shape[0]
        n_layers = cfg.decoder_layers
        n_heads = cfg.decoder_attention_heads
        head_dim = cfg.d_model // n_heads
        max_t = cfg.max_target_positions

        cross_k, cross_v = self.cross_kv_model.execute(self._buf(enc))
        zeros = np.zeros((batch, n_heads, max_t, head_dim), dtype=np.float32)
        k_bufs = [
            Buffer.from_numpy(zeros.copy()).to(self.device)
            for _ in range(n_layers)
        ]
        v_bufs = [
            Buffer.from_numpy(zeros.copy()).to(self.device)
            for _ in range(n_layers)
        ]

        def run(
            tokens_2d: np.ndarray, positions_2d: np.ndarray, cache_len: int
        ):
            t_new = tokens_2d.shape[1]
            mask = self._cached_mask(cache_len, t_new, max_t)
            clen = Buffer.from_numpy(np.array(cache_len, dtype=np.int64))
            out = self.cached_model.execute(
                self._buf(tokens_2d.astype(np.int32)),
                self._buf(positions_2d.astype(np.int32)),
                self._buf(mask),
                clen,
                cross_k,
                cross_v,
                *k_bufs,
                *v_bufs,
            )[0]
            return out.to_numpy()[:, -1].astype(np.float64)  # [batch, vocab]

        max_new = (
            max_t if max_new_tokens is None else min(max_t, max_new_tokens)
        )
        text_tokens: list[list[int]] = [[] for _ in range(batch)]
        token_probs: list[list[float]] = [[] for _ in range(batch)]
        done = [False] * batch

        # Prefill: every row feeds the same SOT prompt (lockstep).
        sot = np.tile(np.array(self.sot, dtype=np.int32), (batch, 1))
        pos = np.tile(np.arange(len(self.sot), dtype=np.int32), (batch, 1))
        logits = run(sot, pos, 0)
        cache_len = len(self.sot)
        first = True
        while cache_len < len(self.sot) + max_new and not all(done):
            next_tokens = np.full(batch, self.eos_id, dtype=np.int32)
            for b in range(batch):
                if done[b]:
                    continue
                row = logits[b]
                row[self.suppress_ids] = -np.inf
                if first and self.begin_suppress_ids.size:
                    row[self.begin_suppress_ids] = -np.inf
                token_id = int(np.argmax(row))
                if token_id == self.eos_id:
                    done[b] = True
                    continue
                token_probs[b].append(float(_softmax(row)[token_id]))
                text_tokens[b].append(token_id)
                next_tokens[b] = token_id
            first = False
            if all(done):
                break
            # Finished rows keep feeding EOS so cache_len stays a single scalar.
            logits = run(
                next_tokens.reshape(batch, 1),
                np.full((batch, 1), cache_len, dtype=np.int32),
                cache_len,
            )
            cache_len += 1
        return text_tokens, [np.array(p, dtype=np.float64) for p in token_probs]

    def transcribe_batch(
        self,
        audio_paths: list[str],
        word_timestamps: bool = True,
        max_new_tokens: int | None = None,
    ) -> list[dict]:
        """Transcribe several ≤30s clips in one batched decode (KV cache only)."""
        if not self.use_kv_cache:
            raise RuntimeError("transcribe_batch requires use_kv_cache=True")
        mels, content_frames = [], []
        for path in audio_paths:
            audio = load_audio(path)
            if len(audio) / 16000.0 > 30.5:
                raise ValueError(f"{path}: audio exceeds the 30s window")
            mel, ncf = extract_features(audio, self.feature_extractor)
            mels.append(mel[0])
            content_frames.append(ncf)
        mel_batch = np.stack(mels, axis=0)  # [B, n_mels, 3000]
        enc = self.encoder_model.execute(self._buf(mel_batch))[0].to_numpy()
        enc = enc.astype(np.float32)

        text_tokens, token_probs = self._greedy_decode_cached_batch(
            enc, max_new_tokens
        )

        results: list[dict] = []
        for b, tokens in enumerate(text_tokens):
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            words: list[dict] = []
            if word_timestamps and tokens:
                align_probs = self._alignment_probs(enc[b : b + 1], tokens)
                words = find_word_alignment(
                    align_probs,
                    tokens,
                    self.tokenizer,
                    content_frames[b],
                    token_probs=token_probs[b],
                    sot_len=len(self.sot),
                )
            results.append({"text": text, "words": words})
        return results
