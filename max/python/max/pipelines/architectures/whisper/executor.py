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
"""Served Whisper executor: batched KV-cached transcription with word timestamps.

The serving counterpart of :class:`~.transcribe.WhisperTranscriber`. Where the
transcriber owns its own :class:`~max.engine.InferenceSession` and drives a CLI,
this executor implements the
:class:`~max.pipelines.lib.pipeline_executor.PipelineExecutor` contract so the
``max serve`` :class:`~max.pipelines.speech.pipeline.SpeechToTextPipeline` /
``OneShotScheduler`` can dynamically batch concurrent ≤30s chunk requests.

Design deviations from the strict executor contract, both deliberate:

* **Inputs** — :class:`WhisperExecInputs` is a :class:`TensorStruct`, but the
  host-driven greedy-decode loop needs per-row host metadata (SOT prompt,
  content-frame count, the word-timestamp flag). That rides in the ``_meta``
  private field, which ``TensorStruct`` skips for both field validation and
  ``.to()``. Only ``mel`` is a device tensor.
* **Output** — :class:`WhisperExecResult` is a plain frozen dataclass, not a
  ``TensorStruct`` (text/tokens/words are not tensors), so the executor is
  parametrized ``PipelineExecutor[SpeechToTextContext, WhisperExecInputs, Any]``.

The decode/align math mirrors ``transcribe.py``'s validated Phase-B batched
path (lockstep KV-cached decode; finished rows feed EOS so ``cache_len`` stays a
single scalar; per-row alignment through the batch-1 align graph).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
from max.driver import Buffer, load_devices
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef
from max.pipelines.context import SpeechToTextContext
from max.pipelines.context.outputs import TranscribedWord
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.lib.pipeline_executor import PipelineExecutor
from max.pipelines.lib.pipeline_runtime_config import PipelineRuntimeConfig
from max.pipelines.modeling.base import TensorStruct

from .decoder import NEG_INF
from .graph import (
    build_cross_kv_graph,
    build_decoder_align_graph,
    build_decoder_cached_graph,
    build_encoder_graph,
)
from .timing import find_word_alignment
from .weight_adapters import (
    _rename_decoder_key,
    _rename_encoder_key,
    load_raw_state_dict,
    rename_state_dict,
)

logger = logging.getLogger("max.pipelines")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def _causal_mask(seq_len: int) -> np.ndarray:
    m = np.triu(np.full((seq_len, seq_len), NEG_INF, dtype=np.float32), k=1)
    return m[None, None, :, :]


def _cached_mask(cache_len: int, t_new: int, max_t: int) -> np.ndarray:
    """Additive mask ``[1,1,t_new,max_t]``: query row i (abs pos cache_len+i)
    attends keys 0..cache_len+i; the unwritten cache tail is masked out."""
    m = np.full((t_new, max_t), NEG_INF, dtype=np.float32)
    for i in range(t_new):
        m[i, : cache_len + i + 1] = 0.0
    return m[None, None, :, :]


@dataclass
class _RowMeta:
    """Per-request host metadata the decode/align loop needs (one per batch row)."""

    prompt_tokens: list[int]
    """The SOT prompt token ids built API-side by the serve tokenizer."""

    num_content_frames: int
    """Real (non-padded) mel frames — crops trailing silence in DTW."""

    word_timestamps: bool
    """Whether to run the alignment pass for this row."""

    language: str
    """Transcription language (fallback SOT construction / response field)."""


@dataclass(frozen=True)
class WhisperExecInputs(TensorStruct):
    """Batched graph inputs for one transcription group.

    ``mel`` is the only device tensor; the host-driven decode loop's per-row
    metadata rides in ``_meta`` (a ``TensorStruct`` private field, skipped by
    field-type validation and by ``.to()``).
    """

    mel: Buffer
    """Stacked log-mel features ``[B, num_mel_bins, 3000]`` float32, on device."""

    _meta: list[_RowMeta]
    """Per-row host metadata; ``len(_meta) == B`` and row-aligned with ``mel``."""


@dataclass(frozen=True)
class WhisperExecResult:
    """Row-ordered transcription result (plain dataclass; not a TensorStruct)."""

    texts: list[str]
    """Decoded transcript per row."""

    tokens: list[list[int]]
    """Generated token ids per row."""

    words: list[list[TranscribedWord]] | None
    """Per-row word timestamps, or ``None`` when the batch didn't request them."""


class WhisperExecutor(
    PipelineExecutor[SpeechToTextContext, WhisperExecInputs, Any]
):
    """KV-cached, dynamically-batchable Whisper transcription executor."""

    # prepare_inputs stacks multiple compatible contexts (same language +
    # word-timestamp flag, grouped by the scheduler's batch_key) into one
    # lockstep batched decode, so the scheduler may dynamically batch requests.
    supports_dynamic_batching: bool = True

    def __init__(
        self,
        manifest: ModelManifest,
        session: InferenceSession,
        runtime_config: PipelineRuntimeConfig,
    ) -> None:
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoConfig,
            GenerationConfig,
            WhisperTokenizerFast,
        )

        self._manifest = manifest
        self._session = session
        self._runtime_config = runtime_config

        main = manifest["main"]
        model_path = main.model_path
        revision = getattr(main, "huggingface_model_revision", None)
        if os.path.isdir(model_path):
            model_dir = model_path
        else:
            model_dir = snapshot_download(model_path, revision=revision)

        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(model_dir)
        self.gen = GenerationConfig.from_pretrained(model_dir)

        # Control ids + suppression (read from config, never hard-coded).
        self.eos_id = self.config.eos_token_id
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

        # Device: resolve from the same specs the session was built with.
        self._device = load_devices(main.device_specs)[0]
        device_ref = DeviceRef.from_device(self._device)

        # Weights -> encoder/decoder numpy state dicts (float32).
        raw = load_raw_state_dict(model_dir)
        enc_sd = rename_state_dict(raw, _rename_encoder_key)
        dec_sd = rename_state_dict(raw, _rename_decoder_key)

        # Four compiled graphs on the injected session: encoder, cross-KV
        # precompute, unified KV-cached step graph, and teacher-forced align.
        self.encoder_model = session.load(
            build_encoder_graph(enc_sd, self.config, DType.float32, device_ref),
            weights_registry=enc_sd,
        )
        self.cross_kv_model = session.load(
            build_cross_kv_graph(
                dec_sd, self.config, DType.float32, device_ref
            ),
            weights_registry=dec_sd,
        )
        self.cached_model = session.load(
            build_decoder_cached_graph(
                dec_sd, self.config, DType.float32, device_ref
            ),
            weights_registry=dec_sd,
        )
        self.align_model = session.load(
            build_decoder_align_graph(
                dec_sd,
                self.config,
                DType.float32,
                device_ref,
                self.alignment_heads,
            ),
            weights_registry=dec_sd,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _buf(self, arr: np.ndarray) -> Buffer:
        return Buffer.from_numpy(np.ascontiguousarray(arr)).to(self._device)

    def _build_sot(self, language: str) -> list[int]:
        """Construct the SOT prompt for ``language`` (transcribe, no timestamps).

        Fallback for the rare case a context arrives without ``prompt_tokens``;
        the serve tokenizer normally supplies them.
        """
        lang_id = self.gen.lang_to_id[f"<|{language}|>"]
        task_id = self.gen.task_to_id["transcribe"]
        return [
            self.config.decoder_start_token_id,
            lang_id,
            task_id,
            self.gen.no_timestamps_token_id,
        ]

    # ------------------------------------------------------------------ #
    # PipelineExecutor contract
    # ------------------------------------------------------------------ #
    def prepare_inputs(
        self, contexts: list[SpeechToTextContext]
    ) -> WhisperExecInputs:
        if not contexts:
            raise ValueError("WhisperExecutor requires at least one context")
        mels: list[np.ndarray] = []
        meta: list[_RowMeta] = []
        for ctx in contexts:
            mel = np.asarray(ctx.mel, dtype=np.float32)
            if mel.ndim == 3:  # [1, n_mels, 3000] -> [n_mels, 3000]
                mel = mel[0]
            mels.append(mel)
            meta.append(
                _RowMeta(
                    prompt_tokens=list(ctx.prompt_tokens),
                    num_content_frames=ctx.num_content_frames,
                    word_timestamps=ctx.word_timestamps,
                    language=ctx.language,
                )
            )
        mel_batch = np.stack(mels, axis=0)  # [B, n_mels, 3000]
        return WhisperExecInputs(mel=self._buf(mel_batch), _meta=meta)

    def execute(self, inputs: WhisperExecInputs) -> WhisperExecResult:
        meta = inputs._meta
        batch = len(meta)
        logger.info("WhisperExecutor.execute: batch_size=%d", batch)
        # batch_key groups by (language, word_timestamps), so the SOT prompt is
        # identical across rows — take row 0's (or rebuild from its language).
        sot = meta[0].prompt_tokens or self._build_sot(meta[0].language)
        want_words = meta[0].word_timestamps

        enc = (
            self.encoder_model.execute(inputs.mel)[0]
            .to_numpy()
            .astype(np.float32)
        )  # [B, 1500, d_model]

        text_tokens, token_probs = self._decode_batch(enc, sot)

        texts: list[str] = []
        tokens_out: list[list[int]] = []
        words_out: list[list[TranscribedWord]] = []
        for b in range(batch):
            toks = text_tokens[b]
            texts.append(
                self.tokenizer.decode(toks, skip_special_tokens=True)
            )
            tokens_out.append(toks)
            row_words: list[TranscribedWord] = []
            if want_words and toks:
                align_probs = self._alignment_probs(enc[b : b + 1], sot, toks)
                for w in find_word_alignment(
                    align_probs,
                    toks,
                    self.tokenizer,
                    meta[b].num_content_frames,
                    token_probs=token_probs[b],
                    sot_len=len(sot),
                ):
                    prob = w["probability"]
                    row_words.append(
                        TranscribedWord(
                            word=w["word"],
                            start=w["start"],
                            end=w["end"],
                            probability=prob if prob is not None else 0.0,
                        )
                    )
            words_out.append(row_words)

        return WhisperExecResult(
            texts=texts,
            tokens=tokens_out,
            words=words_out if want_words else None,
        )

    # ------------------------------------------------------------------ #
    # Decode / align (mirrors transcribe.py's validated Phase-B path)
    # ------------------------------------------------------------------ #
    def _decode_batch(
        self, enc: np.ndarray, sot: list[int]
    ) -> tuple[list[list[int]], list[np.ndarray]]:
        cfg = self.config
        batch = enc.shape[0]
        n_layers = cfg.decoder_layers
        n_heads = cfg.decoder_attention_heads
        head_dim = cfg.d_model // n_heads
        max_t = cfg.max_target_positions

        cross_k, cross_v = self.cross_kv_model.execute(self._buf(enc))
        zeros = np.zeros((batch, n_heads, max_t, head_dim), dtype=np.float32)
        k_bufs = [
            Buffer.from_numpy(zeros.copy()).to(self._device)
            for _ in range(n_layers)
        ]
        v_bufs = [
            Buffer.from_numpy(zeros.copy()).to(self._device)
            for _ in range(n_layers)
        ]

        def run(
            tokens_2d: np.ndarray, positions_2d: np.ndarray, cache_len: int
        ) -> np.ndarray:
            t_new = tokens_2d.shape[1]
            mask = _cached_mask(cache_len, t_new, max_t)
            clen = Buffer.from_numpy(np.array(cache_len, dtype=np.int64))  # CPU
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

        max_new = max_t
        text_tokens: list[list[int]] = [[] for _ in range(batch)]
        token_probs: list[list[float]] = [[] for _ in range(batch)]
        done = [False] * batch

        # Prefill: every row feeds the same SOT prompt (lockstep).
        sot_arr = np.tile(np.array(sot, dtype=np.int32), (batch, 1))
        pos = np.tile(np.arange(len(sot), dtype=np.int32), (batch, 1))
        logits = run(sot_arr, pos, 0)
        cache_len = len(sot)
        first = True
        while cache_len < len(sot) + max_new and not all(done):
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
        return text_tokens, [
            np.array(p, dtype=np.float64) for p in token_probs
        ]

    def _alignment_probs(
        self, enc: np.ndarray, sot: list[int], text_tokens: list[int]
    ) -> np.ndarray:
        seq = sot + text_tokens + [self.eos_id]
        seq_len = len(seq)
        tok = np.array([seq], dtype=np.int32)
        pos = np.arange(seq_len, dtype=np.int32)[None, :]
        mask = _causal_mask(seq_len)
        outs = self.align_model.execute(
            self._buf(tok), self._buf(pos), self._buf(mask), self._buf(enc)
        )
        return outs[1].to_numpy().astype(np.float64)  # [n_align, T, S]
