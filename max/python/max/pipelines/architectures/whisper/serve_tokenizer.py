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
"""Server-side tokenizer for the Whisper SPEECH_TO_TEXT task.

Runs in the API front-end process: it decodes the uploaded audio bytes to a
16 kHz mono waveform, extracts the log-mel spectrogram, builds the SOT prompt
for the requested language, and packages everything into a
:class:`~max.pipelines.context.SpeechToTextContext` (the mel rides the ZMQ OOB
path to the worker). Mirrors the ``PipelineTokenizer`` protocol so the serving
registry can treat it like any other tokenizer, but audio never becomes a token
stream so :meth:`encode` is unused.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any

from max.pipelines.context import SpeechToTextContext
from max.pipelines.context.exceptions import InputError
from max.pipelines.request import RequestID

from .audio import extract_features, load_audio_bytes

logger = logging.getLogger("max.pipelines")

# The encoder is a fixed 30s window; reject anything past a small tolerance.
_MAX_AUDIO_SECONDS: float = 30.5
_SAMPLE_RATE: int = 16000


@dataclass
class TranscriptionRequest:
    """API-side request for ``POST /v1/audio/transcriptions``.

    Built by the route from the multipart form; consumed by
    :meth:`WhisperServeTokenizer.new_context`. A plain dataclass with a
    ``request_id`` property so it satisfies the ``Request`` protocol.
    """

    audio_bytes: bytes
    """Raw bytes of the uploaded audio file."""

    model: str
    """Served model name (echoed back in the response)."""

    language: str | None = None
    """ISO language code; falls back to the tokenizer default when ``None``."""

    response_format: str = "json"
    """One of ``json`` / ``verbose_json`` / ``text`` (route-level concern)."""

    word_timestamps: bool = False
    """Whether ``timestamp_granularities[]=word`` was requested."""

    _request_id: RequestID = field(default_factory=RequestID)

    @property
    def request_id(self) -> RequestID:
        return self._request_id

    def __str__(self) -> str:
        return str(self._request_id)


class WhisperServeTokenizer:
    """Turns an uploaded audio file into a :class:`SpeechToTextContext`."""

    def __init__(
        self,
        model_path: str,
        pipeline_config: Any = None,
        *,
        subfolder: str | None = None,
        revision: str | None = None,
        max_length: int | None = None,
        trust_remote_code: bool = False,
        language: str = "en",
        **unused_kwargs: Any,
    ) -> None:
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoConfig,
            GenerationConfig,
            WhisperFeatureExtractor,
            WhisperTokenizerFast,
        )

        model_dir = (
            model_path
            if os.path.isdir(model_path)
            else snapshot_download(model_path, revision=revision)
        )
        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = WhisperTokenizerFast.from_pretrained(model_dir)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            model_dir
        )
        self.gen = GenerationConfig.from_pretrained(model_dir)
        self._model_name = model_path
        self._default_language = language
        self._eos_id = self.config.eos_token_id

    # ------------------------------------------------------------------ #
    # PipelineTokenizer protocol
    # ------------------------------------------------------------------ #
    @property
    def eos(self) -> int:
        return self._eos_id

    @property
    def expects_content_wrapping(self) -> bool:
        return False

    def _build_sot(self, language: str) -> list[int]:
        """SOT prompt: <sot> <lang> <transcribe> <notimestamps>."""
        try:
            lang_id = self.gen.lang_to_id[f"<|{language}|>"]
        except (KeyError, TypeError) as e:
            raise InputError(
                f"Unsupported transcription language {language!r}."
            ) from e
        task_id = self.gen.task_to_id["transcribe"]
        return [
            self.config.decoder_start_token_id,
            lang_id,
            task_id,
            self.gen.no_timestamps_token_id,
        ]

    def _build_context(
        self, request: TranscriptionRequest
    ) -> SpeechToTextContext:
        """Blocking audio decode + feature extraction (runs off the event loop)."""
        audio = load_audio_bytes(request.audio_bytes)
        duration = len(audio) / _SAMPLE_RATE
        if duration > _MAX_AUDIO_SECONDS:
            raise InputError(
                f"Audio is {duration:.1f}s; this endpoint transcribes a single "
                f"<=30s window. Chunk long audio client-side into <=30s "
                "segments and post them concurrently."
            )
        mel, num_content_frames = extract_features(
            audio, self.feature_extractor
        )
        language = request.language or self._default_language
        sot = self._build_sot(language)
        return SpeechToTextContext(
            mel=mel[0],
            num_content_frames=num_content_frames,
            duration_s=duration,
            language=language,
            prompt_tokens=sot,
            word_timestamps=request.word_timestamps,
            model_name=request.model,
            request_id=request.request_id,
        )

    async def new_context(
        self, request: TranscriptionRequest
    ) -> SpeechToTextContext:
        """Decode audio + extract mel in a worker thread, return the context."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._build_context, request
        )

    async def encode(
        self, prompt: str, add_special_tokens: bool = True
    ) -> list[int]:
        raise NotImplementedError(
            "WhisperServeTokenizer transcribes audio; it has no text encode path."
        )

    async def decode(self, encoded: Any, **kwargs: Any) -> str:
        """Decode transcript token ids back to text (skip specials by default)."""
        kwargs.setdefault("skip_special_tokens", True)
        return self.tokenizer.decode(encoded, **kwargs)
