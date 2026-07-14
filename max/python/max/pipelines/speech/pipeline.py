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
"""MAX pipeline for speech-to-text (Whisper) transcription.

Structural mirror of ``PixelGenerationPipeline`` but executor-path only: it owns
an :class:`~max.pipelines.lib.pipeline_executor.PipelineExecutor` (the Whisper
executor), flattens the scheduler's batch to a context list, runs one batched
``prepare_inputs``/``execute``, and maps the row-ordered result back to per-
request :class:`~max.pipelines.context.outputs.SpeechToTextOutput`.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Generic

from max.driver import load_devices
from max.pipelines.context import GenerationStatus, SpeechToTextContextType
from max.pipelines.context.outputs import SpeechToTextOutput
from max.pipelines.modeling.types import (
    Pipeline,
    PipelineOutputsDict,
    RequestID,
    SpeechToTextInputs,
)

if TYPE_CHECKING:
    from max.pipelines.lib.config import PipelineConfig
    from max.pipelines.lib.pipeline_executor import PipelineExecutor

_logger = logging.getLogger("max.pipelines")


class SpeechToTextPipeline(
    Pipeline[SpeechToTextInputs[SpeechToTextContextType], SpeechToTextOutput],
    Generic[SpeechToTextContextType],
):
    """Speech-to-text pipeline driving a Whisper transcription executor."""

    def __init__(
        self,
        pipeline_config: PipelineConfig,
        pipeline_model: type[PipelineExecutor[Any, Any, Any]],
    ) -> None:
        from max.engine import InferenceSession  # local import to avoid cycles

        self._pipeline_config = pipeline_config
        first_config = next(iter(pipeline_config.models.values()))
        self._devices = load_devices(first_config.device_specs)
        session = InferenceSession(devices=[*self._devices])
        self._pipeline_config.configure_session(session)
        self._executor = pipeline_model(
            manifest=pipeline_config.models,
            session=session,
            runtime_config=pipeline_config.runtime,
        )

    @property
    def pipeline_config(self) -> PipelineConfig:
        """Return the pipeline configuration."""
        return self._pipeline_config

    @property
    def max_batch_size(self) -> int:
        """Max requests the scheduler may batch into one transcription.

        Defaults to 8; set ``MODULAR_WHISPER_MAX_BATCH_SIZE`` to change it.
        Honored only when the executor advertises ``supports_dynamic_batching``.
        """
        try:
            requested = int(
                os.environ.get("MODULAR_WHISPER_MAX_BATCH_SIZE", "8")
            )
        except ValueError:
            requested = 8
        if requested <= 1:
            return 1
        if getattr(self._executor, "supports_dynamic_batching", False):
            return requested
        return 1

    def execute(
        self,
        inputs: SpeechToTextInputs[SpeechToTextContextType],
    ) -> PipelineOutputsDict[SpeechToTextOutput]:
        """Transcribe a (possibly batched) group of requests."""
        batch = inputs.batch
        if not batch:
            return {}

        flat_batch = list(batch.items())
        contexts = [ctx for _rid, ctx in flat_batch]
        model_inputs = self._executor.prepare_inputs(contexts)
        result = self._executor.execute(model_inputs)

        responses: dict[RequestID, SpeechToTextOutput] = {}
        for index, (request_id, ctx) in enumerate(flat_batch):
            words = result.words[index] if result.words is not None else None
            responses[request_id] = SpeechToTextOutput(
                request_id=request_id,
                final_status=GenerationStatus.END_OF_SEQUENCE,
                text=result.texts[index],
                language=ctx.language,
                duration=ctx.duration_s,
                words=words,
            )
        return responses

    def release(self, request_id: RequestID) -> None:
        """Release resources for a request (one-shot; nothing to free)."""
        pass
