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
"""Whisper SPEECH_TO_TEXT architecture registration for ``max serve``.

Registers ``WhisperForConditionalGeneration`` as a SPEECH_TO_TEXT architecture
wiring the served executor (:class:`~.executor.WhisperExecutor`), the audio
front-end tokenizer (:class:`~.serve_tokenizer.WhisperServeTokenizer`), and the
:class:`~max.pipelines.context.SpeechToTextContext` carrier. Kept in its own
module (lazily imported by ``architectures/__init__.py``) so importing the
registry doesn't pull in transformers / soundfile.
"""

from __future__ import annotations

from dataclasses import dataclass

from max.graph.weights import WeightsFormat
from max.pipelines.context import SpeechToTextContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.config import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from max.pipelines.modeling.types import InputModality, PipelineTask
from typing_extensions import Self

from .executor import WhisperExecutor
from .serve_tokenizer import WhisperServeTokenizer

# Whisper's decoder position table is fixed at 448 across every checkpoint
# (tiny .. large-v3), so the transcript can never exceed it.
WHISPER_MAX_TARGET_POSITIONS: int = 448


@dataclass(kw_only=True)
class WhisperArchConfig(ArchConfig):
    """Pipeline-level config for Whisper (implements ArchConfig; no KV cache).

    The KV cache here is a private, fixed-size decoder self-attention cache the
    executor manages internally — not the paged LLM cache the framework plans,
    so this config exposes only the max sequence length.
    """

    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        return WHISPER_MAX_TARGET_POSITIONS

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        return cls(pipeline_config=pipeline_config)


whisper_arch = SupportedArchitecture(
    name="WhisperForConditionalGeneration",
    task=PipelineTask.SPEECH_TO_TEXT,
    input_modalities={InputModality.AUDIO},
    default_encoding="float32",
    # bf16 is opt-in (e.g. --model-override main.quantization_encoding=bfloat16);
    # the executor currently applies it to the encoder (output re-cast to f32).
    supported_encodings={"float32", "bfloat16"},
    example_repo_ids=[
        "openai/whisper-large-v3",
        "openai/whisper-large-v3-turbo",
        "openai/whisper-tiny",
    ],
    pipeline_model=WhisperExecutor,
    context_type=SpeechToTextContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=WhisperServeTokenizer,
    config=WhisperArchConfig,
)
