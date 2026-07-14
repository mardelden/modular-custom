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
"""Pipeline inputs for the speech-to-text (Whisper) task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

from max.pipelines.request import RequestID

from ..pipeline import PipelineInputs

if TYPE_CHECKING:
    from max.pipelines.context.context import SpeechToTextContext

SpeechToTextContextType = TypeVar("SpeechToTextContextType")


@dataclass(frozen=True)
class SpeechToTextInputs(PipelineInputs, Generic[SpeechToTextContextType]):
    """Input data structure for speech-to-text pipelines.

    Mirrors :class:`~max.pipelines.modeling.types.pipeline_variants.pixel_generation.PixelGenerationInputs`:
    a one-shot batch of contexts keyed by request id.
    """

    batch: dict[RequestID, SpeechToTextContextType]
    """A dictionary mapping RequestID to speech-to-text context instances."""
