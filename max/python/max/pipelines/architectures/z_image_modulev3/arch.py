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

from __future__ import annotations

from dataclasses import dataclass

from max.graph.weights import WeightsFormat
from max.pipelines.context import PixelContext
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.config import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from max.pipelines.modeling.types import PipelineTask
from typing_extensions import Self

from .pipeline_z_image import ZImagePipeline
from .tokenizer import ZImageTokenizer


@dataclass(kw_only=True)
class ZImageArchConfig(ArchConfig):
    pipeline_config: PipelineConfig

    def get_max_seq_len(self) -> int:
        return 0

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
    ) -> Self:
        # Do not read ``pipeline_config.model`` here: diffusion pipelines have
        # no single "main" model and that accessor raises "No main model
        # configured". The single-device constraint is validated in
        # ``ZImagePipeline.init_remaining_components`` (which has
        # ``transformer.devices``). Mirrors ``Flux2ArchConfig.initialize``.
        return cls(pipeline_config=pipeline_config)


z_image_arch = SupportedArchitecture(
    name="ZImagePipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    example_repo_ids=[
        "Tongyi-MAI/Z-Image",
        "Tongyi-MAI/Z-Image-Turbo",
        "Zyphra/Z-Image",
    ],
    pipeline_model=ZImagePipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=ZImageTokenizer,
    config=ZImageArchConfig,
)
