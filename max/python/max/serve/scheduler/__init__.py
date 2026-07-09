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

import contextlib
import logging
from collections.abc import AsyncGenerator, Hashable
from typing import Any, cast

_logger = logging.getLogger("max.pipelines")

from max.pipelines.context import (
    BaseContextType,
    PixelContext,
    TextContext,
    TextGenerationOutput,
)
from max.pipelines.context.outputs import GenerationOutput
from max.pipelines.diffusion.pipeline import (
    PixelGenerationPipeline,
)
from max.pipelines.lib import (
    EmbeddingsPipelineType,
    PipelineConfig,
    TextGenerationPipeline,
)
from max.pipelines.modeling.types import (
    EmbeddingsContext,
    EmbeddingsGenerationOutput,
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
    PixelGenerationInputs,
    RequestID,
)
from max.serve.config import Settings
from max.serve.queue import MAXPullQueue, MAXPushQueue
from max.serve.scheduler.interface import Scheduler
from max.serve.scheduler_result import SchedulerResult
from max.serve.worker_interface import WorkerQueues

from .base import CancelRequest, PrefillRequest, PrefillResponse
from .config import TokenGenerationSchedulerConfig
from .decode_scheduler import load_decode_scheduler
from .embeddings_scheduler import EmbeddingsScheduler, EmbeddingsSchedulerConfig
from .one_shot_scheduler import OneShotScheduler
from .prefill_scheduler import load_prefill_scheduler
from .text_generation_scheduler import load_text_generation_scheduler

__all__ = [
    "CancelRequest",
    "EmbeddingsScheduler",
    "EmbeddingsSchedulerConfig",
    "OneShotScheduler",
    "PrefillRequest",
    "PrefillResponse",
    "TokenGenerationSchedulerConfig",
    "load_scheduler",
]


def load_scheduler(
    pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
    pipeline_config: PipelineConfig,
    settings: Settings,
    worker_queues: WorkerQueues[BaseContextType, PipelineOutputType],
) -> Scheduler:
    request_queue = worker_queues.request_queue
    response_queue = worker_queues.response_queue
    cancel_queue = worker_queues.cancel_queue

    _logger.info("max_batch_size: %d", pipeline.max_batch_size)

    if pipeline.__class__.__name__ == "PixelGenerationPipeline":
        pixel_pipeline = cast(PixelGenerationPipeline[Any], pipeline)

        def batch_constructor(
            contexts: list[PixelContext],
        ) -> PixelGenerationInputs[Any]:
            """Collate one or more PixelContexts into PixelGenerationInputs."""
            return PixelGenerationInputs(
                batch={context.request_id: context for context in contexts}
            )

        def batch_key(context: PixelContext) -> Hashable:
            """Group requests that a single batched denoise loop can serve.

            Requests batch only when resolution, steps and num_images match.
            Image-to-image and CFG (classifier-free guidance) requests are
            always solo (unique key): the batched denoise path is text-to-image,
            non-CFG only, so grouping a CFG request would hit the
            ``prepare_inputs_batched`` guard and 500. Solo requests fall through
            to the single-request path, which supports CFG.
            """
            if getattr(context, "input_image", None) is not None:
                return ("solo", context.request_id)
            # CFG here mirrors the pipeline's batched guard: guidance>0 with a
            # negative stream present. Run each such request on its own.
            if (
                context.guidance_scale > 0.0
                and getattr(context, "negative_tokens", None) is not None
            ):
                return ("solo", context.request_id)
            return (
                context.height,
                context.width,
                context.num_inference_steps,
                context.num_images_per_prompt,
                context.negative_tokens is not None,
                context.guidance_scale > 1.0,
            )

        return OneShotScheduler[
            PixelContext, PixelGenerationInputs[Any], GenerationOutput
        ](
            pipeline=pixel_pipeline,
            batch_constructor=batch_constructor,
            request_queue=cast(
                MAXPullQueue[PixelContext],
                request_queue,
            ),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[GenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
            max_batch_size=pipeline.max_batch_size,
            batch_key=batch_key,
        )
    elif pipeline.__class__.__name__ == "EmbeddingsPipeline":
        embeddings_scheduler_config = EmbeddingsSchedulerConfig(
            max_batch_size=pipeline.max_batch_size
        )
        emb_pipeline = cast(EmbeddingsPipelineType, pipeline)
        return EmbeddingsScheduler(
            scheduler_config=embeddings_scheduler_config,
            pipeline=emb_pipeline,
            request_queue=cast(
                MAXPullQueue[EmbeddingsContext],
                request_queue,
            ),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[EmbeddingsGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
        )
    elif pipeline_config.runtime.pipeline_role == "prefill_and_decode":
        text_pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
        return load_text_generation_scheduler(
            text_pipeline,
            pipeline_config,
            request_queue=cast(MAXPullQueue[TextContext], request_queue),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[TextGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
        )
    elif pipeline_config.runtime.pipeline_role == "decode_only":
        text_pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
        return load_decode_scheduler(
            text_pipeline,
            pipeline_config,
            request_queue=cast(MAXPullQueue[TextContext], request_queue),
            response_queue=cast(
                MAXPushQueue[
                    dict[RequestID, SchedulerResult[TextGenerationOutput]]
                ],
                response_queue,
            ),
            cancel_queue=cancel_queue,
            settings=settings,
        )
    elif pipeline_config.runtime.pipeline_role == "prefill_only":
        text_pipeline = cast(TextGenerationPipeline[TextContext], pipeline)
        return load_prefill_scheduler(text_pipeline, pipeline_config, settings)
    else:
        raise ValueError(
            f"No scheduler support for pipeline_role ({pipeline_config.runtime.pipeline_role})."
        )
