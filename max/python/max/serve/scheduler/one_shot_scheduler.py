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
"""OneShotScheduler for non-autoregressive pipelines.

This scheduler is designed for pipelines that process requests in a single pass
without requiring iterative generation (e.g., image generation, non-autoregressive
text models). It processes each request serially, making it simple and suitable
for workloads that don't benefit from batching or continuous generation.
"""

import collections
import logging
import queue
from collections.abc import Callable, Hashable
from typing import Generic

from max.pipelines.context import BaseContextType
from max.pipelines.modeling.types import (
    Pipeline,
    PipelineInputsType,
    PipelineOutputType,
    RequestID,
)
from max.profiler import traced
from max.serve.queue import MAXPullQueue, MAXPushQueue
from max.serve.scheduler.interface import Scheduler
from max.serve.scheduler_result import SchedulerResult

from .base import SchedulerProgress

logger = logging.getLogger("max.serve")


class OneShotScheduler(
    Scheduler,
    Generic[BaseContextType, PipelineInputsType, PipelineOutputType],
):
    """Scheduler for non-autoregressive pipelines that processes requests serially.

    This scheduler is optimized for pipelines that:
    - Complete in a single forward pass (no iterative generation)
    - Don't require KV cache management
    - Process each request independently

    The scheduler pulls one request at a time from the queue, executes the pipeline,
    and returns the result. This simple approach is suitable for image generation,
    embeddings with small batch sizes, and other non-autoregressive workloads.

    Args:
        pipeline: The pipeline to execute requests with
        batch_constructor: Callable that converts a request context into pipeline inputs.
            Takes a BaseContextType and returns a PipelineInputsType.
        request_queue: Queue to pull incoming requests from
        response_queue: Queue to push completed responses to
        cancel_queue: Queue for handling request cancellations
        max_batch_size: Maximum number of requests to process in a single batch.
            Defaults to 1 for serial processing.
    """

    def __init__(
        self,
        pipeline: Pipeline[PipelineInputsType, PipelineOutputType],
        batch_constructor: Callable[
            [list[BaseContextType]], PipelineInputsType
        ],
        request_queue: MAXPullQueue[BaseContextType],
        response_queue: MAXPushQueue[
            dict[RequestID, SchedulerResult[PipelineOutputType]]
        ],
        cancel_queue: MAXPullQueue[list[RequestID]],
        max_batch_size: int = 1,
        batch_key: Callable[[BaseContextType], Hashable] | None = None,
    ) -> None:
        self.max_batch_size = max(1, max_batch_size)
        self.pipeline = pipeline
        self.batch_constructor = batch_constructor
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue
        # When set (and max_batch_size > 1), requests whose ``batch_key``
        # matches are executed together in one batch. Requests pulled from
        # the queue but not yet dispatched wait here across iterations.
        self.batch_key = batch_key
        self._pending: collections.deque[BaseContextType] = (
            collections.deque()
        )
        # Upper bound on how many queued requests to buffer while forming a
        # batch, so a flood of incompatible requests can't grow unbounded.
        self._drain_cap = max(self.max_batch_size * 8, 64)

    @traced
    def _get_next_request(self) -> BaseContextType | None:
        """Pull the next request from the queue.

        Returns:
            The next context to process, or None if the queue is empty.
        """
        try:
            return self.request_queue.get_nowait()
        except queue.Empty:
            return None

    def _next_group(self) -> list[BaseContextType]:
        """Return the next batch of compatible requests to execute.

        Drains newly-queued requests into the pending buffer, then forms a
        group from the front: the first request plus any later pending
        requests sharing its ``batch_key``, up to ``max_batch_size``.
        Non-matching requests stay pending (in order) for later iterations.
        """
        while len(self._pending) < self._drain_cap:
            context = self._get_next_request()
            if context is None:
                break
            self._pending.append(context)

        if not self._pending:
            return []

        first = self._pending.popleft()
        group = [first]
        if self.max_batch_size > 1 and self.batch_key is not None:
            key = self.batch_key(first)
            leftover: collections.deque[BaseContextType] = (
                collections.deque()
            )
            while self._pending and len(group) < self.max_batch_size:
                context = self._pending.popleft()
                if self.batch_key(context) == key:
                    group.append(context)
                else:
                    leftover.append(context)
            leftover.extend(self._pending)
            self._pending = leftover
        return group

    def run_iteration(self) -> SchedulerProgress:
        """Execute one scheduling iteration.

        Forms a batch of compatible queued requests, executes them through the
        pipeline in a single pass, and sends the responses back.

        Returns:
            SchedulerProgress.MADE_PROGRESS if a request was processed,
            SchedulerProgress.NO_PROGRESS if no requests were available.
        """
        group = self._next_group()
        if not group:
            return SchedulerProgress.NO_PROGRESS

        request_ids = [context.request_id for context in group]
        logger.info(
            "OneShotScheduler: Starting %d request(s): %s",
            len(group),
            ", ".join(str(rid) for rid in request_ids),
        )

        try:
            # Convert the contexts to pipeline inputs and execute in one pass.
            pipeline_inputs = self.batch_constructor(group)
            responses = self.pipeline.execute(pipeline_inputs)

            logger.info(
                "OneShotScheduler: Completed %d request(s) with %d response(s)",
                len(group),
                len(responses),
            )

            self.response_queue.put_nowait(
                {
                    request_id: SchedulerResult.create(response)
                    for request_id, response in responses.items()
                }
            )
        except Exception:
            logger.exception(
                "OneShotScheduler: Exception during pipeline execution for "
                "request(s) %s",
                ", ".join(str(rid) for rid in request_ids),
            )

            # Cancel every request in the failed batch (errors logged above).
            self.response_queue.put_nowait(
                {rid: SchedulerResult.cancelled() for rid in request_ids}
            )

        return SchedulerProgress.MADE_PROGRESS
