"""
Async Pipeline Worker
Consumes messages from Redis Streams and executes the LangGraph pipeline.
Handles retries, DLQ routing, status updates, and metrics.
"""
from __future__ import annotations

import asyncio
import signal
import time
import uuid

import structlog
from prometheus_client import Counter, Gauge, Histogram, start_http_server

from config.settings import get_settings
from src.adapters.api.dependencies import get_container
from src.core.state.pipeline_state import PipelineState
from src.domain.entities.models import DocumentId, DocumentStatus

log = structlog.get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Worker Prometheus metrics
# ---------------------------------------------------------------------------

WORKER_TASKS_PROCESSED = Counter(
    "worker_tasks_processed_total",
    "Total tasks processed by the worker",
    ["status"],
)
WORKER_PIPELINE_DURATION = Histogram(
    "worker_pipeline_duration_seconds",
    "Full pipeline execution duration",
    ["final_status"],
    buckets=[1, 5, 10, 30, 60, 120, 300, 600],
)
WORKER_ACTIVE_TASKS = Gauge(
    "worker_active_tasks",
    "Number of currently running pipeline tasks",
)
WORKER_DLQ_TOTAL = Counter(
    "worker_dlq_total",
    "Total tasks sent to DLQ",
)


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

class PipelineWorker:
    """
    Pulls one task at a time from Redis Streams, executes the LangGraph
    pipeline, and handles all status transitions.

    Designed for horizontal scaling: run N containers of this worker.
    Each container has its own consumer identity within the shared group.
    """

    def __init__(self) -> None:
        self._running = False
        self._container = None

    async def start(self) -> None:
        self._container = get_container()
        await self._container.startup()

        # Start Prometheus metrics server on a separate port
        start_http_server(settings.observability.prometheus_port + 1)

        self._running = True
        log.info(
            "worker.started",
            queue=settings.worker.queue_name,
            concurrency=settings.worker.concurrency,
        )

        # Set up graceful shutdown
        loop = asyncio.get_event_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self._shutdown)

        # Run N concurrent consumers
        await asyncio.gather(*[
            self._consume_loop(worker_id=i)
            for i in range(settings.worker.concurrency)
        ])

    def _shutdown(self) -> None:
        log.info("worker.shutdown_signal")
        self._running = False

    async def _consume_loop(self, worker_id: int) -> None:
        log.info("worker.consumer.started", worker_id=worker_id)

        while self._running:
            try:
                message = await self._container.queue.dequeue(
                    queue=settings.worker.queue_name,
                    timeout=5,
                )
                if message is None:
                    continue

                WORKER_ACTIVE_TASKS.inc()
                try:
                    await self._process(message)
                finally:
                    WORKER_ACTIVE_TASKS.dec()

            except Exception as exc:
                log.error("worker.consumer.error", worker_id=worker_id, exc=str(exc))
                await asyncio.sleep(1)

        log.info("worker.consumer.stopped", worker_id=worker_id)

    async def _process(self, message: dict) -> None:
        job_id = message.get("job_id", "unknown")
        document_id = message.get("document_id", "unknown")
        stream_id = message.get("__stream_id")
        stream = message.get("__stream")

        log.info("worker.task.start", job_id=job_id, document_id=document_id)
        start_time = time.monotonic()

        # Build initial pipeline state from queue message
        initial_state: PipelineState = {
            "job_id": job_id,
            "document_id": document_id,
            "pipeline_run_id": str(uuid.uuid4()),
            "raw_file_key": message.get("raw_file_key", ""),
            "source_filename": message.get("source_filename", ""),
            "content_type": message.get("content_type", ""),
            "file_size_bytes": message.get("file_size_bytes", 0),
            "detected_file_type": message.get("file_type", "unknown"),
            "user_tags": message.get("user_tags", []),
            "user_metadata": message.get("user_metadata", {}),
            "retry_count": message.get("retry_count", 0),
            "errors": [],
            "current_status": DocumentStatus.EXTRACTING.value,
        }

        # Update document status to EXTRACTING
        await self._container.doc_repo.update_status(
            DocumentId(document_id), DocumentStatus.EXTRACTING
        )

        try:
            # Execute LangGraph pipeline
            config = {
                "configurable": {
                    "thread_id": initial_state["pipeline_run_id"],
                },
                "recursion_limit": 50,
            }

            async for event in self._container.graph.astream(
                initial_state,
                config=config,
                stream_mode="updates",
            ):
                # Sync status changes to Postgres in real-time
                for node_name, node_output in event.items():
                    if new_status := node_output.get("current_status"):
                        await self._container.doc_repo.update_status(
                            DocumentId(document_id),
                            DocumentStatus(new_status),
                        )
                        # Sync job status
                        job = await self._container.job_repo.get_by_id(job_id)
                        if job:
                            job.status = DocumentStatus(new_status)
                            await self._container.job_repo.update(job)

                    log.debug(
                        "worker.node.complete",
                        node=node_name,
                        document_id=document_id,
                    )

            # Ack the message only after successful completion
            if stream_id and stream:
                await self._container.queue._client.xack(
                    stream,
                    self._container.queue.CONSUMER_GROUP,
                    stream_id,
                )

            duration = time.monotonic() - start_time
            WORKER_TASKS_PROCESSED.labels(status="success").inc()
            WORKER_PIPELINE_DURATION.labels(final_status="completed").observe(duration)

            log.info(
                "worker.task.done",
                job_id=job_id,
                document_id=document_id,
                duration_s=round(duration, 2),
            )

        except Exception as exc:
            duration = time.monotonic() - start_time
            log.error(
                "worker.task.failed",
                job_id=job_id,
                document_id=document_id,
                exc=str(exc),
                duration_s=round(duration, 2),
            )

            WORKER_TASKS_PROCESSED.labels(status="failed").inc()
            WORKER_PIPELINE_DURATION.labels(final_status="failed").observe(duration)
            WORKER_DLQ_TOTAL.inc()

            # Ack even on failure — DLQ is handled inside the graph
            if stream_id and stream:
                try:
                    await self._container.queue._client.xack(
                        stream,
                        self._container.queue.CONSUMER_GROUP,
                        stream_id,
                    )
                except Exception:
                    pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    import logging
    import structlog

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )

    worker = PipelineWorker()
    await worker.start()


if __name__ == "__main__":
    asyncio.run(main())
