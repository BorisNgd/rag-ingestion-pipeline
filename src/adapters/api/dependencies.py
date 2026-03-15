"""
Dependency Injection Container
Single place where all services are instantiated and wired together.
Use get_container() everywhere — never construct services ad-hoc.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import structlog

from config.settings import get_settings
from src.application.use_cases.ingestion import (
    GetJobStatusUseCase,
    HybridSearchUseCase,
    IngestDocumentUseCase,
    RequeueDLQUseCase,
)
from src.core.graph.pipeline_graph import build_pipeline_graph, create_checkpointer
from src.infrastructure.database.postgres_repositories import (
    PostgresChunkRepository,
    PostgresDocumentRepository,
    PostgresIngestionJobRepository,
    create_engine,
    create_tables,
)
from src.infrastructure.embedding.hybrid_embedder import HybridEmbedder
from src.infrastructure.messaging.redis_queue import (
    RedisCacheRepository,
    RedisMessageQueueRepository,
)
from src.infrastructure.storage.minio_repository import MinioObjectStorageRepository
from src.infrastructure.vector_store.qdrant_repository import QdrantVectorRepository
from src.adapters.parsers.registry import ParserRegistry
from src.adapters.processors.registry import (
    ChunkerRegistry,
    DeduplicatorService,
    LanguageDetectorService,
    NERExtractorService,
    PIIRedactorService,
    SummarizerService,
)

log = structlog.get_logger(__name__)
settings = get_settings()


class Container:
    """
    Wires the entire application. Handles startup/shutdown lifecycle.
    All attributes are lazy-initialized on first access.
    """

    def __init__(self) -> None:
        self._initialized = False

        # Infrastructure
        self._engine = None
        self._cache = None
        self._queue = None
        self._storage = None
        self._vector = None

        # Repositories
        self._doc_repo = None
        self._chunk_repo = None
        self._job_repo = None

        # Services
        self._embedder = None
        self._parser_registry = None
        self._chunker_registry = None
        self._language_detector = None
        self._ner_extractor = None
        self._summarizer = None
        self._deduplicator = None
        self._pii_redactor = None

        # Graph
        self._graph = None
        self._checkpointer = None

        # Use cases
        self._ingest_use_case = None
        self._job_status_use_case = None
        self._search_use_case = None
        self._requeue_use_case = None

    async def startup(self) -> None:
        log.info("container.startup.begin")

        # --- Database ---
        self._engine = create_engine()
        await create_tables(self._engine)

        # --- Cache & Queue ---
        self._cache = RedisCacheRepository.create()
        self._queue = RedisMessageQueueRepository.create()

        # --- Storage ---
        self._storage = MinioObjectStorageRepository.create()
        await self._storage.ensure_buckets()

        # --- Vector Store ---
        self._vector = QdrantVectorRepository.create()

        # --- Repositories ---
        self._doc_repo = PostgresDocumentRepository(self._engine)
        self._chunk_repo = PostgresChunkRepository(self._engine)
        self._job_repo = PostgresIngestionJobRepository(self._engine)

        # --- Services ---
        self._embedder = HybridEmbedder(cache=self._cache)
        self._parser_registry = ParserRegistry(storage=self._storage)
        self._chunker_registry = ChunkerRegistry()
        self._language_detector = LanguageDetectorService()
        self._ner_extractor = NERExtractorService()
        self._summarizer = SummarizerService()
        self._deduplicator = DeduplicatorService(embedder=self._embedder)
        self._pii_redactor = PIIRedactorService()

        # --- LangGraph checkpointer ---
        if settings.worker.checkpointer_backend == "postgres":
            self._checkpointer = await create_checkpointer(
                settings.postgres.sync_dsn
            )

        # --- Build pipeline graph ---
        self._graph = build_pipeline_graph(
            parser_registry=self._parser_registry,
            chunker_registry=self._chunker_registry,
            language_detector=self._language_detector,
            ner_extractor=self._ner_extractor,
            summarizer=self._summarizer,
            deduplicator=self._deduplicator,
            pii_redactor=self._pii_redactor,
            embedder=self._embedder,
            vector_repo=self._vector,
            queue_repo=self._queue,
            checkpointer=self._checkpointer,
        )

        # --- Use cases ---
        self._ingest_use_case = IngestDocumentUseCase(
            doc_repo=self._doc_repo,
            job_repo=self._job_repo,
            storage=self._storage,
            queue=self._queue,
        )
        self._job_status_use_case = GetJobStatusUseCase(
            job_repo=self._job_repo,
            doc_repo=self._doc_repo,
        )
        self._search_use_case = HybridSearchUseCase(
            vector_repo=self._vector,
            embedder=self._embedder,
        )
        self._requeue_use_case = RequeueDLQUseCase(queue=self._queue)

        self._initialized = True
        log.info("container.startup.done")

    async def shutdown(self) -> None:
        log.info("container.shutdown")
        if self._engine:
            await self._engine.dispose()

    async def health_check(self) -> dict[str, bool]:
        checks: dict[str, bool] = {}

        # Postgres
        try:
            async with self._engine.connect() as conn:
                await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
            checks["postgres"] = True
        except Exception:
            checks["postgres"] = False

        # Redis
        try:
            await self._cache._client.ping()
            checks["redis"] = True
        except Exception:
            checks["redis"] = False

        # Qdrant
        try:
            await self._vector._client.get_collections()
            checks["qdrant"] = True
        except Exception:
            checks["qdrant"] = False

        # MinIO
        try:
            await self._storage._client.bucket_exists(settings.minio.bucket_raw)
            checks["minio"] = True
        except Exception:
            checks["minio"] = False

        return checks

    # --- Properties ---

    @property
    def graph(self):
        return self._graph

    @property
    def queue(self):
        return self._queue

    @property
    def doc_repo(self):
        return self._doc_repo

    @property
    def chunk_repo(self):
        return self._chunk_repo

    @property
    def job_repo(self):
        return self._job_repo

    @property
    def ingest_use_case(self) -> IngestDocumentUseCase:
        return self._ingest_use_case

    @property
    def job_status_use_case(self) -> GetJobStatusUseCase:
        return self._job_status_use_case

    @property
    def search_use_case(self) -> HybridSearchUseCase:
        return self._search_use_case

    @property
    def requeue_use_case(self) -> RequeueDLQUseCase:
        return self._requeue_use_case


# ---------------------------------------------------------------------------
# Singleton accessor (used as FastAPI dependency)
# ---------------------------------------------------------------------------

_container: Container | None = None


def get_container() -> Container:
    global _container
    if _container is None:
        _container = Container()
    return _container
