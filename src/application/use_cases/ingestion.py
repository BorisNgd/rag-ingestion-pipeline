"""
Application Use Cases
Orchestrate domain operations and infrastructure — no business logic here,
just coordination. Each use case is a single-responsibility class.
"""
from __future__ import annotations

import mimetypes
import os
import uuid
from dataclasses import dataclass

import structlog

from config.settings import get_settings
from src.domain.entities.models import (
    Document,
    DocumentId,
    DocumentMetadata,
    DocumentStatus,
    FileType,
    IngestionJob,
    RAGCollection,
    StorageLocation,
    utcnow,
)
from src.domain.repositories.interfaces import (
    DocumentRepository,
    IngestionJobRepository,
    MessageQueueRepository,
    ObjectStorageRepository,
    VectorRepository,
)
from src.infrastructure.embedding.hybrid_embedder import HybridEmbedder

log = structlog.get_logger(__name__)
settings = get_settings()


# ---------------------------------------------------------------------------
# DTOs
# ---------------------------------------------------------------------------

@dataclass
class IngestDocumentInput:
    file_bytes: bytes
    filename: str
    content_type: str | None = None
    tags: list[str] | None = None
    metadata: dict | None = None
    webhook_url: str | None = None


@dataclass
class IngestDocumentOutput:
    job_id: str
    document_id: str
    status: str


@dataclass
class SearchInput:
    query: str
    collection: RAGCollection | None = None
    limit: int = 10
    filters: dict | None = None


@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict


@dataclass
class JobStatusOutput:
    job_id: str
    document_id: str
    status: str
    errors: list[dict]
    created_at: str
    updated_at: str
    completed_at: str | None


# ---------------------------------------------------------------------------
# Use Case: Ingest Document
# ---------------------------------------------------------------------------

class IngestDocumentUseCase:
    """
    Receives raw file bytes, stores in MinIO, creates domain objects,
    and enqueues the document for async pipeline processing.
    """

    def __init__(
        self,
        doc_repo: DocumentRepository,
        job_repo: IngestionJobRepository,
        storage: ObjectStorageRepository,
        queue: MessageQueueRepository,
    ) -> None:
        self._doc_repo = doc_repo
        self._job_repo = job_repo
        self._storage = storage
        self._queue = queue

    async def execute(self, inp: IngestDocumentInput) -> IngestDocumentOutput:
        # --- Detect file type ---
        file_type = _detect_file_type(inp.filename, inp.content_type)
        content_type = inp.content_type or _content_type_for(file_type)

        # --- Create domain objects ---
        doc = Document(
            metadata=DocumentMetadata(
                source_filename=inp.filename,
                file_type=file_type,
                content_type=content_type,
                size_bytes=len(inp.file_bytes),
                tags=inp.tags or [],
                custom=inp.metadata or {},
            ),
        )

        job = IngestionJob(
            document_id=str(doc.id),
            status=DocumentStatus.PENDING,
            webhook_url=inp.webhook_url,
        )

        # --- Store raw file in MinIO ---
        raw_key = f"{str(doc.id)}/{inp.filename}"
        await self._storage.upload(
            bucket=settings.minio.bucket_raw,
            key=raw_key,
            data=inp.file_bytes,
            content_type=content_type,
            metadata={"document_id": str(doc.id), "job_id": job.id},
        )

        doc.storage = StorageLocation(
            bucket=settings.minio.bucket_raw,
            key=raw_key,
            content_type=content_type,
            size_bytes=len(inp.file_bytes),
        )

        # --- Persist document and job ---
        await self._doc_repo.save(doc)
        await self._job_repo.save(job)

        # --- Enqueue for pipeline ---
        await self._queue.enqueue(
            queue=settings.worker.queue_name,
            message={
                "job_id": job.id,
                "document_id": str(doc.id),
                "raw_file_key": raw_key,
                "source_filename": inp.filename,
                "content_type": content_type,
                "file_size_bytes": len(inp.file_bytes),
                "file_type": file_type.value,
                "user_tags": inp.tags or [],
                "user_metadata": inp.metadata or {},
                "webhook_url": inp.webhook_url,
                "retry_count": 0,
            },
        )

        log.info(
            "ingest.enqueued",
            document_id=str(doc.id),
            job_id=job.id,
            file_type=file_type.value,
        )

        return IngestDocumentOutput(
            job_id=job.id,
            document_id=str(doc.id),
            status=DocumentStatus.PENDING.value,
        )


# ---------------------------------------------------------------------------
# Use Case: Get Job Status
# ---------------------------------------------------------------------------

class GetJobStatusUseCase:

    def __init__(
        self,
        job_repo: IngestionJobRepository,
        doc_repo: DocumentRepository,
    ) -> None:
        self._job_repo = job_repo
        self._doc_repo = doc_repo

    async def execute(self, job_id: str) -> JobStatusOutput | None:
        job = await self._job_repo.get_by_id(job_id)
        if not job:
            return None

        doc = await self._doc_repo.get_by_id(DocumentId(job.document_id))

        return JobStatusOutput(
            job_id=job.id,
            document_id=job.document_id,
            status=job.status.value,
            errors=[
                {"stage": e.stage, "message": e.message}
                for e in (doc.errors if doc else [])
            ],
            created_at=job.created_at.isoformat(),
            updated_at=job.updated_at.isoformat(),
            completed_at=doc.completed_at.isoformat() if doc and doc.completed_at else None,
        )


# ---------------------------------------------------------------------------
# Use Case: Hybrid Search
# ---------------------------------------------------------------------------

class HybridSearchUseCase:

    def __init__(
        self,
        vector_repo: VectorRepository,
        embedder: HybridEmbedder,
    ) -> None:
        self._vector_repo = vector_repo
        self._embedder = embedder

    async def execute(self, inp: SearchInput) -> list[SearchResult]:
        dense, sparse = await self._embedder.embed_query(inp.query)

        # If no collection specified, search all collections
        collections = (
            [inp.collection]
            if inp.collection
            else list(RAGCollection)
        )

        all_results: list[SearchResult] = []
        for collection in collections:
            raw = await self._vector_repo.hybrid_search(
                collection=collection,
                dense_vector=dense,
                sparse_vector=sparse,
                limit=inp.limit,
                filters=inp.filters,
            )
            for r in raw:
                payload = r.get("payload", {})
                all_results.append(SearchResult(
                    chunk_id=str(r["id"]),
                    document_id=payload.get("document_id", ""),
                    text=payload.get("text", ""),
                    score=r["score"],
                    metadata={
                        k: v for k, v in payload.items()
                        if k not in ("text", "document_id")
                    },
                ))

        # Sort by RRF score descending, cap at limit
        all_results.sort(key=lambda x: x.score, reverse=True)
        return all_results[: inp.limit]


# ---------------------------------------------------------------------------
# Use Case: Requeue DLQ document
# ---------------------------------------------------------------------------

class RequeueDLQUseCase:

    def __init__(self, queue: MessageQueueRepository) -> None:
        self._queue = queue

    async def execute(self, message_id: str) -> bool:
        return await self._queue.requeue_from_dlq(message_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXTENSION_MAP: dict[str, FileType] = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".doc": FileType.DOCX,
    ".xlsx": FileType.EXCEL,
    ".xls": FileType.EXCEL,
    ".csv": FileType.EXCEL,
    ".html": FileType.HTML,
    ".htm": FileType.HTML,
    ".png": FileType.IMAGE,
    ".jpg": FileType.IMAGE,
    ".jpeg": FileType.IMAGE,
    ".tiff": FileType.IMAGE,
    ".webp": FileType.IMAGE,
    ".mp3": FileType.AUDIO,
    ".wav": FileType.AUDIO,
    ".m4a": FileType.AUDIO,
    ".ogg": FileType.AUDIO,
    ".py": FileType.CODE,
    ".js": FileType.CODE,
    ".ts": FileType.CODE,
    ".java": FileType.CODE,
    ".go": FileType.CODE,
    ".rs": FileType.CODE,
    ".cpp": FileType.CODE,
    ".c": FileType.CODE,
    ".md": FileType.MARKDOWN,
    ".markdown": FileType.MARKDOWN,
    ".txt": FileType.MARKDOWN,
}

_CONTENT_TYPE_MAP: dict[FileType, str] = {
    FileType.PDF: "application/pdf",
    FileType.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    FileType.EXCEL: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    FileType.HTML: "text/html",
    FileType.IMAGE: "image/jpeg",
    FileType.AUDIO: "audio/mpeg",
    FileType.CODE: "text/plain",
    FileType.MARKDOWN: "text/markdown",
    FileType.UNKNOWN: "application/octet-stream",
}


def _detect_file_type(filename: str, content_type: str | None) -> FileType:
    ext = os.path.splitext(filename.lower())[1]
    return _EXTENSION_MAP.get(ext, FileType.UNKNOWN)


def _content_type_for(file_type: FileType) -> str:
    return _CONTENT_TYPE_MAP.get(file_type, "application/octet-stream")
