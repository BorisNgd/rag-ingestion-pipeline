"""
PostgreSQL Repository Implementations
Uses SQLAlchemy async core (no ORM bloat) for full control.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import AsyncIterator, Any

import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine

from config.settings import get_settings
from src.domain.entities.models import (
    Chunk,
    ChunkMetadata,
    ChunkId,
    Document,
    DocumentId,
    DocumentMetadata,
    DocumentStatus,
    FileType,
    IngestionJob,
    ProcessingError,
    RAGCollection,
    StorageLocation,
    ChunkingStrategy,
)
from src.domain.repositories.interfaces import (
    ChunkRepository,
    DocumentRepository,
    IngestionJobRepository,
)

settings = get_settings()

# ---------------------------------------------------------------------------
# Schema definitions
# ---------------------------------------------------------------------------

metadata = sa.MetaData()

documents_table = sa.Table(
    "documents",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("status", sa.String(32), nullable=False, index=True),
    sa.Column("file_type", sa.String(32)),
    sa.Column("source_filename", sa.String(1024)),
    sa.Column("content_type", sa.String(256)),
    sa.Column("size_bytes", sa.BigInteger),
    sa.Column("language", sa.String(16)),
    sa.Column("detected_language", sa.String(16)),
    sa.Column("rag_collection", sa.String(64), index=True),
    sa.Column("classification_confidence", sa.Float),
    sa.Column("classification_reasoning", sa.Text),
    sa.Column("chunking_strategy", sa.String(32)),
    sa.Column("chunking_reasoning", sa.Text),
    sa.Column("raw_bucket", sa.String(256)),
    sa.Column("raw_key", sa.String(1024)),
    sa.Column("retry_count", sa.Integer, default=0),
    sa.Column("errors", sa.JSON, default=list),
    sa.Column("tags", sa.JSON, default=list),
    sa.Column("custom_metadata", sa.JSON, default=dict),
    sa.Column("pipeline_run_id", sa.String(128)),
    sa.Column("created_at", sa.DateTime(timezone=True)),
    sa.Column("updated_at", sa.DateTime(timezone=True)),
    sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
)

chunks_table = sa.Table(
    "chunks",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("document_id", sa.String(36), sa.ForeignKey("documents.id"), index=True),
    sa.Column("rag_collection", sa.String(64), index=True),
    sa.Column("text", sa.Text),
    sa.Column("text_redacted", sa.Text),
    sa.Column("chunk_index", sa.Integer),
    sa.Column("total_chunks", sa.Integer),
    sa.Column("start_char", sa.Integer),
    sa.Column("end_char", sa.Integer),
    sa.Column("page_number", sa.Integer, nullable=True),
    sa.Column("section_title", sa.String(512), nullable=True),
    sa.Column("language", sa.String(16)),
    sa.Column("entities", sa.JSON, default=list),
    sa.Column("summary", sa.Text, nullable=True),
    sa.Column("has_pii", sa.Boolean, default=False),
    sa.Column("pii_types", sa.JSON, default=list),
    sa.Column("token_count", sa.Integer, nullable=True),
    sa.Column("qdrant_point_id", sa.String(36), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True)),
)

ingestion_jobs_table = sa.Table(
    "ingestion_jobs",
    metadata,
    sa.Column("id", sa.String(36), primary_key=True),
    sa.Column("document_id", sa.String(36), index=True),
    sa.Column("status", sa.String(32), index=True),
    sa.Column("webhook_url", sa.String(1024), nullable=True),
    sa.Column("created_at", sa.DateTime(timezone=True)),
    sa.Column("updated_at", sa.DateTime(timezone=True)),
)


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def create_engine() -> AsyncEngine:
    return create_async_engine(
        settings.postgres.dsn,
        pool_size=settings.postgres.pool_size,
        max_overflow=settings.postgres.max_overflow,
        pool_timeout=settings.postgres.pool_timeout,
        echo=settings.app.debug,
    )


async def create_tables(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


# ---------------------------------------------------------------------------
# Document Repository
# ---------------------------------------------------------------------------

class PostgresDocumentRepository(DocumentRepository):

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save(self, doc: Document) -> None:
        row = _doc_to_row(doc)
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.dialects.postgresql.insert(documents_table)
                .values(**row)
                .on_conflict_do_update(
                    index_elements=["id"], set_=row
                )
            )

    async def get_by_id(self, doc_id: DocumentId) -> Document | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                sa.select(documents_table).where(
                    documents_table.c.id == str(doc_id)
                )
            )
            row = result.mappings().first()
            return _row_to_doc(row) if row else None

    async def update_status(
        self, doc_id: DocumentId, status: DocumentStatus
    ) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.update(documents_table)
                .where(documents_table.c.id == str(doc_id))
                .values(
                    status=status.value,
                    updated_at=datetime.utcnow(),
                )
            )

    async def list_by_status(
        self, status: DocumentStatus, limit: int = 100
    ) -> list[Document]:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                sa.select(documents_table)
                .where(documents_table.c.status == status.value)
                .limit(limit)
                .order_by(documents_table.c.created_at.asc())
            )
            return [_row_to_doc(row) for row in result.mappings()]

    async def stream_all(self) -> AsyncIterator[Document]:
        async with self._engine.connect() as conn:
            async for row in await conn.stream(
                sa.select(documents_table).order_by(
                    documents_table.c.created_at.desc()
                )
            ):
                yield _row_to_doc(row._mapping)


# ---------------------------------------------------------------------------
# Chunk Repository
# ---------------------------------------------------------------------------

class PostgresChunkRepository(ChunkRepository):

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save_many(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        rows = [_chunk_to_row(c) for c in chunks]
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.dialects.postgresql.insert(chunks_table)
                .values(rows)
                .on_conflict_do_update(
                    index_elements=["id"],
                    set_={k: sa.text(f"EXCLUDED.{k}") for k in rows[0]},
                )
            )

    async def get_by_document(self, document_id: str) -> list[Chunk]:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                sa.select(chunks_table)
                .where(chunks_table.c.document_id == document_id)
                .order_by(chunks_table.c.chunk_index.asc())
            )
            return [_row_to_chunk(row) for row in result.mappings()]

    async def delete_by_document(self, document_id: str) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.delete(chunks_table).where(
                    chunks_table.c.document_id == document_id
                )
            )

    async def count_by_collection(self, collection: RAGCollection) -> int:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                sa.select(sa.func.count()).where(
                    chunks_table.c.rag_collection == collection.value
                )
            )
            return result.scalar() or 0


# ---------------------------------------------------------------------------
# IngestionJob Repository
# ---------------------------------------------------------------------------

class PostgresIngestionJobRepository(IngestionJobRepository):

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def save(self, job: IngestionJob) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.dialects.postgresql.insert(ingestion_jobs_table)
                .values(
                    id=job.id,
                    document_id=job.document_id,
                    status=job.status.value,
                    webhook_url=job.webhook_url,
                    created_at=job.created_at,
                    updated_at=job.updated_at,
                )
                .on_conflict_do_nothing()
            )

    async def get_by_id(self, job_id: str) -> IngestionJob | None:
        async with self._engine.connect() as conn:
            result = await conn.execute(
                sa.select(ingestion_jobs_table).where(
                    ingestion_jobs_table.c.id == job_id
                )
            )
            row = result.mappings().first()
            if not row:
                return None
            return IngestionJob(
                id=row["id"],
                document_id=row["document_id"],
                status=DocumentStatus(row["status"]),
                webhook_url=row["webhook_url"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )

    async def update(self, job: IngestionJob) -> None:
        async with self._engine.begin() as conn:
            await conn.execute(
                sa.update(ingestion_jobs_table)
                .where(ingestion_jobs_table.c.id == job.id)
                .values(
                    status=job.status.value,
                    updated_at=datetime.utcnow(),
                )
            )


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def _doc_to_row(doc: Document) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": str(doc.id),
        "status": doc.status.value,
        "retry_count": doc.retry_count,
        "errors": [
            {
                "stage": e.stage,
                "message": e.message,
                "traceback": e.traceback,
                "retryable": e.retryable,
            }
            for e in doc.errors
        ],
        "pipeline_run_id": doc.pipeline_run_id,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "completed_at": doc.completed_at,
    }
    if doc.metadata:
        row.update({
            "file_type": doc.metadata.file_type.value,
            "source_filename": doc.metadata.source_filename,
            "content_type": doc.metadata.content_type,
            "size_bytes": doc.metadata.size_bytes,
            "language": doc.metadata.language,
            "detected_language": doc.metadata.detected_language,
            "rag_collection": doc.metadata.rag_collection.value if doc.metadata.rag_collection else None,
            "classification_confidence": doc.metadata.classification_confidence,
            "classification_reasoning": doc.metadata.classification_reasoning,
            "tags": doc.metadata.tags,
            "custom_metadata": doc.metadata.custom,
        })
    if doc.storage:
        row.update({
            "raw_bucket": doc.storage.bucket,
            "raw_key": doc.storage.key,
        })
    if doc.chunking_strategy:
        row["chunking_strategy"] = doc.chunking_strategy.value
    if doc.chunking_reasoning:
        row["chunking_reasoning"] = doc.chunking_reasoning
    return row


def _row_to_doc(row: Any) -> Document:
    doc = Document(
        id=DocumentId(row["id"]),
        status=DocumentStatus(row["status"]),
        retry_count=row.get("retry_count", 0),
        errors=[
            ProcessingError(
                stage=e["stage"],
                message=e["message"],
                traceback=e.get("traceback"),
                retryable=e.get("retryable", True),
            )
            for e in (row.get("errors") or [])
        ],
        pipeline_run_id=row.get("pipeline_run_id"),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        completed_at=row.get("completed_at"),
    )
    if row.get("source_filename"):
        doc.metadata = DocumentMetadata(
            source_filename=row["source_filename"],
            file_type=FileType(row["file_type"]) if row.get("file_type") else FileType.UNKNOWN,
            content_type=row.get("content_type", ""),
            size_bytes=row.get("size_bytes", 0),
            language=row.get("language"),
            detected_language=row.get("detected_language"),
            rag_collection=RAGCollection(row["rag_collection"]) if row.get("rag_collection") else None,
            classification_confidence=row.get("classification_confidence"),
            classification_reasoning=row.get("classification_reasoning"),
            tags=row.get("tags") or [],
            custom=row.get("custom_metadata") or {},
        )
    if row.get("raw_bucket"):
        doc.storage = StorageLocation(
            bucket=row["raw_bucket"],
            key=row["raw_key"],
            content_type=row.get("content_type", ""),
            size_bytes=row.get("size_bytes", 0),
        )
    if row.get("chunking_strategy"):
        doc.chunking_strategy = ChunkingStrategy(row["chunking_strategy"])
    doc.chunking_reasoning = row.get("chunking_reasoning")
    return doc


def _chunk_to_row(chunk: Chunk) -> dict[str, Any]:
    m = chunk.metadata
    return {
        "id": str(chunk.id),
        "document_id": chunk.document_id,
        "rag_collection": chunk.rag_collection.value,
        "text": chunk.text,
        "text_redacted": chunk.text_redacted,
        "chunk_index": m.chunk_index if m else 0,
        "total_chunks": m.total_chunks if m else 0,
        "start_char": m.start_char if m else 0,
        "end_char": m.end_char if m else 0,
        "page_number": m.page_number if m else None,
        "section_title": m.section_title if m else None,
        "language": m.language if m else None,
        "entities": m.entities if m else [],
        "summary": m.summary if m else None,
        "has_pii": m.has_pii if m else False,
        "pii_types": m.pii_types if m else [],
        "token_count": m.token_count if m else None,
        "qdrant_point_id": chunk.qdrant_point_id,
        "created_at": chunk.created_at,
    }


def _row_to_chunk(row: Any) -> Chunk:
    return Chunk(
        id=ChunkId(row["id"]),
        document_id=row["document_id"],
        rag_collection=RAGCollection(row["rag_collection"]),
        text=row["text"],
        text_redacted=row.get("text_redacted", ""),
        metadata=ChunkMetadata(
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            total_chunks=row["total_chunks"],
            start_char=row["start_char"],
            end_char=row["end_char"],
            page_number=row.get("page_number"),
            section_title=row.get("section_title"),
            language=row.get("language"),
            entities=row.get("entities") or [],
            summary=row.get("summary"),
            has_pii=row.get("has_pii", False),
            pii_types=row.get("pii_types") or [],
            token_count=row.get("token_count"),
        ),
        qdrant_point_id=row.get("qdrant_point_id"),
        created_at=row["created_at"],
    )
