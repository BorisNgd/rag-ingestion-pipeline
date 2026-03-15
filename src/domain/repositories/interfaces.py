"""
Repository interfaces — the domain's contracts with the outside world.
Infrastructure implements these; the domain never imports infrastructure.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator

from src.domain.entities.models import (
    Chunk,
    Document,
    DocumentId,
    DocumentStatus,
    IngestionJob,
    RAGCollection,
)


class DocumentRepository(ABC):
    """Postgres-backed persistence for Document aggregates."""

    @abstractmethod
    async def save(self, document: Document) -> None: ...

    @abstractmethod
    async def get_by_id(self, doc_id: DocumentId) -> Document | None: ...

    @abstractmethod
    async def update_status(
        self, doc_id: DocumentId, status: DocumentStatus
    ) -> None: ...

    @abstractmethod
    async def list_by_status(
        self, status: DocumentStatus, limit: int = 100
    ) -> list[Document]: ...

    @abstractmethod
    async def stream_all(self) -> AsyncIterator[Document]: ...


class ChunkRepository(ABC):
    """Postgres-backed persistence for Chunk records."""

    @abstractmethod
    async def save_many(self, chunks: list[Chunk]) -> None: ...

    @abstractmethod
    async def get_by_document(self, document_id: str) -> list[Chunk]: ...

    @abstractmethod
    async def delete_by_document(self, document_id: str) -> None: ...

    @abstractmethod
    async def count_by_collection(self, collection: RAGCollection) -> int: ...


class IngestionJobRepository(ABC):
    """Postgres-backed persistence for IngestionJob tracking."""

    @abstractmethod
    async def save(self, job: IngestionJob) -> None: ...

    @abstractmethod
    async def get_by_id(self, job_id: str) -> IngestionJob | None: ...

    @abstractmethod
    async def update(self, job: IngestionJob) -> None: ...


class ObjectStorageRepository(ABC):
    """MinIO-backed raw file storage."""

    @abstractmethod
    async def upload(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str,
        metadata: dict | None = None,
    ) -> str: ...

    @abstractmethod
    async def download(self, bucket: str, key: str) -> bytes: ...

    @abstractmethod
    async def delete(self, bucket: str, key: str) -> None: ...

    @abstractmethod
    async def presigned_url(
        self, bucket: str, key: str, expiry: int = 3600
    ) -> str: ...

    @abstractmethod
    async def move(
        self, src_bucket: str, src_key: str,
        dst_bucket: str, dst_key: str,
    ) -> None: ...


class VectorRepository(ABC):
    """Qdrant-backed vector storage with hybrid (dense+sparse) support."""

    @abstractmethod
    async def upsert_chunks(
        self, collection: RAGCollection, chunks: list[Chunk]
    ) -> None: ...

    @abstractmethod
    async def delete_by_document(
        self, collection: RAGCollection, document_id: str
    ) -> None: ...

    @abstractmethod
    async def hybrid_search(
        self,
        collection: RAGCollection,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[dict]: ...

    @abstractmethod
    async def ensure_collection(
        self, collection: RAGCollection, dense_size: int
    ) -> None: ...


class MessageQueueRepository(ABC):
    """Redis Streams / List-backed task queue."""

    @abstractmethod
    async def enqueue(
        self, queue: str, message: dict, priority: int = 0
    ) -> str: ...

    @abstractmethod
    async def dequeue(
        self, queue: str, timeout: int = 30
    ) -> dict | None: ...

    @abstractmethod
    async def enqueue_dlq(self, message: dict, reason: str) -> None: ...

    @abstractmethod
    async def list_dlq(self, limit: int = 50) -> list[dict]: ...

    @abstractmethod
    async def requeue_from_dlq(self, message_id: str) -> bool: ...


class CacheRepository(ABC):
    """Redis-backed cache for embeddings, classification results, etc."""

    @abstractmethod
    async def get(self, key: str) -> bytes | None: ...

    @abstractmethod
    async def set(
        self, key: str, value: bytes, ttl: int | None = None
    ) -> None: ...

    @abstractmethod
    async def delete(self, key: str) -> None: ...

    @abstractmethod
    async def exists(self, key: str) -> bool: ...
