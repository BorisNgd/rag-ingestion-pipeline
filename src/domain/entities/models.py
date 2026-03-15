"""
Domain Entities — pure Python, zero framework dependencies.
These are the heart of the system: stable, testable, framework-agnostic.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class DocumentStatus(str, Enum):
    PENDING = "pending"
    EXTRACTING = "extracting"
    CLASSIFYING = "classifying"
    CHUNKING = "chunking"
    PROCESSING = "processing"
    EMBEDDING = "embedding"
    INDEXING = "indexing"
    COMPLETED = "completed"
    FAILED = "failed"
    DLQ = "dlq"  # Dead-letter queue


class FileType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    EXCEL = "excel"
    HTML = "html"
    IMAGE = "image"
    AUDIO = "audio"
    CODE = "code"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"


class ChunkingStrategy(str, Enum):
    SEMANTIC = "semantic"
    FIXED = "fixed"
    SENTENCE = "sentence"
    RECURSIVE = "recursive"
    CODE_AWARE = "code_aware"
    TABLE_AWARE = "table_aware"


class RAGCollection(str, Enum):
    """Target RAG collections — LLM classifies documents into these."""
    TECHNICAL = "technical"
    LEGAL = "legal"
    FINANCIAL = "financial"
    MEDICAL = "medical"
    GENERAL = "general"
    CODE = "code"
    MULTIMEDIA = "multimedia"


# ---------------------------------------------------------------------------
# Value Objects
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DocumentId:
    value: str = field(default_factory=new_id)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class ChunkId:
    value: str = field(default_factory=new_id)

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class StorageLocation:
    bucket: str
    key: str
    content_type: str
    size_bytes: int

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


@dataclass(frozen=True)
class ProcessingError:
    stage: str
    message: str
    traceback: str | None = None
    timestamp: datetime = field(default_factory=utcnow)
    retryable: bool = True


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------

@dataclass
class DocumentMetadata:
    """User-supplied and system-derived metadata."""
    source_filename: str
    file_type: FileType
    content_type: str
    size_bytes: int
    language: str | None = None
    detected_language: str | None = None
    tags: list[str] = field(default_factory=list)
    custom: dict[str, Any] = field(default_factory=dict)
    # Set by classifier
    rag_collection: RAGCollection | None = None
    classification_confidence: float | None = None
    classification_reasoning: str | None = None


@dataclass
class Document:
    """
    Root aggregate — owns the full lifecycle of an ingested document.
    All state transitions go through this entity.
    """
    id: DocumentId = field(default_factory=DocumentId)
    status: DocumentStatus = DocumentStatus.PENDING
    metadata: DocumentMetadata | None = None
    storage: StorageLocation | None = None
    raw_text: str | None = None
    chunking_strategy: ChunkingStrategy | None = None
    chunking_reasoning: str | None = None
    retry_count: int = 0
    errors: list[ProcessingError] = field(default_factory=list)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    completed_at: datetime | None = None
    pipeline_run_id: str | None = None  # LangGraph thread_id

    def transition_to(self, status: DocumentStatus) -> None:
        self.status = status
        self.updated_at = utcnow()
        if status == DocumentStatus.COMPLETED:
            self.completed_at = utcnow()

    def add_error(self, error: ProcessingError) -> None:
        self.errors.append(error)
        self.updated_at = utcnow()

    def increment_retry(self) -> None:
        self.retry_count += 1
        self.updated_at = utcnow()

    def send_to_dlq(self, reason: str) -> None:
        self.transition_to(DocumentStatus.DLQ)
        self.add_error(ProcessingError(
            stage="dlq",
            message=f"Sent to DLQ: {reason}",
            retryable=False,
        ))

    @property
    def is_terminal(self) -> bool:
        return self.status in (
            DocumentStatus.COMPLETED,
            DocumentStatus.DLQ,
        )

    @property
    def can_retry(self) -> bool:
        return (
            self.status == DocumentStatus.FAILED
            and not self.is_terminal
        )


@dataclass
class ChunkMetadata:
    """Per-chunk derived metadata."""
    document_id: str
    chunk_index: int
    total_chunks: int
    start_char: int
    end_char: int
    page_number: int | None = None
    section_title: str | None = None
    language: str | None = None
    entities: list[dict[str, Any]] = field(default_factory=list)
    summary: str | None = None
    has_pii: bool = False
    pii_types: list[str] = field(default_factory=list)
    token_count: int | None = None


@dataclass
class Chunk:
    """
    A processed text chunk ready for embedding and indexing.
    """
    id: ChunkId = field(default_factory=ChunkId)
    document_id: str = ""
    rag_collection: RAGCollection = RAGCollection.GENERAL
    text: str = ""
    text_redacted: str = ""  # PII-redacted version stored in vector DB
    metadata: ChunkMetadata | None = None
    # Set after embedding
    dense_vector: list[float] | None = None
    sparse_vector: dict[int, float] | None = None  # BM25 token_id → weight
    qdrant_point_id: str | None = None
    created_at: datetime = field(default_factory=utcnow)

    @property
    def embeddable_text(self) -> str:
        """Use redacted text for embedding if PII was detected."""
        if self.metadata and self.metadata.has_pii:
            return self.text_redacted
        return self.text


@dataclass
class IngestionJob:
    """
    Tracks a single ingestion request submitted via API.
    One job = one file upload.
    """
    id: str = field(default_factory=new_id)
    document_id: str = ""
    status: DocumentStatus = DocumentStatus.PENDING
    webhook_url: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
