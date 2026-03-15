"""
LangGraph Pipeline State
Typed state object that flows through every node in the ingestion graph.
All fields are optional — nodes only populate what they produce.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TypedDict

from src.domain.entities.models import (
    Chunk,
    ChunkingStrategy,
    Document,
    DocumentStatus,
    ProcessingError,
    RAGCollection,
)


class PipelineState(TypedDict, total=False):
    """
    The single state object threaded through the entire LangGraph.

    Convention:
    - Nodes READ from state and write their outputs back.
    - Nodes must never mutate upstream fields — only append/set their own.
    - 'errors' uses a reducer that appends (never overwrites).
    """

    # --- Identity ---
    job_id: str
    document_id: str
    pipeline_run_id: str  # LangGraph thread_id

    # --- Input ---
    raw_file_bytes: bytes | None
    raw_file_key: str          # MinIO key in raw bucket
    source_filename: str
    content_type: str
    file_size_bytes: int
    user_tags: list[str]
    user_metadata: dict[str, Any]

    # --- Extraction outputs ---
    extracted_text: str
    detected_file_type: str    # FileType enum value
    extraction_metadata: dict[str, Any]

    # --- Classification outputs (LLM-driven) ---
    rag_collection: str        # RAGCollection enum value
    classification_confidence: float
    classification_reasoning: str

    # --- Chunking outputs (LLM-driven strategy decision) ---
    chunking_strategy: str     # ChunkingStrategy enum value
    chunking_reasoning: str    # LLM explanation for the choice
    raw_chunks: list[str]      # Plain text chunks before processing

    # --- Processing outputs ---
    detected_language: str
    chunks_with_ner: list[dict[str, Any]]
    chunks_with_summaries: list[dict[str, Any]]
    chunks_deduplicated: list[dict[str, Any]]
    chunks_pii_redacted: list[dict[str, Any]]

    # --- Embedding outputs ---
    chunks_embedded: list[Chunk]

    # --- Indexing outputs ---
    indexed_chunk_ids: list[str]
    qdrant_collection: str

    # --- Control flow ---
    current_status: str        # DocumentStatus enum value
    retry_count: int
    should_retry: bool
    send_to_dlq: bool
    dlq_reason: str

    # --- Error accumulation (append-only reducer) ---
    errors: list[ProcessingError]
