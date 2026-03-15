"""
Unit tests — no infrastructure required, fully isolated.
Run with: pytest tests/unit/ -v
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.domain.entities.models import (
    Chunk,
    ChunkId,
    Document,
    DocumentId,
    DocumentMetadata,
    DocumentStatus,
    FileType,
    ProcessingError,
    RAGCollection,
)
from src.core.state.pipeline_state import PipelineState


# ===========================================================================
# Domain entity tests
# ===========================================================================

class TestDocument:

    def test_initial_status_is_pending(self):
        doc = Document()
        assert doc.status == DocumentStatus.PENDING

    def test_transition_updates_status(self):
        doc = Document()
        doc.transition_to(DocumentStatus.EXTRACTING)
        assert doc.status == DocumentStatus.EXTRACTING

    def test_completed_sets_completed_at(self):
        doc = Document()
        doc.transition_to(DocumentStatus.COMPLETED)
        assert doc.completed_at is not None

    def test_is_terminal_for_completed(self):
        doc = Document()
        doc.transition_to(DocumentStatus.COMPLETED)
        assert doc.is_terminal is True

    def test_is_terminal_for_dlq(self):
        doc = Document()
        doc.send_to_dlq("test reason")
        assert doc.is_terminal is True
        assert doc.status == DocumentStatus.DLQ

    def test_can_retry_when_failed(self):
        doc = Document()
        doc.transition_to(DocumentStatus.FAILED)
        assert doc.can_retry is True

    def test_cannot_retry_when_completed(self):
        doc = Document()
        doc.transition_to(DocumentStatus.COMPLETED)
        assert doc.can_retry is False

    def test_add_error_appends(self):
        doc = Document()
        err = ProcessingError(stage="extract", message="test error")
        doc.add_error(err)
        assert len(doc.errors) == 1
        assert doc.errors[0].stage == "extract"

    def test_increment_retry(self):
        doc = Document()
        doc.increment_retry()
        doc.increment_retry()
        assert doc.retry_count == 2


class TestChunk:

    def test_embeddable_text_returns_redacted_when_pii(self):
        from src.domain.entities.models import ChunkMetadata
        chunk = Chunk(
            text="My name is John Doe",
            text_redacted="My name is <PERSON>",
            metadata=ChunkMetadata(
                document_id="doc-1",
                chunk_index=0,
                total_chunks=1,
                start_char=0,
                end_char=19,
                has_pii=True,
            ),
        )
        assert chunk.embeddable_text == "My name is <PERSON>"

    def test_embeddable_text_returns_original_without_pii(self):
        from src.domain.entities.models import ChunkMetadata
        chunk = Chunk(
            text="Hello world",
            text_redacted="Hello world",
            metadata=ChunkMetadata(
                document_id="doc-1",
                chunk_index=0,
                total_chunks=1,
                start_char=0,
                end_char=11,
                has_pii=False,
            ),
        )
        assert chunk.embeddable_text == "Hello world"


# ===========================================================================
# Graph node tests (mocked dependencies)
# ===========================================================================

class TestNodeClassify:

    @pytest.mark.asyncio
    async def test_classify_returns_general_on_llm_failure(self, monkeypatch):
        from src.core.nodes.pipeline_nodes import node_classify

        # Mock ChatOllama to raise
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM unavailable"))

        monkeypatch.setattr(
            "src.core.nodes.pipeline_nodes.ChatOllama",
            lambda **kwargs: mock_llm,
        )

        state: PipelineState = {
            "document_id": "doc-1",
            "extracted_text": "Some document text",
        }

        result = await node_classify(state)
        assert result["rag_collection"] == "general"
        assert result["classification_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_classify_success(self, monkeypatch):
        import json
        from src.core.nodes.pipeline_nodes import node_classify

        mock_response = MagicMock()
        mock_response.content = json.dumps({
            "collection": "technical",
            "confidence": 0.92,
            "reasoning": "Document discusses software architecture",
        })

        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(return_value=mock_response)

        monkeypatch.setattr(
            "src.core.nodes.pipeline_nodes.ChatOllama",
            lambda **kwargs: mock_llm,
        )

        state: PipelineState = {
            "document_id": "doc-1",
            "extracted_text": "This document describes a microservices architecture.",
        }

        result = await node_classify(state)
        assert result["rag_collection"] == "technical"
        assert result["classification_confidence"] == 0.92


class TestNodeHandleError:

    @pytest.mark.asyncio
    async def test_increments_retry_below_max(self, monkeypatch):
        from src.core.nodes.pipeline_nodes import node_handle_error

        monkeypatch.setattr(
            "src.core.nodes.pipeline_nodes.settings.worker.max_retries", 3
        )

        state: PipelineState = {
            "document_id": "doc-1",
            "retry_count": 1,
            "errors": [ProcessingError(stage="extract", message="fail")],
        }

        result = await node_handle_error(state)
        assert result["retry_count"] == 2
        assert result["should_retry"] is True
        assert result["send_to_dlq"] is False

    @pytest.mark.asyncio
    async def test_sends_to_dlq_at_max_retries(self, monkeypatch):
        from src.core.nodes.pipeline_nodes import node_handle_error

        monkeypatch.setattr(
            "src.core.nodes.pipeline_nodes.settings.worker.max_retries", 3
        )

        state: PipelineState = {
            "document_id": "doc-1",
            "retry_count": 3,
            "errors": [ProcessingError(stage="embed", message="OOM")],
        }

        result = await node_handle_error(state)
        assert result["send_to_dlq"] is True
        assert result["should_retry"] is False
        assert result["current_status"] == DocumentStatus.DLQ.value


# ===========================================================================
# Processor tests
# ===========================================================================

class TestDeduplicator:

    @pytest.mark.asyncio
    async def test_removes_exact_duplicates(self):
        from src.adapters.processors.registry import DeduplicatorService

        mock_embedder = MagicMock()
        # Return identical vectors for first two, different for third
        mock_embedder._embed_dense = AsyncMock(return_value=[
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],  # duplicate
            [0.0, 1.0, 0.0],  # unique
        ])

        dedup = DeduplicatorService(embedder=mock_embedder)
        chunks = [
            {"text": "Hello world"},
            {"text": "Hello world"},  # duplicate
            {"text": "Different content"},
        ]

        result = await dedup.deduplicate(chunks, threshold=0.95)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_keeps_all_unique(self):
        from src.adapters.processors.registry import DeduplicatorService

        mock_embedder = MagicMock()
        mock_embedder._embed_dense = AsyncMock(return_value=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])

        dedup = DeduplicatorService(embedder=mock_embedder)
        chunks = [
            {"text": "Doc A"},
            {"text": "Doc B"},
            {"text": "Doc C"},
        ]

        result = await dedup.deduplicate(chunks, threshold=0.95)
        assert len(result) == 3


class TestPIIRedactor:

    @pytest.mark.asyncio
    async def test_redacts_email_via_regex_fallback(self):
        from src.adapters.processors.registry import PIIRedactorService

        redactor = PIIRedactorService()
        redactor._use_presidio = False  # Force regex fallback

        chunks = [{"text": "Contact me at john@example.com"}]
        result = await redactor.redact_batch(chunks, entities=["EMAIL"])

        assert result[0]["has_pii"] is True
        assert "john@example.com" not in result[0]["text_redacted"]
        assert "<EMAIL>" in result[0]["text_redacted"]

    @pytest.mark.asyncio
    async def test_no_pii_unchanged(self):
        from src.adapters.processors.registry import PIIRedactorService

        redactor = PIIRedactorService()
        redactor._use_presidio = False

        chunks = [{"text": "This is a clean document about software."}]
        result = await redactor.redact_batch(chunks, entities=["EMAIL", "PHONE"])

        assert result[0]["has_pii"] is False
        assert result[0]["text_redacted"] == chunks[0]["text"]
