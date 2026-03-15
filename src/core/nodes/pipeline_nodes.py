"""
LangGraph Pipeline Nodes
Each node is a pure async function: PipelineState → PipelineState patch.
Nodes are stateless — all dependencies injected via DI container.
"""
from __future__ import annotations

import json
import traceback
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from config.settings import get_settings
from src.core.state.pipeline_state import PipelineState
from src.domain.entities.models import (
    ChunkingStrategy,
    DocumentStatus,
    FileType,
    ProcessingError,
    RAGCollection,
)

log = structlog.get_logger(__name__)
settings = get_settings()


# ===========================================================================
# NODE: Extract
# ===========================================================================

async def node_extract(state: PipelineState, parser_registry: Any) -> dict:
    """
    Dispatch to the correct parser based on detected file type.
    Reads raw bytes from MinIO via the raw_file_key.
    """
    log.info("node.extract.start", document_id=state.get("document_id"))
    try:
        file_type = FileType(state.get("detected_file_type", FileType.UNKNOWN))
        parser = parser_registry.get(file_type)
        if parser is None:
            raise ValueError(f"No parser registered for file type: {file_type}")

        result = await parser.parse(
            file_key=state["raw_file_key"],
            metadata=state.get("user_metadata", {}),
        )

        log.info(
            "node.extract.done",
            document_id=state.get("document_id"),
            chars=len(result.text),
        )
        return {
            "extracted_text": result.text,
            "extraction_metadata": result.metadata,
            "current_status": DocumentStatus.CLASSIFYING,
        }
    except Exception as exc:
        log.error("node.extract.failed", exc=str(exc))
        return _error_patch(state, "extract", exc)


# ===========================================================================
# NODE: Classify  (LLM-driven)
# ===========================================================================

_CLASSIFY_SYSTEM = """You are a document classification expert.
Given the first 2000 characters of a document, classify it into exactly one collection.
Collections: {collections}

Respond ONLY with valid JSON:
{{
  "collection": "<collection_name>",
  "confidence": <0.0-1.0>,
  "reasoning": "<one sentence>"
}}"""

async def node_classify(state: PipelineState) -> dict:
    """Use Ollama LLM to classify document into a RAG collection."""
    log.info("node.classify.start", document_id=state.get("document_id"))
    try:
        text_preview = (state.get("extracted_text") or "")[:2000]
        collections = [c.value for c in RAGCollection]

        llm = ChatOllama(
            model=settings.ollama.classifier_model,
            base_url=settings.ollama.base_url,
            temperature=0.0,
            format="json",
        )

        messages = [
            SystemMessage(content=_CLASSIFY_SYSTEM.format(collections=collections)),
            HumanMessage(content=f"Document preview:\n\n{text_preview}"),
        ]

        response = await llm.ainvoke(messages)
        result = json.loads(response.content)

        collection = RAGCollection(result.get("collection", RAGCollection.GENERAL))
        log.info(
            "node.classify.done",
            collection=collection,
            confidence=result.get("confidence"),
        )
        return {
            "rag_collection": collection.value,
            "classification_confidence": float(result.get("confidence", 0.0)),
            "classification_reasoning": result.get("reasoning", ""),
            "current_status": DocumentStatus.CHUNKING,
        }
    except Exception as exc:
        log.error("node.classify.failed", exc=str(exc))
        # Fallback to GENERAL — classification failure is non-fatal
        return {
            "rag_collection": RAGCollection.GENERAL.value,
            "classification_confidence": 0.0,
            "classification_reasoning": f"Fallback due to error: {exc}",
            "current_status": DocumentStatus.CHUNKING,
        }


# ===========================================================================
# NODE: Decide Chunking Strategy  (LLM-driven)
# ===========================================================================

_CHUNKING_SYSTEM = """You are a text chunking strategy expert for RAG systems.
Given document metadata and a text preview, decide the optimal chunking strategy.

Available strategies:
- semantic: Best for narrative prose, articles, reports. Splits on semantic boundaries.
- recursive: Best for mixed content. Hierarchical splitting with fallbacks.
- fixed: Best for uniform data, logs, tabular content.
- sentence: Best for FAQ, Q&A, structured dialogues.
- code_aware: Best for source code files. Respects function/class boundaries.
- table_aware: Best for spreadsheets and heavily tabular documents.

Respond ONLY with valid JSON:
{{
  "strategy": "<strategy_name>",
  "reasoning": "<one sentence explaining the choice>"
}}"""

async def node_decide_chunking(state: PipelineState) -> dict:
    """LLM decides the optimal chunking strategy for this document."""
    log.info("node.chunking_decision.start", document_id=state.get("document_id"))

    if not settings.chunking.llm_driven:
        return {
            "chunking_strategy": settings.chunking.fallback_strategy,
            "chunking_reasoning": "LLM-driven chunking disabled; using configured fallback.",
        }

    try:
        text_preview = (state.get("extracted_text") or "")[:1000]
        context = {
            "file_type": state.get("detected_file_type"),
            "rag_collection": state.get("rag_collection"),
            "extraction_metadata": state.get("extraction_metadata", {}),
        }

        llm = ChatOllama(
            model=settings.ollama.chunker_model,
            base_url=settings.ollama.base_url,
            temperature=0.0,
            format="json",
        )

        messages = [
            SystemMessage(content=_CHUNKING_SYSTEM),
            HumanMessage(content=(
                f"Document context: {json.dumps(context)}\n\n"
                f"Text preview:\n{text_preview}"
            )),
        ]

        response = await llm.ainvoke(messages)
        result = json.loads(response.content)

        strategy = ChunkingStrategy(
            result.get("strategy", settings.chunking.fallback_strategy)
        )
        log.info("node.chunking_decision.done", strategy=strategy.value)
        return {
            "chunking_strategy": strategy.value,
            "chunking_reasoning": result.get("reasoning", ""),
        }
    except Exception as exc:
        log.warning("node.chunking_decision.fallback", exc=str(exc))
        return {
            "chunking_strategy": settings.chunking.fallback_strategy,
            "chunking_reasoning": f"Fallback due to error: {exc}",
        }


# ===========================================================================
# NODE: Chunk
# ===========================================================================

async def node_chunk(state: PipelineState, chunker_registry: Any) -> dict:
    """Apply the chosen chunking strategy to produce raw text chunks."""
    log.info("node.chunk.start", document_id=state.get("document_id"))
    try:
        strategy = ChunkingStrategy(
            state.get("chunking_strategy", settings.chunking.fallback_strategy)
        )
        chunker = chunker_registry.get(strategy)
        chunks = await chunker.chunk(
            text=state["extracted_text"],
            metadata=state.get("extraction_metadata", {}),
        )
        log.info("node.chunk.done", n_chunks=len(chunks))
        return {
            "raw_chunks": chunks,
            "current_status": DocumentStatus.PROCESSING,
        }
    except Exception as exc:
        log.error("node.chunk.failed", exc=str(exc))
        return _error_patch(state, "chunk", exc)


# ===========================================================================
# NODE: Detect Language
# ===========================================================================

async def node_detect_language(
    state: PipelineState, language_detector: Any
) -> dict:
    """Detect document language from extracted text."""
    if not settings.processing.enable_language_detection:
        return {"detected_language": "unknown"}
    try:
        lang = await language_detector.detect(
            text=(state.get("extracted_text") or "")[:512]
        )
        return {"detected_language": lang}
    except Exception as exc:
        log.warning("node.language.failed", exc=str(exc))
        return {"detected_language": "unknown"}


# ===========================================================================
# NODE: NER
# ===========================================================================

async def node_ner(state: PipelineState, ner_extractor: Any) -> dict:
    """Extract named entities from each chunk."""
    if not settings.processing.enable_ner:
        return {"chunks_with_ner": [
            {"text": c, "entities": []} for c in state.get("raw_chunks", [])
        ]}
    try:
        chunks = state.get("raw_chunks", [])
        enriched = await ner_extractor.extract_batch(chunks)
        return {"chunks_with_ner": enriched}
    except Exception as exc:
        log.warning("node.ner.failed", exc=str(exc))
        # Non-fatal: continue without NER
        return {"chunks_with_ner": [
            {"text": c, "entities": []} for c in state.get("raw_chunks", [])
        ]}


# ===========================================================================
# NODE: Summarize
# ===========================================================================

async def node_summarize(state: PipelineState, summarizer: Any) -> dict:
    """Generate a short summary for each chunk."""
    if not settings.processing.enable_summarization:
        source = state.get("chunks_with_ner", [])
        return {"chunks_with_summaries": [
            {**c, "summary": None} for c in source
        ]}
    try:
        chunks = state.get("chunks_with_ner", [])
        enriched = await summarizer.summarize_batch(chunks)
        return {"chunks_with_summaries": enriched}
    except Exception as exc:
        log.warning("node.summarize.failed", exc=str(exc))
        return {"chunks_with_summaries": state.get("chunks_with_ner", [])}


# ===========================================================================
# NODE: Deduplicate
# ===========================================================================

async def node_deduplicate(
    state: PipelineState, deduplicator: Any
) -> dict:
    """Remove near-duplicate chunks using cosine similarity."""
    if not settings.processing.enable_deduplication:
        return {"chunks_deduplicated": state.get("chunks_with_summaries", [])}
    try:
        chunks = state.get("chunks_with_summaries", [])
        deduped = await deduplicator.deduplicate(
            chunks, threshold=settings.processing.dedup_threshold
        )
        removed = len(chunks) - len(deduped)
        log.info("node.dedup.done", removed=removed)
        return {"chunks_deduplicated": deduped}
    except Exception as exc:
        log.warning("node.dedup.failed", exc=str(exc))
        return {"chunks_deduplicated": state.get("chunks_with_summaries", [])}


# ===========================================================================
# NODE: PII Redaction
# ===========================================================================

async def node_pii_redact(state: PipelineState, pii_redactor: Any) -> dict:
    """Detect and redact PII from chunk text before storage."""
    if not settings.processing.enable_pii_redaction:
        chunks = state.get("chunks_deduplicated", [])
        return {"chunks_pii_redacted": [
            {**c, "text_redacted": c["text"], "has_pii": False, "pii_types": []}
            for c in chunks
        ]}
    try:
        chunks = state.get("chunks_deduplicated", [])
        redacted = await pii_redactor.redact_batch(
            chunks, entities=settings.processing.pii_entities
        )
        return {"chunks_pii_redacted": redacted}
    except Exception as exc:
        log.warning("node.pii.failed", exc=str(exc))
        return {"chunks_pii_redacted": state.get("chunks_deduplicated", [])}


# ===========================================================================
# NODE: Embed
# ===========================================================================

async def node_embed(state: PipelineState, embedder: Any) -> dict:
    """Generate dense + sparse (BM25) vectors for each chunk."""
    log.info("node.embed.start", document_id=state.get("document_id"))
    try:
        chunks = state.get("chunks_pii_redacted", [])
        embedded_chunks = await embedder.embed_batch(
            chunks=chunks,
            document_id=state["document_id"],
            rag_collection=RAGCollection(state["rag_collection"]),
            language=state.get("detected_language", "unknown"),
        )
        log.info("node.embed.done", n_chunks=len(embedded_chunks))
        return {
            "chunks_embedded": embedded_chunks,
            "current_status": DocumentStatus.INDEXING,
        }
    except Exception as exc:
        log.error("node.embed.failed", exc=str(exc))
        return _error_patch(state, "embed", exc)


# ===========================================================================
# NODE: Index
# ===========================================================================

async def node_index(state: PipelineState, vector_repo: Any) -> dict:
    """Upsert embedded chunks into the correct Qdrant collection."""
    log.info("node.index.start", document_id=state.get("document_id"))
    try:
        collection = RAGCollection(state["rag_collection"])
        chunks = state.get("chunks_embedded", [])

        await vector_repo.upsert_chunks(collection=collection, chunks=chunks)

        point_ids = [str(c.id) for c in chunks]
        log.info("node.index.done", n_indexed=len(point_ids))
        return {
            "indexed_chunk_ids": point_ids,
            "qdrant_collection": collection.value,
            "current_status": DocumentStatus.COMPLETED,
        }
    except Exception as exc:
        log.error("node.index.failed", exc=str(exc))
        return _error_patch(state, "index", exc)


# ===========================================================================
# NODE: Handle Error / DLQ
# ===========================================================================

async def node_handle_error(state: PipelineState) -> dict:
    """
    Evaluate retry count. Either increment retry or send to DLQ.
    """
    retry_count = state.get("retry_count", 0)
    max_retries = settings.worker.max_retries

    if retry_count < max_retries:
        log.warning(
            "node.error.retry",
            document_id=state.get("document_id"),
            attempt=retry_count + 1,
        )
        return {
            "retry_count": retry_count + 1,
            "should_retry": True,
            "send_to_dlq": False,
            "current_status": DocumentStatus.FAILED,
        }
    else:
        errors = state.get("errors", [])
        dlq_reason = "; ".join(e.message for e in errors[-3:]) if errors else "max retries exceeded"
        log.error(
            "node.error.dlq",
            document_id=state.get("document_id"),
            reason=dlq_reason,
        )
        return {
            "should_retry": False,
            "send_to_dlq": True,
            "dlq_reason": dlq_reason,
            "current_status": DocumentStatus.DLQ,
        }


# ===========================================================================
# Helpers
# ===========================================================================

def _error_patch(
    state: PipelineState, stage: str, exc: Exception
) -> dict:
    error = ProcessingError(
        stage=stage,
        message=str(exc),
        traceback=traceback.format_exc(),
        retryable=True,
    )
    existing_errors = list(state.get("errors") or [])
    existing_errors.append(error)
    return {
        "current_status": DocumentStatus.FAILED,
        "errors": existing_errors,
    }
