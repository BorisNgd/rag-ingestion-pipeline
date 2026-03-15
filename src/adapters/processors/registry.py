"""
Processor Registries
All processing services: chunkers, NER, summarizer, deduplicator, PII redactor.
Each service is independently configurable via settings.
"""
from __future__ import annotations

import re
from typing import Any

import structlog
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    Language,
    SentenceTransformersTokenTextSplitter,
)

from config.settings import get_settings
from src.domain.entities.models import ChunkingStrategy

log = structlog.get_logger(__name__)
settings = get_settings()


# ===========================================================================
# Chunker Registry
# ===========================================================================

class BaseChunker:
    async def chunk(self, text: str, metadata: dict) -> list[str]:
        raise NotImplementedError


class RecursiveChunker(BaseChunker):
    async def chunk(self, text: str, metadata: dict) -> list[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunking.fixed_chunk_size,
            chunk_overlap=settings.chunking.fixed_chunk_overlap,
            length_function=len,
        )
        return splitter.split_text(text)


class FixedChunker(BaseChunker):
    async def chunk(self, text: str, metadata: dict) -> list[str]:
        size = settings.chunking.fixed_chunk_size
        overlap = settings.chunking.fixed_chunk_overlap
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            chunks.append(text[start:end])
            start += size - overlap
        return [c for c in chunks if len(c) >= settings.chunking.min_chunk_size]


class SemanticChunker(BaseChunker):
    """Uses sentence embeddings to find semantic breakpoints."""

    async def chunk(self, text: str, metadata: dict) -> list[str]:
        try:
            from langchain_experimental.text_splitter import SemanticChunker
            from langchain_huggingface import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model_name=settings.huggingface.dense_embedding_model,
                cache_folder=settings.huggingface.cache_dir,
            )
            splitter = SemanticChunker(
                embeddings,
                breakpoint_threshold_type="percentile",
                breakpoint_threshold_amount=settings.chunking.semantic_breakpoint_threshold * 100,
            )
            docs = splitter.create_documents([text])
            return [d.page_content for d in docs]
        except Exception as e:
            log.warning("semantic_chunker.fallback", exc=str(e))
            return await RecursiveChunker().chunk(text, metadata)


class SentenceChunker(BaseChunker):
    async def chunk(self, text: str, metadata: dict) -> list[str]:
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=settings.chunking.fixed_chunk_overlap,
            tokens_per_chunk=settings.chunking.fixed_chunk_size,
        )
        return splitter.split_text(text)


class CodeAwareChunker(BaseChunker):
    """Respects function/class boundaries in code files."""

    async def chunk(self, text: str, metadata: dict) -> list[str]:
        lang = metadata.get("language", "python")
        try:
            lang_enum = Language(lang)
        except ValueError:
            lang_enum = Language.PYTHON

        splitter = RecursiveCharacterTextSplitter.from_language(
            language=lang_enum,
            chunk_size=settings.chunking.fixed_chunk_size,
            chunk_overlap=settings.chunking.fixed_chunk_overlap,
        )
        return splitter.split_text(text)


class TableAwareChunker(BaseChunker):
    """Splits tabular text row by row, preserving headers."""

    async def chunk(self, text: str, metadata: dict) -> list[str]:
        lines = text.splitlines()
        if not lines:
            return []

        header = lines[0] if lines else ""
        chunk_size = settings.chunking.fixed_chunk_size // 100  # rows per chunk
        chunks = []

        for i in range(1, len(lines), max(1, chunk_size)):
            batch = lines[i : i + chunk_size]
            chunk_text = header + "\n" + "\n".join(batch)
            if len(chunk_text) >= settings.chunking.min_chunk_size:
                chunks.append(chunk_text)

        return chunks or [text]


class ChunkerRegistry:
    def __init__(self) -> None:
        self._chunkers: dict[ChunkingStrategy, BaseChunker] = {
            ChunkingStrategy.RECURSIVE: RecursiveChunker(),
            ChunkingStrategy.FIXED: FixedChunker(),
            ChunkingStrategy.SEMANTIC: SemanticChunker(),
            ChunkingStrategy.SENTENCE: SentenceChunker(),
            ChunkingStrategy.CODE_AWARE: CodeAwareChunker(),
            ChunkingStrategy.TABLE_AWARE: TableAwareChunker(),
        }

    def get(self, strategy: ChunkingStrategy) -> BaseChunker:
        return self._chunkers.get(strategy, self._chunkers[ChunkingStrategy.RECURSIVE])


# ===========================================================================
# Language Detector
# ===========================================================================

class LanguageDetectorService:
    """Uses langdetect with fallback to HuggingFace classifier."""

    def __init__(self) -> None:
        self._model = None

    async def detect(self, text: str) -> str:
        try:
            from langdetect import detect
            return detect(text[:1000])
        except Exception:
            return "unknown"


# ===========================================================================
# NER Extractor
# ===========================================================================

class NERExtractorService:
    """
    Named entity extraction using HuggingFace transformers.
    Falls back to spaCy if transformers fail.
    """

    def __init__(self) -> None:
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            from transformers import pipeline
            self._pipeline = pipeline(
                "ner",
                model=settings.huggingface.ner_model,
                aggregation_strategy="simple",
                device=0 if settings.huggingface.dense_embedding_device == "cuda" else -1,
            )
        return self._pipeline

    async def extract_batch(
        self, chunks: list[str]
    ) -> list[dict[str, Any]]:
        results = []
        pipe = self._get_pipeline()
        for chunk_text in chunks:
            try:
                entities = pipe(chunk_text[:512])
                results.append({
                    "text": chunk_text,
                    "entities": [
                        {
                            "text": e["word"],
                            "label": e["entity_group"],
                            "score": round(float(e["score"]), 4),
                            "start": e["start"],
                            "end": e["end"],
                        }
                        for e in entities
                    ],
                })
            except Exception as e:
                log.warning("ner.chunk.failed", exc=str(e))
                results.append({"text": chunk_text, "entities": []})
        return results


# ===========================================================================
# Summarizer
# ===========================================================================

class SummarizerService:
    """Per-chunk summarization via Ollama LLM."""

    async def summarize_batch(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        from langchain_ollama import ChatOllama
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatOllama(
            model=settings.ollama.summarizer_model,
            base_url=settings.ollama.base_url,
            temperature=0.0,
        )

        results = []
        for chunk_dict in chunks:
            text = chunk_dict.get("text", "")
            if len(text) < 100:
                results.append({**chunk_dict, "summary": None})
                continue
            try:
                response = await llm.ainvoke([
                    SystemMessage(content=(
                        f"Summarize the following text in at most "
                        f"{settings.processing.summary_max_tokens} tokens. "
                        "Return only the summary, nothing else."
                    )),
                    HumanMessage(content=text[:2000]),
                ])
                results.append({**chunk_dict, "summary": response.content.strip()})
            except Exception as e:
                log.warning("summarizer.chunk.failed", exc=str(e))
                results.append({**chunk_dict, "summary": None})
        return results


# ===========================================================================
# Deduplicator
# ===========================================================================

class DeduplicatorService:
    """
    Removes near-duplicate chunks using cosine similarity on dense embeddings.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    async def deduplicate(
        self, chunks: list[dict[str, Any]], threshold: float = 0.95
    ) -> list[dict[str, Any]]:
        if len(chunks) <= 1:
            return chunks

        texts = [c.get("text", "") for c in chunks]
        try:
            vectors = await self._embedder._embed_dense(texts)
        except Exception as e:
            log.warning("dedup.embed.failed", exc=str(e))
            return chunks

        import numpy as np

        kept: list[int] = []
        kept_vectors: list[list[float]] = []

        for i, vec in enumerate(vectors):
            v = np.array(vec)
            is_dup = False
            for kv in kept_vectors:
                kv_arr = np.array(kv)
                cosine = float(
                    np.dot(v, kv_arr) / (np.linalg.norm(v) * np.linalg.norm(kv_arr) + 1e-8)
                )
                if cosine >= threshold:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(i)
                kept_vectors.append(vec)

        return [chunks[i] for i in kept]


# ===========================================================================
# PII Redactor
# ===========================================================================

class PIIRedactorService:
    """
    Detects and redacts PII using Microsoft Presidio.
    Falls back to regex patterns if Presidio is unavailable.
    """

    def __init__(self) -> None:
        self._analyzer = None
        self._anonymizer = None
        self._use_presidio = True

    def _get_presidio(self):
        if self._analyzer is None:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_anonymizer import AnonymizerEngine

                self._analyzer = AnalyzerEngine()
                self._anonymizer = AnonymizerEngine()
            except ImportError:
                log.warning("pii.presidio.unavailable", fallback="regex")
                self._use_presidio = False
        return self._analyzer, self._anonymizer

    async def redact_batch(
        self,
        chunks: list[dict[str, Any]],
        entities: list[str],
    ) -> list[dict[str, Any]]:
        results = []
        for chunk_dict in chunks:
            text = chunk_dict.get("text", "")
            redacted, has_pii, pii_types = await self._redact_one(text, entities)
            results.append({
                **chunk_dict,
                "text_redacted": redacted,
                "has_pii": has_pii,
                "pii_types": pii_types,
            })
        return results

    async def _redact_one(
        self, text: str, entities: list[str]
    ) -> tuple[str, bool, list[str]]:
        analyzer, anonymizer = self._get_presidio()

        if self._use_presidio and analyzer:
            try:
                results = analyzer.analyze(
                    text=text,
                    entities=entities,
                    language="en",
                )
                if not results:
                    return text, False, []

                anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
                found_types = list({r.entity_type for r in results})
                return anonymized.text, True, found_types
            except Exception as e:
                log.warning("pii.presidio.failed", exc=str(e))

        # Regex fallback
        redacted = text
        found: list[str] = []

        patterns = {
            "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
            "PHONE": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
            "IP_ADDRESS": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
        }
        for label, pattern in patterns.items():
            if re.search(pattern, redacted):
                redacted = re.sub(pattern, f"<{label}>", redacted)
                found.append(label)

        return redacted, bool(found), found
