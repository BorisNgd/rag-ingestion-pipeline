"""
Hybrid Embedder
Generates dense vectors (BGE-M3 via HuggingFace) and sparse vectors (BM25)
for every chunk. Results are cached in Redis to avoid redundant computation.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from typing import Any

import structlog
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi

from config.settings import get_settings
from src.domain.entities.models import Chunk, ChunkId, ChunkMetadata, RAGCollection
from src.domain.repositories.interfaces import CacheRepository

log = structlog.get_logger(__name__)
settings = get_settings()


class HybridEmbedder:
    """
    Produces dense + sparse embeddings for a batch of processed chunks.
    Handles BGE-M3 (dense + colbert) and BM25 (sparse) in one pass.
    """

    def __init__(self, cache: CacheRepository) -> None:
        self._cache = cache
        self._model: BGEM3FlagModel | None = None

    def _get_model(self) -> BGEM3FlagModel:
        if self._model is None:
            log.info("embedder.model.loading", model=settings.huggingface.dense_embedding_model)
            self._model = BGEM3FlagModel(
                model_name_or_path=settings.huggingface.dense_embedding_model,
                use_fp16=True,  # Half precision for speed
                device=settings.huggingface.dense_embedding_device,
                cache_dir=settings.huggingface.cache_dir,
            )
            log.info("embedder.model.loaded")
        return self._model

    async def embed_batch(
        self,
        chunks: list[dict[str, Any]],
        document_id: str,
        rag_collection: RAGCollection,
        language: str = "unknown",
    ) -> list[Chunk]:
        """
        Embed a list of processed chunk dicts into Chunk domain objects
        with dense and sparse vectors populated.
        """
        if not chunks:
            return []

        texts = [c.get("text_redacted") or c.get("text", "") for c in chunks]
        total = len(texts)

        # --- Dense embeddings via BGE-M3 ---
        dense_vectors = await self._embed_dense(texts)

        # --- Sparse embeddings via BM25 ---
        sparse_vectors = self._embed_sparse(texts)

        # --- Assemble Chunk objects ---
        result: list[Chunk] = []
        for idx, (chunk_dict, dense, sparse) in enumerate(
            zip(chunks, dense_vectors, sparse_vectors)
        ):
            chunk = Chunk(
                document_id=document_id,
                rag_collection=rag_collection,
                text=chunk_dict.get("text", ""),
                text_redacted=chunk_dict.get("text_redacted", chunk_dict.get("text", "")),
                metadata=ChunkMetadata(
                    document_id=document_id,
                    chunk_index=idx,
                    total_chunks=total,
                    start_char=chunk_dict.get("start_char", 0),
                    end_char=chunk_dict.get("end_char", 0),
                    page_number=chunk_dict.get("page_number"),
                    section_title=chunk_dict.get("section_title"),
                    language=language,
                    entities=chunk_dict.get("entities", []),
                    summary=chunk_dict.get("summary"),
                    has_pii=chunk_dict.get("has_pii", False),
                    pii_types=chunk_dict.get("pii_types", []),
                    token_count=len(texts[idx].split()),
                ),
                dense_vector=dense,
                sparse_vector=sparse,
            )
            result.append(chunk)

        log.info(
            "embedder.batch.done",
            document_id=document_id,
            n_chunks=len(result),
            collection=rag_collection.value,
        )
        return result

    async def embed_query(
        self, query: str
    ) -> tuple[list[float], dict[int, float]]:
        """
        Embed a search query for hybrid retrieval.
        Returns (dense_vector, sparse_vector).
        """
        cache_key = f"qemb:{hashlib.sha256(query.encode()).hexdigest()}"
        cached = await self._cache.get(cache_key)
        if cached:
            return pickle.loads(cached)

        dense = (await self._embed_dense([query]))[0]
        sparse = self._embed_sparse([query])[0]

        await self._cache.set(cache_key, pickle.dumps((dense, sparse)), ttl=3600)
        return dense, sparse

    async def _embed_dense(self, texts: list[str]) -> list[list[float]]:
        """Batch dense embeddings with Redis caching per text."""
        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # Check cache
        for i, text in enumerate(texts):
            key = _text_cache_key(text)
            cached = await self._cache.get(key)
            if cached:
                results[i] = pickle.loads(cached)
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Compute missing
        if uncached_texts:
            model = self._get_model()
            batch_size = settings.huggingface.dense_batch_size

            all_embeddings: list[list[float]] = []
            for start in range(0, len(uncached_texts), batch_size):
                batch = uncached_texts[start : start + batch_size]
                output = model.encode(
                    batch,
                    batch_size=len(batch),
                    max_length=8192,
                    return_dense=True,
                    return_sparse=False,
                    return_colbert_vecs=False,
                )
                all_embeddings.extend(output["dense_vecs"].tolist())

            # Fill results and cache
            for idx, embedding in zip(uncached_indices, all_embeddings):
                results[idx] = embedding
                key = _text_cache_key(texts[idx])
                await self._cache.set(
                    key, pickle.dumps(embedding), ttl=settings.redis.ttl_seconds
                )

        return [r for r in results if r is not None]

    def _embed_sparse(self, texts: list[str]) -> list[dict[int, float]]:
        """
        BM25 sparse embeddings.
        Returns per-text dict of {token_id: weight}.
        """
        tokenized = [text.lower().split() for text in texts]
        bm25 = BM25Okapi(tokenized)

        # Build a shared vocabulary across this batch
        vocab: dict[str, int] = {}
        for tokens in tokenized:
            for tok in tokens:
                if tok not in vocab:
                    vocab[tok] = len(vocab)

        # For each token, bm25.get_scores([tok]) returns an array of size n_docs.
        # We accumulate per-document weights keyed by vocabulary index.
        sparse_vectors: list[dict[int, float]] = [{} for _ in tokenized]
        for tok, tok_id in vocab.items():
            doc_scores = bm25.get_scores([tok])  # shape: (n_docs,)
            for doc_idx, score in enumerate(doc_scores):
                if score > 0:
                    sparse_vectors[doc_idx][tok_id] = float(score)

        return sparse_vectors


def _text_cache_key(text: str) -> str:
    digest = hashlib.sha256(text.encode()).hexdigest()
    return f"emb:{digest}"
