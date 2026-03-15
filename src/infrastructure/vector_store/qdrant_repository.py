"""
Qdrant Vector Repository
Implements hybrid search: dense (BGE-M3) + sparse (BM25).
Collections are auto-created per RAGCollection enum value.
"""
from __future__ import annotations

from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from config.settings import get_settings
from src.domain.entities.models import Chunk, RAGCollection
from src.domain.repositories.interfaces import VectorRepository

log = structlog.get_logger(__name__)
settings = get_settings()

_DENSE_VECTOR_NAME = "dense"
_SPARSE_VECTOR_NAME = "sparse"


class QdrantVectorRepository(VectorRepository):

    def __init__(self, client: AsyncQdrantClient) -> None:
        self._client = client
        self._initialized_collections: set[str] = set()

    @classmethod
    def create(cls) -> "QdrantVectorRepository":
        client = AsyncQdrantClient(
            url=settings.qdrant.url,
            api_key=settings.qdrant.api_key or None,
            prefer_grpc=settings.qdrant.prefer_grpc,
            timeout=settings.qdrant.timeout,
        )
        return cls(client)

    async def ensure_collection(
        self,
        collection: RAGCollection,
        dense_size: int | None = None,
    ) -> None:
        name = collection.value
        if name in self._initialized_collections:
            return

        dense_size = dense_size or settings.qdrant.default_dense_size

        try:
            await self._client.get_collection(name)
            self._initialized_collections.add(name)
            log.info("qdrant.collection.exists", collection=name)
            return
        except (UnexpectedResponse, Exception):
            pass  # Collection doesn't exist, create it

        log.info("qdrant.collection.creating", collection=name)
        await self._client.create_collection(
            collection_name=name,
            vectors_config={
                _DENSE_VECTOR_NAME: qmodels.VectorParams(
                    size=dense_size,
                    distance=qmodels.Distance.COSINE,
                    on_disk=True,  # Large collections on disk
                ),
            },
            sparse_vectors_config={
                _SPARSE_VECTOR_NAME: qmodels.SparseVectorParams(
                    index=qmodels.SparseIndexParams(on_disk=False),
                ),
            },
            optimizers_config=qmodels.OptimizersConfigDiff(
                indexing_threshold=10_000,
                memmap_threshold=50_000,
            ),
            hnsw_config=qmodels.HnswConfigDiff(
                m=16,
                ef_construct=100,
                full_scan_threshold=10_000,
            ),
            quantization_config=qmodels.ScalarQuantization(
                scalar=qmodels.ScalarQuantizationConfig(
                    type=qmodels.ScalarType.INT8,
                    always_ram=True,
                ),
            ),
        )

        # Create payload indexes for efficient filtering
        for field_name in ("document_id", "language", "has_pii", "chunk_index"):
            await self._client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=qmodels.PayloadSchemaType.KEYWORD
                if field_name not in ("chunk_index",)
                else qmodels.PayloadSchemaType.INTEGER,
            )

        self._initialized_collections.add(name)
        log.info("qdrant.collection.created", collection=name)

    async def upsert_chunks(
        self,
        collection: RAGCollection,
        chunks: list[Chunk],
    ) -> None:
        await self.ensure_collection(collection)

        points: list[qmodels.PointStruct] = []
        for chunk in chunks:
            if chunk.dense_vector is None:
                log.warning("qdrant.skip_chunk.no_vector", chunk_id=str(chunk.id))
                continue

            payload = _build_payload(chunk)

            vectors: dict[str, Any] = {
                _DENSE_VECTOR_NAME: chunk.dense_vector,
            }
            if chunk.sparse_vector:
                indices = list(chunk.sparse_vector.keys())
                values = list(chunk.sparse_vector.values())
                vectors[_SPARSE_VECTOR_NAME] = qmodels.SparseVector(
                    indices=indices,
                    values=values,
                )

            points.append(
                qmodels.PointStruct(
                    id=str(chunk.id),
                    vector=vectors,
                    payload=payload,
                )
            )

        if not points:
            return

        # Batch upsert in chunks of 100
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            await self._client.upsert(
                collection_name=collection.value,
                points=batch,
                wait=True,
            )
            log.debug(
                "qdrant.upsert.batch",
                collection=collection.value,
                batch=i // batch_size + 1,
                count=len(batch),
            )

    async def delete_by_document(
        self, collection: RAGCollection, document_id: str
    ) -> None:
        await self._client.delete(
            collection_name=collection.value,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )
        log.info(
            "qdrant.delete.done",
            collection=collection.value,
            document_id=document_id,
        )

    async def hybrid_search(
        self,
        collection: RAGCollection,
        dense_vector: list[float],
        sparse_vector: dict[int, float],
        limit: int = 10,
        filters: dict | None = None,
    ) -> list[dict]:
        """
        Hybrid search: fuse dense (semantic) + sparse (BM25) results
        using Reciprocal Rank Fusion (RRF).
        """
        qdrant_filter = _build_filter(filters) if filters else None

        # Dense search
        dense_results = await self._client.query_points(
            collection_name=collection.value,
            query=dense_vector,
            using=_DENSE_VECTOR_NAME,
            limit=limit * 2,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # Sparse (BM25) search
        sparse_results = await self._client.query_points(
            collection_name=collection.value,
            query=qmodels.SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values()),
            ),
            using=_SPARSE_VECTOR_NAME,
            limit=limit * 2,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        # RRF fusion
        fused = _rrf_fuse(
            dense=[(p.id, p.score, p.payload) for p in dense_results.points],
            sparse=[(p.id, p.score, p.payload) for p in sparse_results.points],
            k=60,
            limit=limit,
        )
        return fused


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_payload(chunk: Chunk) -> dict[str, Any]:
    m = chunk.metadata
    return {
        "document_id": chunk.document_id,
        "rag_collection": chunk.rag_collection.value,
        "text": chunk.text_redacted if (m and m.has_pii) else chunk.text,
        "chunk_index": m.chunk_index if m else 0,
        "total_chunks": m.total_chunks if m else 0,
        "page_number": m.page_number if m else None,
        "section_title": m.section_title if m else None,
        "language": m.language if m else None,
        "entities": m.entities if m else [],
        "summary": m.summary if m else None,
        "has_pii": m.has_pii if m else False,
        "token_count": m.token_count if m else None,
    }


def _build_filter(filters: dict) -> qmodels.Filter:
    conditions = []
    for key, value in filters.items():
        conditions.append(
            qmodels.FieldCondition(
                key=key,
                match=qmodels.MatchValue(value=value),
            )
        )
    return qmodels.Filter(must=conditions)


def _rrf_fuse(
    dense: list[tuple],
    sparse: list[tuple],
    k: int = 60,
    limit: int = 10,
) -> list[dict]:
    """Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    payloads: dict[str, Any] = {}

    for rank, (pid, _, payload) in enumerate(dense):
        key = str(pid)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        payloads[key] = payload

    for rank, (pid, _, payload) in enumerate(sparse):
        key = str(pid)
        scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
        payloads.setdefault(key, payload)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"id": pid, "score": score, "payload": payloads.get(pid, {})}
        for pid, score in ranked[:limit]
    ]
