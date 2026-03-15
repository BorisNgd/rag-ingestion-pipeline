"""
FastAPI Application
Full REST API: upload, job status, search, DLQ management, metrics.
All routes are thin — they delegate entirely to use cases.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated, Any

import structlog
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)
from pydantic import BaseModel, Field
from starlette.responses import Response

from config.settings import get_settings
from src.application.use_cases.ingestion import (
    HybridSearchUseCase,
    IngestDocumentInput,
    IngestDocumentUseCase,
    GetJobStatusUseCase,
    RequeueDLQUseCase,
    SearchInput,
)
from src.domain.entities.models import RAGCollection
from src.adapters.api.dependencies import get_container

log = structlog.get_logger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Prometheus metrics
# ---------------------------------------------------------------------------

INGEST_COUNTER = Counter(
    "rag_ingestion_total",
    "Total documents submitted for ingestion",
    ["file_type"],
)
INGEST_LATENCY = Histogram(
    "rag_ingestion_duration_seconds",
    "Duration of ingestion pipeline",
    ["status"],
)
SEARCH_COUNTER = Counter(
    "rag_search_total",
    "Total search queries",
    ["collection"],
)
SEARCH_LATENCY = Histogram(
    "rag_search_duration_seconds",
    "Search query latency",
)
DLQ_GAUGE = Counter(
    "rag_dlq_total",
    "Total documents sent to DLQ",
)


# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    log.info("api.startup", version=settings.app.version)
    container = get_container()
    await container.startup()
    app.state.container = container
    yield
    log.info("api.shutdown")
    await container.shutdown()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    app = FastAPI(
        title="RAG Ingestion Pipeline API",
        version=settings.app.version,
        description="Document ingestion API for multi-collection RAG systems.",
        docs_url="/docs" if settings.app.environment.value != "production" else None,
        redoc_url="/redoc" if settings.app.environment.value != "production" else None,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(search_router, prefix="/api/v1")
    app.include_router(admin_router, prefix="/api/v1/admin")
    app.include_router(health_router)

    return app


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------

async def verify_api_key(
    x_api_key: Annotated[str | None, Header()] = None,
) -> None:
    if x_api_key != settings.api.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


# ---------------------------------------------------------------------------
# Ingestion Router
# ---------------------------------------------------------------------------

from fastapi import APIRouter

ingest_router = APIRouter(tags=["Ingestion"])


class IngestResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    message: str


@ingest_router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(verify_api_key)],
    summary="Upload a document for ingestion",
)
async def ingest_document(
    file: UploadFile = File(..., description="Document file to ingest"),
    tags: str = Form("", description="Comma-separated tags"),
    webhook_url: str = Form("", description="Optional webhook for completion callback"),
    request: Any = Depends(lambda: None),
    container: Any = Depends(get_container),
) -> IngestResponse:
    """
    Upload a document to the ingestion pipeline.
    Supported formats: PDF, DOCX, XLSX, HTML, images (OCR), audio, code, Markdown.
    Processing is asynchronous — poll `/jobs/{job_id}` for status.
    """
    if file.size and file.size > settings.api.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.api.max_upload_size_mb}MB limit.",
        )

    file_bytes = await file.read()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []

    use_case: IngestDocumentUseCase = container.ingest_use_case
    result = await use_case.execute(
        IngestDocumentInput(
            file_bytes=file_bytes,
            filename=file.filename or "unknown",
            content_type=file.content_type,
            tags=tag_list,
            webhook_url=webhook_url or None,
        )
    )

    INGEST_COUNTER.labels(file_type=file.content_type or "unknown").inc()

    return IngestResponse(
        job_id=result.job_id,
        document_id=result.document_id,
        status=result.status,
        message="Document accepted for processing.",
    )


class JobStatusResponse(BaseModel):
    job_id: str
    document_id: str
    status: str
    errors: list[dict]
    created_at: str
    updated_at: str
    completed_at: str | None


@ingest_router.get(
    "/jobs/{job_id}",
    response_model=JobStatusResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Get ingestion job status",
)
async def get_job_status(
    job_id: str,
    container: Any = Depends(get_container),
) -> JobStatusResponse:
    use_case: GetJobStatusUseCase = container.job_status_use_case
    result = await use_case.execute(job_id)

    if not result:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    return JobStatusResponse(
        job_id=result.job_id,
        document_id=result.document_id,
        status=result.status,
        errors=result.errors,
        created_at=result.created_at,
        updated_at=result.updated_at,
        completed_at=result.completed_at,
    )


# ---------------------------------------------------------------------------
# Search Router
# ---------------------------------------------------------------------------

search_router = APIRouter(tags=["Search"])


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2048)
    collection: str | None = Field(None, description="RAG collection to search")
    limit: int = Field(10, ge=1, le=100)
    filters: dict | None = None


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    total: int


@search_router.post(
    "/search",
    response_model=SearchResponse,
    dependencies=[Depends(verify_api_key)],
    summary="Hybrid search across RAG collections",
)
async def hybrid_search(
    req: SearchRequest,
    container: Any = Depends(get_container),
) -> SearchResponse:
    """
    Hybrid search combining dense (semantic) and sparse (BM25) retrieval
    with Reciprocal Rank Fusion.
    """
    collection = RAGCollection(req.collection) if req.collection else None
    use_case: HybridSearchUseCase = container.search_use_case

    SEARCH_COUNTER.labels(collection=req.collection or "all").inc()

    with SEARCH_LATENCY.time():
        results = await use_case.execute(
            SearchInput(
                query=req.query,
                collection=collection,
                limit=req.limit,
                filters=req.filters,
            )
        )

    return SearchResponse(
        query=req.query,
        results=[
            SearchResultItem(
                chunk_id=r.chunk_id,
                document_id=r.document_id,
                text=r.text,
                score=r.score,
                metadata=r.metadata,
            )
            for r in results
        ],
        total=len(results),
    )


# ---------------------------------------------------------------------------
# Admin Router (DLQ management)
# ---------------------------------------------------------------------------

admin_router = APIRouter(tags=["Admin"])


@admin_router.get(
    "/dlq",
    dependencies=[Depends(verify_api_key)],
    summary="List dead-letter queue items",
)
async def list_dlq(
    limit: int = Query(50, ge=1, le=500),
    container: Any = Depends(get_container),
) -> dict:
    items = await container.queue.list_dlq(limit=limit)
    return {"items": items, "total": len(items)}


@admin_router.post(
    "/dlq/{message_id}/requeue",
    dependencies=[Depends(verify_api_key)],
    summary="Requeue a DLQ document for reprocessing",
)
async def requeue_dlq(
    message_id: str,
    container: Any = Depends(get_container),
) -> dict:
    use_case: RequeueDLQUseCase = container.requeue_use_case
    success = await use_case.execute(message_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DLQ message {message_id} not found.",
        )
    return {"message": f"Message {message_id} requeued successfully."}


@admin_router.get(
    "/collections",
    dependencies=[Depends(verify_api_key)],
    summary="List available RAG collections",
)
async def list_collections() -> dict:
    return {
        "collections": [c.value for c in RAGCollection],
    }


# ---------------------------------------------------------------------------
# Health & Metrics Router
# ---------------------------------------------------------------------------

health_router = APIRouter(tags=["Health"])


@health_router.get("/health", summary="Liveness probe")
async def health() -> dict:
    return {"status": "ok", "version": settings.app.version}


@health_router.get("/ready", summary="Readiness probe")
async def readiness(container: Any = Depends(get_container)) -> dict:
    checks = await container.health_check()
    all_ok = all(v for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )


@health_router.get(
    settings.observability.metrics_path,
    summary="Prometheus metrics endpoint",
)
async def metrics() -> Response:
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

app = create_app()
