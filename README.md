# RAG Ingestion Pipeline

A production-grade, multi-collection document ingestion system built with **Clean Architecture**, **LangGraph**, **LangChain**, and a fully containerized infrastructure stack.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Infrastructure Stack](#infrastructure-stack)
4. [Pipeline Graph](#pipeline-graph)
5. [Clean Architecture Layers](#clean-architecture-layers)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Getting Started](#getting-started)
9. [Observability](#observability)
10. [Extending the System](#extending-the-system)
11. [Design Decisions](#design-decisions)

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        CLIENT / EXTERNAL SYSTEM                         │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ HTTP (REST)
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          FASTAPI (api service)                          │
│  POST /ingest  │  GET /jobs/:id  │  POST /search  │  Admin / Metrics    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ enqueue
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                    REDIS STREAMS (task queue)                           │
│          Queue: ingestion        DLQ: ingestion_dlq                     │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ dequeue
                               ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                     WORKER (LangGraph pipeline)                         │
│                                                                         │
│  ┌──────────┐   ┌──────────┐   ┌───────────────┐   ┌──────────────┐   │
│  │ Extract  │──▶│ Classify │──▶│ Decide Chunk  │──▶│    Chunk     │   │
│  │ (parser  │   │ (Ollama  │   │ Strategy      │   │ (LangChain   │   │
│  │  per     │   │  LLM)    │   │ (Ollama LLM)  │   │  splitters)  │   │
│  │ filetype)│   └──────────┘   └───────────────┘   └──────┬───────┘   │
│  └──────────┘                                              │           │
│                                                            ▼           │
│  ┌──────────┐   ┌──────────┐   ┌───────────────┐   ┌──────────────┐   │
│  │  Index   │◀──│  Embed   │◀──│  PII Redact   │◀──│    NER +     │   │
│  │ (Qdrant) │   │ (BGE-M3  │   │  (Presidio)   │   │ Summarize +  │   │
│  │  hybrid) │   │  + BM25) │   │               │   │   Dedup      │   │
│  └──────────┘   └──────────┘   └───────────────┘   └──────────────┘   │
│                                                                         │
│         ┌──────────────────────────────────────┐                       │
│         │         Error Handler                │                       │
│         │  retry (< max) ──▶ back to Extract   │                       │
│         │  max retries    ──▶ DLQ terminal      │                       │
│         └──────────────────────────────────────┘                       │
└─────────────────────────────────────────────────────────────────────────┘
         │              │              │              │
         ▼              ▼              ▼              ▼
    PostgreSQL        MinIO          Qdrant          Redis
  (metadata,       (raw files,    (dense+sparse    (cache,
   chunks,          processed,     vectors per      queue,
   jobs, graph      DLQ files)     collection)      DLQ)
   checkpoints)
```

---

## Project Structure

```
rag-ingestion-pipeline/
│
├── config/
│   └── settings.py              # All config via pydantic-settings (env vars)
│
├── src/
│   ├── domain/                  # ◀ Pure Python. Zero framework imports.
│   │   ├── entities/
│   │   │   └── models.py        # Document, Chunk, IngestionJob, enums
│   │   └── repositories/
│   │       └── interfaces.py    # Abstract ports (ABC) for all infra
│   │
│   ├── application/             # ◀ Use cases. Orchestrates domain + infra.
│   │   └── use_cases/
│   │       └── ingestion.py     # IngestDocument, Search, JobStatus, RequeueD LQ
│   │
│   ├── core/                    # ◀ LangGraph pipeline (framework layer)
│   │   ├── state/
│   │   │   └── pipeline_state.py   # Typed PipelineState TypedDict
│   │   ├── nodes/
│   │   │   └── pipeline_nodes.py   # All 13 graph nodes (pure async fns)
│   │   └── graph/
│   │       └── pipeline_graph.py   # Graph builder + edge wiring
│   │
│   ├── infrastructure/          # ◀ Concrete implementations of domain ports
│   │   ├── database/
│   │   │   └── postgres_repositories.py   # SQLAlchemy async
│   │   ├── vector_store/
│   │   │   └── qdrant_repository.py       # Hybrid dense+sparse, RRF fusion
│   │   ├── embedding/
│   │   │   └── hybrid_embedder.py         # BGE-M3 (dense) + BM25 (sparse)
│   │   ├── messaging/
│   │   │   └── redis_queue.py             # Redis Streams + DLQ
│   │   └── storage/
│   │       └── minio_repository.py        # S3-compatible file storage
│   │
│   └── adapters/               # ◀ Translates between domain and the world
│       ├── api/
│       │   ├── app.py           # FastAPI app, all routers
│       │   └── dependencies.py  # DI container + get_container()
│       ├── parsers/
│       │   └── registry.py      # FileType → Parser (PDF/DOCX/Excel/...)
│       └── processors/
│           └── registry.py      # Chunkers, NER, Summarizer, Dedup, PII
│
├── services/
│   └── worker/
│       └── worker.py            # Async worker: Redis consumer + graph runner
│
├── tests/
│   ├── unit/
│   │   └── test_domain.py       # Domain entities + node unit tests
│   └── integration/
│       └── test_pipeline.py     # Full API integration tests
│
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.worker
│   ├── postgres/init.sql
│   ├── qdrant/config.yaml
│   ├── prometheus/
│   │   ├── prometheus.yml
│   │   └── rules/pipeline_alerts.yml
│   └── grafana/
│       ├── provisioning/
│       └── dashboards/pipeline.json
│
├── requirements/
│   ├── base.txt                 # Shared: langchain, langgraph, infra clients
│   ├── api.txt                  # API: fastapi, parsers, presidio
│   └── worker.txt               # Worker: same parsers as API
│
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```

---

## Infrastructure Stack

| Service | Image | Role | Port |
|---|---|---|---|
| **PostgreSQL 16** | `postgres:16-alpine` | Document metadata, chunk records, job tracking, LangGraph checkpointer | 5432 |
| **Redis 7** | `redis:7-alpine` | Task queue (Streams), DLQ stream, embedding cache | 6379 |
| **MinIO** | `minio/minio:latest` | Raw file storage, processed files, DLQ file archive | 9000/9001 |
| **Qdrant 1.9** | `qdrant/qdrant:v1.9.2` | Hybrid vector DB (dense HNSW + sparse BM25) per collection | 6333/6334 |
| **Prometheus** | `prom/prometheus:v2.51.2` | Metrics scraping from API, Worker, Qdrant | 9090 |
| **Grafana** | `grafana/grafana:10.4.2` | Dashboards, alerting | 3000 |
| **API** | `./docker/Dockerfile.api` | FastAPI REST gateway | 8000 |
| **Worker** | `./docker/Dockerfile.worker` | LangGraph pipeline executor (2 replicas) | — |

### LLM / Embedding Providers

| Provider | Usage |
|---|---|
| **Ollama** (local) | Classification, chunking strategy, NER assist, summarization, PII detection |
| **HuggingFace** (local) | Dense embeddings (BGE-M3), reranking (BGE-Reranker-v2-M3), NER (BERT-NER), language detection |
| **BM25** (in-process) | Sparse vector generation via `rank-bm25` |

---

## Pipeline Graph

```
START
  │
  ▼
[extract]  ──── detect file type → call correct parser → load text from MinIO
  │ fail↘
  │      [handle_error]
  │         │ retry < max → back to [extract]
  │         │ retry >= max → [dlq_terminal] → END
  ▼
[classify]  ─── Ollama LLM classifies document → RAGCollection
  │
  ▼
[decide_chunking]  ─── Ollama LLM chooses strategy (semantic/fixed/recursive/...)
  │
  ▼
[chunk]  ─── apply chosen LangChain splitter
  │ fail↘ → [handle_error]
  │
  ▼
[detect_language]  ─── langdetect on first 512 chars
  │
  ▼
[ner]  ─── HuggingFace BERT-NER per chunk (non-fatal)
  │
  ▼
[summarize]  ─── Ollama LLM per chunk (non-fatal)
  │
  ▼
[deduplicate]  ─── cosine similarity on BGE-M3 vectors (threshold: 0.95)
  │
  ▼
[pii_redact]  ─── Presidio analyzer + anonymizer (regex fallback)
  │ fail↘ → [handle_error]
  │
  ▼
[embed]  ─── BGE-M3 dense + BM25 sparse, Redis-cached per chunk
  │ fail↘ → [handle_error]
  │
  ▼
[index]  ─── Qdrant upsert into classified collection (batch 100)
  │ fail↘ → [handle_error]
  │
  ▼
END
```

**LangGraph features used:**
- `StateGraph` with typed `PipelineState` (TypedDict)
- Conditional edges for error routing
- `AsyncPostgresSaver` checkpointer — every node's output is persisted, enabling resume after crash
- `astream()` for real-time status syncing to Postgres during execution

---

## Clean Architecture Layers

```
┌────────────────────────────────────────┐
│            Adapters Layer              │  FastAPI routes, Parsers, Processors
│  Knows: Application, Domain            │  Translates HTTP ↔ use cases
├────────────────────────────────────────┤
│          Application Layer             │  Use cases: IngestDocument, Search
│  Knows: Domain, Infrastructure ports   │  Orchestrates — no business logic
├────────────────────────────────────────┤
│      Core Layer (LangGraph)            │  Pipeline graph, nodes, state
│  Knows: Domain, Infrastructure ports   │  Framework-specific orchestration
├────────────────────────────────────────┤
│         Infrastructure Layer           │  Postgres, Qdrant, Redis, MinIO
│  Knows: Domain interfaces only         │  Implements domain repository ports
├────────────────────────────────────────┤
│            Domain Layer                │  Entities, value objects, repo ABCs
│  Knows: nothing external               │  Pure Python — zero framework imports
└────────────────────────────────────────┘
```

**The Dependency Rule is strictly enforced:**
- `domain/` imports nothing from other layers
- `infrastructure/` imports only `domain/` interfaces
- `application/` imports `domain/` and calls infrastructure via interfaces
- `adapters/` imports `application/` use cases only
- `core/` (LangGraph) imports `domain/` entities and infrastructure interfaces

---

## Configuration

**All configuration is externalized.** No values are hardcoded in source code.

```bash
# Copy and fill in required values
cp .env.example .env
```

The `config/settings.py` uses **pydantic-settings** with sub-settings grouped by concern:

| Settings class | Env prefix | Key fields |
|---|---|---|
| `AppSettings` | `APP_` | environment, log_level, debug |
| `APISettings` | `API_` | host, port, api_key, max_upload_size_mb |
| `PostgresSettings` | `POSTGRES_` | host, port, db, user, password |
| `RedisSettings` | `REDIS_` | host, port, db_cache, db_queue, db_dlq |
| `MinioSettings` | `MINIO_` | endpoint, access_key, secret_key, buckets |
| `QdrantSettings` | `QDRANT_` | host, port, grpc_port, default_dense_size |
| `OllamaSettings` | `OLLAMA_` | base_url, per-task model names, timeout |
| `HuggingFaceSettings` | `HF_` | model names, device, batch_size, cache_dir |
| `WorkerSettings` | `WORKER_` | concurrency, max_retries, retry_backoff |
| `ChunkingSettings` | `CHUNKING_` | llm_driven, fallback_strategy, sizes |
| `ProcessingSettings` | `PROCESSING_` | enable_*/disable_* per stage, thresholds |
| `ObservabilitySettings` | `OBSERVABILITY_` | prometheus_port, tracing, langsmith |

Access settings anywhere:
```python
from config.settings import get_settings
settings = get_settings()  # cached singleton
```

---

## API Reference

### Authentication
All endpoints require `X-API-Key` header matching `API_KEY` in `.env`.

### Endpoints

#### `POST /api/v1/ingest`
Upload a document for async ingestion.

**Request:** `multipart/form-data`
| Field | Type | Required | Description |
|---|---|---|---|
| `file` | File | ✅ | Document file |
| `tags` | string | ❌ | Comma-separated tags |
| `webhook_url` | string | ❌ | Callback URL on completion |

**Response `202`:**
```json
{
  "job_id": "uuid",
  "document_id": "uuid",
  "status": "pending",
  "message": "Document accepted for processing."
}
```

#### `GET /api/v1/jobs/{job_id}`
Poll ingestion job status.

**Response `200`:**
```json
{
  "job_id": "uuid",
  "document_id": "uuid",
  "status": "completed",
  "errors": [],
  "created_at": "2024-01-01T00:00:00Z",
  "updated_at": "2024-01-01T00:01:23Z",
  "completed_at": "2024-01-01T00:01:23Z"
}
```

Status lifecycle: `pending → extracting → classifying → chunking → processing → embedding → indexing → completed`

On failure: `→ failed → (retry) → ... → dlq`

#### `POST /api/v1/search`
Hybrid semantic + BM25 search across RAG collections.

**Request:**
```json
{
  "query": "microservices architecture patterns",
  "collection": "technical",
  "limit": 10,
  "filters": { "language": "en" }
}
```

**Response `200`:**
```json
{
  "query": "...",
  "results": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid",
      "text": "...",
      "score": 0.847,
      "metadata": { "page_number": 3, "section_title": "..." }
    }
  ],
  "total": 10
}
```

#### `GET /api/v1/admin/dlq`
List dead-letter queue items for manual review.

#### `POST /api/v1/admin/dlq/{message_id}/requeue`
Requeue a failed document for reprocessing.

#### `GET /api/v1/admin/collections`
List available RAG collections.

#### `GET /health` / `GET /ready`
Liveness and readiness probes.

#### `GET /metrics`
Prometheus metrics endpoint.

---

## Getting Started

### Prerequisites
- Docker + Docker Compose
- Ollama running locally or accessible (with `llama3.1:8b` pulled)
- 8GB+ RAM for local embedding models

### 1. Configure environment
```bash
cp .env.example .env
# Edit .env — fill in passwords, API keys, Ollama URL
```

### 2. Start infrastructure + services
```bash
docker compose up -d
```

### 3. Pull required Ollama models
```bash
ollama pull llama3.1:8b
```

### 4. Verify everything is running
```bash
curl http://localhost:8000/ready
# Expected: {"status": "ready", "checks": {"postgres": true, "redis": true, ...}}
```

### 5. Ingest a document
```bash
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "X-API-Key: your_api_key" \
  -F "file=@/path/to/document.pdf" \
  -F "tags=finance,q3"
```

### 6. Poll status
```bash
curl http://localhost:8000/api/v1/jobs/{job_id} \
  -H "X-API-Key: your_api_key"
```

### 7. Search
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "X-API-Key: your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"query": "quarterly revenue", "limit": 5}'
```

### 8. Open Grafana
Navigate to `http://localhost:3000` (default: admin/admin)
→ Dashboards → RAG Pipeline → Pipeline Overview

---

## Observability

### Prometheus Metrics

| Metric | Type | Labels | Description |
|---|---|---|---|
| `rag_ingestion_total` | Counter | `file_type` | Documents submitted |
| `rag_ingestion_duration_seconds` | Histogram | `status` | E2E pipeline duration |
| `rag_search_total` | Counter | `collection` | Search queries |
| `rag_search_duration_seconds` | Histogram | — | Search latency |
| `rag_dlq_total` | Counter | — | Documents sent to DLQ |
| `worker_tasks_processed_total` | Counter | `status` | Worker task completions |
| `worker_pipeline_duration_seconds` | Histogram | `final_status` | Per-worker pipeline time |
| `worker_active_tasks` | Gauge | — | Concurrent running tasks |

### Alerting Rules
Configured in `docker/prometheus/rules/pipeline_alerts.yml`:
- **DLQGrowing** — > 5 DLQ entries in 5 minutes
- **PipelineSlowP95** — P95 duration > 5 minutes
- **WorkerIdle** — no tasks processed in 10 minutes
- **HighFailureRate** — failure rate > 20%
- **SearchLatencyHigh** — search P95 > 5 seconds

### Structured Logging
All services use `structlog` with JSON output, compatible with Loki/ELK/CloudWatch.

### LangGraph Checkpointing
Every node output is persisted to Postgres via `AsyncPostgresSaver`.
If a worker crashes mid-pipeline, the graph can be resumed from the last checkpoint.

---

## Extending the System

### Add a new file type parser
```python
# src/adapters/parsers/registry.py
class MyCustomParser(BaseParser):
    async def parse(self, file_key: str, metadata: dict) -> ParseResult:
        ...

# Register at startup in dependencies.py
parser_registry.register(FileType.MY_TYPE, MyCustomParser(storage))
```

### Add a new RAG collection
```python
# src/domain/entities/models.py
class RAGCollection(str, Enum):
    ...
    MY_DOMAIN = "my_domain"   # Add here
```
Qdrant collection is auto-created on first document indexed to it.

### Add a new processing stage
1. Write a new async node function in `src/core/nodes/pipeline_nodes.py`
2. Add it to `PipelineState` in `src/core/state/pipeline_state.py`
3. Wire it into the graph in `src/core/graph/pipeline_graph.py`
4. Register any service dependencies in `src/adapters/api/dependencies.py`

### Scale workers
```bash
docker compose up -d --scale worker=4
```
Workers share the Redis consumer group — tasks are automatically distributed.

### Switch embedding model
```bash
# .env
HF_DENSE_EMBEDDING_MODEL=BAAI/bge-large-en-v1.5
HF_DENSE_EMBEDDING_DEVICE=cuda
QDRANT_DEFAULT_DENSE_SIZE=1024
```
Update `QDRANT_DEFAULT_DENSE_SIZE` to match the new model's output dimensions.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **Clean Architecture** | Domain layer has zero framework dependencies — swap LangGraph for Prefect without touching domain logic |
| **Redis Streams** (not Lists) | Consumer groups enable horizontal worker scaling with at-least-once delivery guarantees |
| **Postgres checkpointer** | LangGraph state persisted per node — crash recovery without re-processing from scratch |
| **Hybrid embeddings** | Dense (BGE-M3) captures semantic meaning; sparse (BM25) captures exact keyword matches; RRF fusion outperforms either alone |
| **LLM-driven chunking** | Document structure varies wildly — an LLM evaluating content type and layout picks the right strategy per document |
| **Per-collection Qdrant** | Isolates RAG contexts, enables per-collection tuning of HNSW/quantization, prevents cross-domain noise in retrieval |
| **PII redact before index** | Redacted text is stored in Qdrant; original text stays in Postgres (access-controlled) — privacy by design |
| **DLQ + manual review** | Permanent failures need human eyes — silent skips lose documents forever; DLQ makes failures visible and recoverable |
| **`pydantic-settings`** | Type-safe config, automatic env var parsing, grouped by concern, `.env` file support, no magic strings in code |
