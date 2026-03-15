"""
Central configuration — all values come from environment variables or .env files.
Never hardcode secrets or infrastructure addresses in source code.
"""
from __future__ import annotations

from enum import Enum
from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


# ---------------------------------------------------------------------------
# Sub-settings (grouped by concern)
# ---------------------------------------------------------------------------

class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="APP_", env_file=".env", extra="ignore")

    name: str = "rag-ingestion-pipeline"
    version: str = "1.0.0"
    environment: Environment = Environment.DEVELOPMENT
    log_level: LogLevel = LogLevel.INFO
    debug: bool = False


class APISettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="API_", env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    cors_origins: list[str] = ["*"]
    api_key_header: str = "X-API-Key"
    api_key: str = Field(..., description="Master API key for securing endpoints")
    rate_limit_per_minute: int = 60
    max_upload_size_mb: int = 500


class PostgresSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="POSTGRES_", env_file=".env", extra="ignore")

    host: str = "postgres"
    port: int = 5432
    db: str = "rag_pipeline"
    user: str = "rag_user"
    password: str = Field(..., description="Postgres password")
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

    @property
    def dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )

    @property
    def sync_dsn(self) -> str:
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.db}"
        )


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")

    host: str = "redis"
    port: int = 6379
    password: str = ""
    db_cache: int = 0
    db_queue: int = 1
    db_dlq: int = 2
    ttl_seconds: int = 3600
    max_connections: int = 20

    @property
    def url(self) -> str:
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}"


class MinioSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MINIO_", env_file=".env", extra="ignore")

    endpoint: str = "minio:9000"
    access_key: str = Field(..., description="MinIO access key")
    secret_key: str = Field(..., description="MinIO secret key")
    secure: bool = False
    bucket_raw: str = "raw-documents"
    bucket_processed: str = "processed-documents"
    bucket_dlq: str = "dlq-documents"
    presigned_url_expiry: int = 3600


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")

    host: str = "qdrant"
    port: int = 6333
    grpc_port: int = 6334
    api_key: str = ""
    prefer_grpc: bool = True
    timeout: int = 30
    # Dense vector size depends on embedding model — configure per collection
    default_dense_size: int = 1024
    # Sparse vector config (BM25)
    sparse_model: str = "Qdrant/bm25"

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")

    base_url: str = "http://ollama:11434"
    classifier_model: str = "llama3.1:8b"
    chunker_model: str = "llama3.1:8b"
    ner_model: str = "llama3.1:8b"
    summarizer_model: str = "llama3.1:8b"
    pii_model: str = "llama3.1:8b"
    timeout: int = 120
    max_retries: int = 3


class HuggingFaceSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="HF_", env_file=".env", extra="ignore")

    # Dense embedding model
    dense_embedding_model: str = "BAAI/bge-m3"
    dense_embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    dense_batch_size: int = 32
    # Reranker model
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: Literal["cpu", "cuda", "mps"] = "cpu"
    # NER model (spaCy/transformers)
    ner_model: str = "dslim/bert-base-NER"
    # Language detection
    language_model: str = "papluca/xlm-roberta-base-language-detection"
    # Cache dir for downloaded models
    cache_dir: str = "/models/huggingface"
    hub_token: str = ""


class WorkerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="WORKER_", env_file=".env", extra="ignore")

    concurrency: int = 4
    queue_name: str = "ingestion"
    dlq_name: str = "ingestion_dlq"
    max_retries: int = 3
    retry_backoff_seconds: int = 30
    task_timeout_seconds: int = 600
    # LangGraph checkpointer backend
    checkpointer_backend: Literal["postgres", "memory"] = "postgres"


class ChunkingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CHUNKING_", env_file=".env", extra="ignore")

    # LLM-driven chunking uses Ollama to decide strategy per document
    llm_driven: bool = True
    # Fallback strategy if LLM is unavailable
    fallback_strategy: Literal["semantic", "fixed", "sentence", "recursive"] = "recursive"
    fixed_chunk_size: int = 512
    fixed_chunk_overlap: int = 64
    semantic_breakpoint_threshold: float = 0.85
    min_chunk_size: int = 50
    max_chunk_size: int = 2048


class ProcessingSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PROCESSING_", env_file=".env", extra="ignore")

    enable_ner: bool = True
    enable_summarization: bool = True
    enable_deduplication: bool = True
    enable_language_detection: bool = True
    enable_pii_redaction: bool = True
    # Similarity threshold for deduplication (cosine)
    dedup_threshold: float = 0.95
    # Max summary length in tokens
    summary_max_tokens: int = 256
    # PII entities to redact
    pii_entities: list[str] = [
        "PERSON", "EMAIL", "PHONE_NUMBER", "CREDIT_CARD",
        "SSN", "IP_ADDRESS", "IBAN_CODE", "URL",
    ]


class ObservabilitySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OBSERVABILITY_", env_file=".env", extra="ignore")

    prometheus_port: int = 9090
    metrics_path: str = "/metrics"
    enable_tracing: bool = False  # OpenTelemetry — off by default
    otel_endpoint: str = ""
    langsmith_api_key: str = ""
    langsmith_project: str = "rag-ingestion"
    langsmith_tracing: bool = False


# ---------------------------------------------------------------------------
# Root Settings — composes all sub-settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app: AppSettings = AppSettings()
    api: APISettings = APISettings()
    postgres: PostgresSettings = PostgresSettings()
    redis: RedisSettings = RedisSettings()
    minio: MinioSettings = MinioSettings()
    qdrant: QdrantSettings = QdrantSettings()
    ollama: OllamaSettings = OllamaSettings()
    huggingface: HuggingFaceSettings = HuggingFaceSettings()
    worker: WorkerSettings = WorkerSettings()
    chunking: ChunkingSettings = ChunkingSettings()
    processing: ProcessingSettings = ProcessingSettings()
    observability: ObservabilitySettings = ObservabilitySettings()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance — import this everywhere."""
    return Settings()
