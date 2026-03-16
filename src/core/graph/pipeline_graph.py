"""
LangGraph Pipeline Graph Builder
Assembles the full ingestion graph with:
- Linear happy path
- LLM-driven chunking decision
- Error handling with retry loop
- Dead-letter queue terminal node
"""
from __future__ import annotations

from functools import partial
from typing import Any, Literal

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph

from config.settings import get_settings
from src.core.nodes.pipeline_nodes import (
    node_chunk,
    node_classify,
    node_decide_chunking,
    node_deduplicate,
    node_detect_language,
    node_embed,
    node_extract,
    node_handle_error,
    node_index,
    node_ner,
    node_pii_redact,
    node_summarize,
)
from src.core.state.pipeline_state import PipelineState
from src.domain.entities.models import DocumentStatus

settings = get_settings()


# ---------------------------------------------------------------------------
# Edge condition functions
# ---------------------------------------------------------------------------

def route_after_node(
    state: PipelineState,
    success_next: str,
) -> Literal["handle_error", str]:
    """Generic routing: if current_status is FAILED, go to error handler."""
    if state.get("current_status") == DocumentStatus.FAILED:
        return "handle_error"
    return success_next


def route_after_error(
    state: PipelineState,
) -> Literal["extract", "dlq_terminal"]:
    """After error handling: retry from extract or go to DLQ."""
    if state.get("send_to_dlq"):
        return "dlq_terminal"
    return "extract"  # Always re-enter from extraction on retry


async def node_dlq_terminal(state: PipelineState, queue_repo: Any) -> dict:
    """Persist DLQ entry and mark pipeline as terminal."""
    await queue_repo.enqueue_dlq(
        message={
            "document_id": state.get("document_id"),
            "job_id": state.get("job_id"),
            "errors": [
                {"stage": e.stage, "message": e.message}
                for e in (state.get("errors") or [])
            ],
        },
        reason=state.get("dlq_reason", "unknown"),
    )
    return {"current_status": DocumentStatus.DLQ}


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_pipeline_graph(
    parser_registry: Any,
    chunker_registry: Any,
    language_detector: Any,
    ner_extractor: Any,
    summarizer: Any,
    deduplicator: Any,
    pii_redactor: Any,
    embedder: Any,
    vector_repo: Any,
    queue_repo: Any,
    checkpointer: Any = None,
) -> Any:
    """
    Build and compile the full ingestion StateGraph.

    All service dependencies are injected here — nodes receive them
    via functools.partial so the graph itself stays stateless.

    Graph topology:
    ┌─────────────────────────────────────────────────────────────────┐
    │  START                                                          │
    │    ↓                                                            │
    │  extract ──[fail]──→ handle_error ──[dlq]──→ dlq_terminal → END│
    │    ↓                     ↑                                      │
    │  classify                │  [retry]←──────────────────────────┐│
    │    ↓                     │                                     ││
    │  decide_chunking         │                                     ││
    │    ↓                     │                                     ││
    │  chunk ──[fail]──────────┘                                     ││
    │    ↓                                                            ││
    │  detect_language (parallel)                                     ││
    │    ↓                                                            ││
    │  ner ──[fail]──[non-fatal, continues]                          ││
    │    ↓                                                            ││
    │  summarize ──[fail]──[non-fatal, continues]                    ││
    │    ↓                                                            ││
    │  deduplicate                                                    ││
    │    ↓                                                            ││
    │  pii_redact                                                     ││
    │    ↓                                                            ││
    │  embed ──[fail]──────────────────────────────────────────────┘ ││
    │    ↓                                                            ││
    │  index ──[fail]───────────────────────────────────────────────┘│
    │    ↓                                                            │
    │  END                                                            │
    └─────────────────────────────────────────────────────────────────┘
    """

    graph = StateGraph(PipelineState)

    # --- Register nodes with injected dependencies ---
    graph.add_node(
        "extract",
        partial(node_extract, parser_registry=parser_registry),
    )
    graph.add_node("classify", node_classify)
    graph.add_node("decide_chunking", node_decide_chunking)
    graph.add_node(
        "chunk",
        partial(node_chunk, chunker_registry=chunker_registry),
    )
    graph.add_node(
        "detect_language",
        partial(node_detect_language, language_detector=language_detector),
    )
    graph.add_node(
        "ner",
        partial(node_ner, ner_extractor=ner_extractor),
    )
    graph.add_node(
        "summarize",
        partial(node_summarize, summarizer=summarizer),
    )
    graph.add_node(
        "deduplicate",
        partial(node_deduplicate, deduplicator=deduplicator),
    )
    graph.add_node(
        "pii_redact",
        partial(node_pii_redact, pii_redactor=pii_redactor),
    )
    graph.add_node(
        "embed",
        partial(node_embed, embedder=embedder),
    )
    graph.add_node(
        "index",
        partial(node_index, vector_repo=vector_repo),
    )
    graph.add_node("handle_error", node_handle_error)
    graph.add_node(
        "dlq_terminal",
        partial(node_dlq_terminal, queue_repo=queue_repo),
    )

    # --- Edges: happy path ---
    graph.add_edge(START, "extract")

    graph.add_conditional_edges(
        "extract",
        partial(route_after_node, success_next="classify"),
        {"classify": "classify", "handle_error": "handle_error"},
    )
    graph.add_edge("classify", "decide_chunking")
    graph.add_edge("decide_chunking", "chunk")

    graph.add_conditional_edges(
        "chunk",
        partial(route_after_node, success_next="detect_language"),
        {"detect_language": "detect_language", "handle_error": "handle_error"},
    )

    # Processing nodes: language detection feeds NER chain
    graph.add_edge("detect_language", "ner")
    graph.add_edge("ner", "summarize")
    graph.add_edge("summarize", "deduplicate")
    graph.add_edge("deduplicate", "pii_redact")

    graph.add_conditional_edges(
        "pii_redact",
        partial(route_after_node, success_next="embed"),
        {"embed": "embed", "handle_error": "handle_error"},
    )

    graph.add_conditional_edges(
        "embed",
        partial(route_after_node, success_next="index"),
        {"index": "index", "handle_error": "handle_error"},
    )

    graph.add_conditional_edges(
        "index",
        partial(route_after_node, success_next=END),
        {END: END, "handle_error": "handle_error"},
    )

    # --- Error routing ---
    graph.add_conditional_edges(
        "handle_error",
        route_after_error,
        {"extract": "extract", "dlq_terminal": "dlq_terminal"},
    )
    graph.add_edge("dlq_terminal", END)

    # --- Compile with optional checkpointer ---
    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    return graph.compile(**compile_kwargs)


async def create_checkpointer(postgres_dsn: str) -> AsyncPostgresSaver:
    """Create and set up the Postgres-backed LangGraph checkpointer."""
    from psycopg_pool import AsyncConnectionPool

    pool = AsyncConnectionPool(
        conninfo=postgres_dsn,
        open=False,
        kwargs={"autocommit": True, "prepare_threshold": 0},
    )
    await pool.open()
    checkpointer = AsyncPostgresSaver(conn=pool)
    await checkpointer.setup()
    return checkpointer
