"""Persist a completed query run and mark it PENDING for the edge builder worker."""

from __future__ import annotations

from typing import Any

from vikingrag.application.orchestration.query import mark_run_for_edge_build
from vikingrag.domain.models.experience import (
    QueryRun,
    QueryRunRoute,
    QueryRunStatus,
    RetrievalEvent,
    RetrievalEventType,
    new_query_run_id,
    new_retrieval_event_id,
)
from vikingrag.domain.models.representation import EmbeddingIdentity
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.experience import (
    SqlQueryRunRepository,
    SqlRetrievalEventRepository,
)
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)

_EVENT_MAP = {
    "Search": RetrievalEventType.SEARCH,
    "List": RetrievalEventType.LIST,
    "Grep": RetrievalEventType.GREP,
    "Read": RetrievalEventType.READ,
    "Stop": RetrievalEventType.ANSWER,
    "EDGE_EXPAND": RetrievalEventType.EDGE_EXPAND,
    "EdgeExpand": RetrievalEventType.EDGE_EXPAND,
}


async def enqueue_experience_learning(
    database: Database,
    *,
    query_text: str,
    answer: str | None,
    status: QueryRunStatus,
    route: QueryRunRoute,
    query_embedding: list[float] | None,
    embedding_identity: EmbeddingIdentity | None,
    trace_events: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    citation_uris: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> QueryRun | None:
    """Write query_run + retrieval_events, set edge_build_status=PENDING when learnable.

    Returns the persisted run, or None when nothing was enqueued.
    """
    run = QueryRun(
        id=new_query_run_id(),
        query_text=query_text,
        status=status,
        route=route,
        answer=answer,
        query_embedding=query_embedding,
        embedding_identity=embedding_identity,
        metadata={
            **(metadata or {}),
            "citation_uris": list(citation_uris or []),
        },
    )
    if query_embedding is None or embedding_identity is None:
        from vikingrag.domain.models.experience import EdgeBuildStatus

        run.edge_build_status = EdgeBuildStatus.SKIPPED
        run.edge_build_error = "missing_query_embedding"
    else:
        mark_run_for_edge_build(run)

    async with database.session() as session:
        runs = SqlQueryRunRepository(session)
        events = SqlRetrievalEventRepository(session)
        run = await runs.create(run)
        for i, ev in enumerate(trace_events):
            name = str(ev.get("name") or ev.get("event_type") or "")
            etype = _EVENT_MAP.get(name)
            if etype is None and name == "edge_expand":
                etype = RetrievalEventType.EDGE_EXPAND
            if etype is None:
                continue
            refs = ev.get("result_uris") or ev.get("result_refs") or []
            if isinstance(refs, str):
                refs = [refs]
            await events.append(
                RetrievalEvent(
                    id=new_retrieval_event_id(),
                    query_run_id=run.id,
                    round_no=int(ev.get("round_no") or i),
                    event_type=etype,
                    arguments=dict(ev.get("arguments") or {}),
                    result_refs=[str(r) for r in refs],
                    latency_ms=ev.get("latency_ms"),
                    token_cost=ev.get("token_cost"),
                )
            )

    logger.info(
        "experience_learning_enqueued",
        run_id=str(run.id),
        edge_build_status=run.edge_build_status.value,
        events=len(trace_events),
    )
    return run
