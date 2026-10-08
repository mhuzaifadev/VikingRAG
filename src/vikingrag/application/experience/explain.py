"""Replayable retrieval decision export (offline — no new LLM calls)."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from vikingrag.domain.errors import DocumentNotFound, ScopeDeniedError
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.experience import QueryRunId
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.experience import (
    SqlQueryRunRepository,
    SqlRetrievalEventRepository,
)


async def explain_query_run(
    database: Database,
    *,
    query_id: UUID,
    permitted_document_ids: frozenset[DocumentId] | None = None,
) -> dict[str, Any]:
    """Return a structured explanation for a persisted query run / answer.

    Looks up by query_run primary key first, then by ``metadata.query_id``.
    Evidence text is omitted when the caller is outside document scope.
    """
    async with database.session() as session:
        runs = SqlQueryRunRepository(session)
        events = SqlRetrievalEventRepository(session)
        run = await runs.get(QueryRunId(query_id))
        if run is None:
            run = await runs.get_by_answer_query_id(query_id)
        if run is None:
            raise DocumentNotFound(f"Query run not found: {query_id}")

        if permitted_document_ids is not None:
            if not permitted_document_ids and run.document_ids:
                raise ScopeDeniedError("Empty allowlist denies explain")
            if (
                permitted_document_ids
                and run.document_ids
                and not any(d in permitted_document_ids for d in run.document_ids)
            ):
                raise ScopeDeniedError("Query run outside permitted document scope")

        event_rows = await events.list_for_run(run.id)
        include_evidence = permitted_document_ids is None or (
            not run.document_ids or any(d in permitted_document_ids for d in run.document_ids)
        )

        trace: list[dict[str, Any]] = []
        contributing_edges: list[dict[str, Any]] = []
        for ev in event_rows:
            item: dict[str, Any] = {
                "round_no": ev.round_no,
                "event_type": ev.event_type.value,
                "arguments": dict(ev.arguments),
                "result_refs": list(ev.result_refs),
                "latency_ms": ev.latency_ms,
                "token_cost": ev.token_cost,
            }
            if not include_evidence:
                args = dict(item["arguments"])
                for key in ("text", "quote", "content", "snippet"):
                    args.pop(key, None)
                item["arguments"] = args
            trace.append(item)

            if ev.event_type.value in {"edge_expand", "EDGE_EXPAND"} or (
                str(ev.event_type.value).lower() == "edge_expand"
            ):
                args = dict(ev.arguments)
                edge_meta = {
                    "edge_id": args.get("edge_id"),
                    "source_uri": args.get("source_uri"),
                    "target_uri": args.get("target_uri") or args.get("uri"),
                    "activation_similarity": args.get("similarity")
                    or args.get("query_similarity")
                    or args.get("activation_similarity"),
                    "result_refs": list(ev.result_refs),
                }
                if not include_evidence:
                    edge_meta.pop("target_uri", None)
                contributing_edges.append(edge_meta)

        citation_uris = list(run.metadata.get("citation_uris") or [])
        return {
            "query_run_id": str(run.id),
            "answer_query_id": run.metadata.get("query_id"),
            "query_text": run.query_text,
            "status": run.status.value,
            "route": run.route.value if run.route else None,
            "answer": run.answer if include_evidence else None,
            "learning_policy": run.learning_policy.value,
            "snapshot_id": str(run.snapshot_id) if run.snapshot_id else None,
            "edge_build_status": run.edge_build_status.value,
            "document_ids": [str(d) for d in run.document_ids],
            "escalation_reason": run.metadata.get("escalation_reason"),
            "sufficiency": run.metadata.get("sufficiency") or run.metadata.get("assessment"),
            "execution_mode": run.metadata.get("execution_mode"),
            "usage": {
                "input_tokens": run.total_input_tokens,
                "output_tokens": run.total_output_tokens,
                "retrieval_tokens": run.retrieval_tokens,
                "llm_calls": run.llm_calls,
                "retrieval_rounds": run.retrieval_rounds,
                "latency_ms": run.latency_ms,
            },
            "citation_uris": citation_uris if include_evidence else [],
            "contributing_edges": contributing_edges,
            "trace_events": trace,
            "replay": "recorded",
            "notes": "Offline replay of persisted events; not a live LLM rerun.",
        }
