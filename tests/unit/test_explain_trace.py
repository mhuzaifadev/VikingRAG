"""Explain / offline replay shapes and auth-style scope stripping."""

from __future__ import annotations

from uuid import uuid4

from vikingrag.application.experience.policy import apply_learning_policy
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.experience import (
    LearningPolicy,
    QueryRun,
    QueryRunRoute,
    QueryRunStatus,
    RetrievalEvent,
    RetrievalEventType,
    new_query_run_id,
    new_retrieval_event_id,
)
from vikingrag.evaluation.runner import (
    MethodResult,
    export_latency_and_usage,
    resolve_methods,
)


def test_resolve_methods_includes_flat_rag() -> None:
    class _Gen:
        async def generate(self, *a: object, **k: object) -> object:
            del a, k
            raise RuntimeError("unused")

    class _OK:
        def answer_generator(self) -> _Gen:
            return _Gen()

    methods = resolve_methods(["flat_rag", "vikingrag"], client=_OK())
    assert "flat_rag" in methods
    assert "vikingrag" in methods


def test_export_latency_percentiles() -> None:
    results = [
        MethodResult(method="vikingrag", example_id="a", answer="x", latency_ms=10.0),
        MethodResult(method="vikingrag", example_id="b", answer="y", latency_ms=20.0),
        MethodResult(method="vikingrag", example_id="c", answer="z", latency_ms=30.0),
        MethodResult(
            method="vikingrag", example_id="d", answer=None, latency_ms=99.0, error="boom"
        ),
    ]
    metrics = export_latency_and_usage(results)
    assert metrics["vikingrag"]["n"] == 4
    assert metrics["vikingrag"]["errors"] == 1
    assert metrics["vikingrag"]["latency_ms"]["p50"] == 20.0
    assert metrics["vikingrag"]["latency_ms"]["p95"] is not None


def test_frozen_policy_on_run_shape() -> None:
    run = QueryRun(
        id=new_query_run_id(),
        query_text="q",
        status=QueryRunStatus.SUCCEEDED,
        route=QueryRunRoute.ONE_ROUND,
        learning_policy=LearningPolicy.LEARN,
        document_ids=(DocumentId(uuid4()),),
        metadata={
            "query_id": str(uuid4()),
            "escalation_reason": "insufficient_evidence",
            "citation_uris": ["viking://documents/x"],
        },
    )
    apply_learning_policy(run, LearningPolicy.FROZEN)
    assert run.learning_policy is LearningPolicy.FROZEN
    assert run.edge_build_status.value == "skipped"

    ev = RetrievalEvent(
        id=new_retrieval_event_id(),
        query_run_id=run.id,
        round_no=0,
        event_type=RetrievalEventType.EDGE_EXPAND,
        arguments={
            "edge_id": str(uuid4()),
            "source_uri": "viking://documents/a",
            "target_uri": "viking://documents/b",
            "similarity": 0.91,
            "text": "secret evidence",
        },
        result_refs=["viking://documents/b"],
    )
    # Shape expected by explain when stripping evidence
    stripped = {k: v for k, v in ev.arguments.items() if k not in {"text", "quote", "content"}}
    assert "text" not in stripped
    assert stripped["similarity"] == 0.91
