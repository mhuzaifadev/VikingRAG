from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.domain.models import (
    DocumentId,
    EvidenceBundle,
    RetrievalBudget,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalTrace,
    RetrievedEvidence,
)


def test_retrieval_query_validation() -> None:
    q = RetrievalQuery(text="What is OAuth?")
    assert q.top_k == 8
    with pytest.raises(ValueError):
        RetrievalQuery(text="   ")
    with pytest.raises(ValueError):
        RetrievalQuery(text="ok", top_k=0)


def test_retrieval_candidate_and_budget() -> None:
    cand = RetrievalCandidate(
        uri="viking://doc/1/chunk-1",
        score=0.9,
        preview="preview",
        object_type="chunk",
        document_id=DocumentId(uuid4()),
    )
    assert cand.uri.startswith("viking://")
    budget = RetrievalBudget(max_rounds=4)
    assert budget.max_rounds == 4
    with pytest.raises(ValueError):
        RetrievalBudget(max_tool_calls=0)


def test_evidence_bundle() -> None:
    item = RetrievedEvidence(uri="viking://x", text="hello", token_count=2)
    bundle = EvidenceBundle.from_items([item])
    assert bundle.total_tokens == 2
    assert len(bundle.items) == 1


def test_retrieval_trace_records_events() -> None:
    trace = RetrievalTrace(query_id=uuid4())
    trace.record("search", hits=3)
    assert len(trace.events) == 1
    assert trace.events[0]["event_type"] == "search"
