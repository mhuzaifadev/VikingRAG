"""Search request/response model validation."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.domain.models.document import DocumentId, NodeId, NodeType
from vikingrag.domain.models.representation import (
    EmbeddingIdentity,
    RepresentationType,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SearchTiming,
    SearchUsage,
)


def test_search_request_rejects_empty_query() -> None:
    with pytest.raises(ValueError):
        SearchRequest(query="   ")


def test_search_request_rejects_bad_min_score() -> None:
    with pytest.raises(ValueError):
        SearchRequest(query="hello", min_score=1.5)


def test_search_response_round_trip_fields() -> None:
    hit = SearchHit(
        uri="viking://documents/x/nodes/y",
        title="Evidence Verification",
        node_id=NodeId(uuid4()),
        document_id=DocumentId(uuid4()),
        node_type=NodeType.SUBSECTION,
        representation_type=RepresentationType.NODE_SUMMARY,
        score=0.84,
        similarity=0.84,
        preview="claim evidence",
    )
    response = SearchResponse(
        query="How do we verify retrieved evidence?",
        query_id=uuid4(),
        candidates=(hit,),
        timing=SearchTiming(total_ms=12.0),
        usage=SearchUsage(embedding_calls=1, vector_searches=1, nodes_returned=1),
        embedding_identity=EmbeddingIdentity(provider="deterministic", model="m", dimensions=1536),
    )
    assert response.candidates[0].title == "Evidence Verification"
    assert response.usage.nodes_returned == 1
