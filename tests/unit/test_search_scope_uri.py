"""SearchRequest.scope_uri plumbing (subtree containment)."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.domain.models.document import NodeId
from vikingrag.domain.models.representation import SearchRequest
from vikingrag.infrastructure.database.repositories.embedding import SqlEmbeddingRepository


def test_search_request_scope_uri_optional() -> None:
    req = SearchRequest(query="fees")
    assert req.scope_uri is None
    scoped = SearchRequest(query="fees", scope_uri="viking://doc/x#node")
    assert scoped.scope_uri == "viking://doc/x#node"


def test_search_request_rejects_blank_scope() -> None:
    with pytest.raises(ValueError, match="scope_uri"):
        SearchRequest(query="fees", scope_uri="   ")


def test_search_cosine_signature_accepts_scope_node_id() -> None:
    """Regression: repository method must accept scope_node_id kwarg."""
    import inspect

    sig = inspect.signature(SqlEmbeddingRepository.search_cosine)
    assert "scope_node_id" in sig.parameters
    # Type default is None
    assert sig.parameters["scope_node_id"].default is None
    # NodeId remains constructible for callers
    _ = NodeId(uuid4())
