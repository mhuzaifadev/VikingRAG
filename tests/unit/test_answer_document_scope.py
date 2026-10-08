"""Answer path must intersect request.document_ids with auth allowlist."""

from __future__ import annotations

from uuid import uuid4

from vikingrag.application.answer.generate import intersect_document_scope
from vikingrag.domain.models.document import DocumentId


def test_intersect_request_narrows_unrestricted() -> None:
    a = DocumentId(uuid4())
    b = DocumentId(uuid4())
    scoped = intersect_document_scope(None, (a,))
    assert scoped == frozenset({a})
    assert b not in scoped


def test_intersect_request_with_allowlist() -> None:
    a = DocumentId(uuid4())
    b = DocumentId(uuid4())
    c = DocumentId(uuid4())
    allowed = frozenset({a, b})
    scoped = intersect_document_scope(allowed, (a, c))
    assert scoped == frozenset({a})


def test_intersect_empty_request_keeps_auth() -> None:
    a = DocumentId(uuid4())
    assert intersect_document_scope(frozenset({a}), ()) == frozenset({a})
    assert intersect_document_scope(None, ()) is None
