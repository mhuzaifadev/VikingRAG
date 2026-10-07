"""Evidence bundle invariants and deduplication."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.application.evidence_bundle import assemble_evidence_bundle
from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.evidence import EvidenceBundle, ExclusionReason, RetrievedEvidence


def _item(**kwargs: object) -> RetrievedEvidence:
    defaults = {
        "uri": "viking://documents/x/nodes/y",
        "text": "hello evidence",
        "token_count": 2,
        "document_id": DocumentId(uuid4()),
        "node_id": NodeId(uuid4()),
        "content_hash": "abc",
        "start_offset": 0,
        "end_offset": 5,
        "evidence_id": "",
        "provenance": ("read",),
    }
    defaults.update(kwargs)
    return RetrievedEvidence(**defaults)  # type: ignore[arg-type]


def test_bundle_rejects_mismatched_total_tokens() -> None:
    item = _item()
    with pytest.raises(ValueError, match="total_tokens"):
        EvidenceBundle(items=(item,), total_tokens=99)


def test_dedupe_identical_ranges() -> None:
    doc = DocumentId(uuid4())
    node = NodeId(uuid4())
    a = _item(document_id=doc, node_id=node, content_hash="h1", start_offset=0, end_offset=10)
    b = _item(document_id=doc, node_id=node, content_hash="h1", start_offset=0, end_offset=10)
    bundle = assemble_evidence_bundle([a, b], max_tokens=100)
    assert len(bundle.items) == 1
    assert bundle.excluded[0].reason is ExclusionReason.DUPLICATE_RANGE


def test_token_limit_truncates() -> None:
    items = [
        _item(
            token_count=3,
            text=f"text-{i}",
            evidence_id=f"e{i}",
            node_id=NodeId(uuid4()),
            start_offset=0,
            end_offset=10,
        )
        for i in range(5)
    ]
    bundle = assemble_evidence_bundle(items, max_tokens=6)
    assert bundle.total_tokens <= 6
    assert bundle.truncated
    assert not bundle.complete


def test_mixed_revision_excluded() -> None:
    node = NodeId(uuid4())
    a = _item(node_id=node, content_hash="v1", text="one", token_count=1)
    b = _item(
        node_id=node, content_hash="v2", text="two", token_count=1, start_offset=10, end_offset=20
    )
    bundle = assemble_evidence_bundle([a, b], max_tokens=100)
    assert len(bundle.items) == 1
    assert any(e.reason is ExclusionReason.MIXED_REVISION for e in bundle.excluded)
