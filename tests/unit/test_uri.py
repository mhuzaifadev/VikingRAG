from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.domain.errors import InvalidVikingURI
from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.uri import VikingURI, VikingURIKind, VikingURIParser


def test_build_and_parse_document_uri() -> None:
    doc = DocumentId(uuid4())
    uri = VikingURIParser.build_document_uri(doc)
    parsed = VikingURIParser.parse(uri)
    assert parsed.kind is VikingURIKind.DOCUMENT
    assert parsed.document_id == doc
    assert parsed.node_id is None


def test_build_and_parse_node_uri() -> None:
    doc = DocumentId(uuid4())
    node = NodeId(uuid4())
    uri = VikingURIParser.build_node_uri(doc, node)
    parsed = VikingURIParser.parse(uri)
    assert parsed.kind is VikingURIKind.NODE
    assert parsed.document_id == doc
    assert parsed.node_id == node
    assert str(VikingURI(kind=VikingURIKind.NODE, document_id=doc, node_id=node)) == uri


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "https://documents/x",
        "viking://",
        "viking:///documents/" + str(uuid4()),
        "viking://documents/" + str(uuid4()) + "/",
        "viking://documents/not-a-uuid",
        "viking://documents/" + str(uuid4()) + "/nodes/not-a-uuid",
        "viking://documents/" + str(uuid4()) + "/../nodes/" + str(uuid4()),
        "viking://documents/" + str(uuid4()) + "//nodes/" + str(uuid4()),
        "viking://documents/" + str(uuid4()) + "/chunks/" + str(uuid4()),
        "viking://documents/" + str(uuid4()) + "?x=1",
        "viking://documents/" + str(uuid4()) + "#frag",
        "viking://tables/documents/" + str(uuid4()),
    ],
)
def test_malformed_uris(bad: str) -> None:
    with pytest.raises(InvalidVikingURI):
        VikingURIParser.parse(bad)
