"""Embedding identity and deterministic provider tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.domain.errors import EmbeddingIdentityMismatch
from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.representation import (
    EmbeddingId,
    EmbeddingIdentity,
    NodeEmbedding,
    RepresentationId,
)
from vikingrag.providers.embeddings.deterministic import DeterministicEmbeddingProvider


def test_embedding_identity_rejects_bad_dimensions() -> None:
    with pytest.raises(ValueError):
        EmbeddingIdentity(provider="x", model="y", dimensions=0)


def test_embedding_identity_compatibility() -> None:
    a = EmbeddingIdentity(provider="deterministic", model="m", dimensions=32, version="1")
    b = EmbeddingIdentity(provider="deterministic", model="m", dimensions=32, version="1")
    c = EmbeddingIdentity(provider="deterministic", model="m", dimensions=64, version="1")
    assert a.compatible_with(b)
    assert not a.compatible_with(c)


def test_node_embedding_dimension_guard() -> None:
    identity = EmbeddingIdentity(provider="p", model="m", dimensions=4)
    with pytest.raises(ValueError):
        NodeEmbedding(
            id=EmbeddingId(uuid4()),
            representation_id=RepresentationId(uuid4()),
            node_id=NodeId(uuid4()),
            document_id=DocumentId(uuid4()),
            identity=identity,
            embedding=[0.1, 0.2],
        )


@pytest.mark.asyncio
async def test_deterministic_embeddings_are_stable_and_similar() -> None:
    provider = DeterministicEmbeddingProvider(dimensions=64)
    a = await provider.embed_text("Evidence Verification claim mapping")
    b = await provider.embed_text("Evidence Verification claim mapping")
    c = await provider.embed_text("totally unrelated zebra astronomy")
    assert a.vectors[0] == b.vectors[0]

    def cosine(x: list[float], y: list[float]) -> float:
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert cosine(a.vectors[0], b.vectors[0]) > 0.99
    assert cosine(a.vectors[0], c.vectors[0]) < cosine(a.vectors[0], b.vectors[0])


def test_embedding_identity_mismatch_error_code() -> None:
    err = EmbeddingIdentityMismatch("nope")
    assert err.code == "embedding_identity_mismatch"
