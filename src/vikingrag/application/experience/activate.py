"""Gamma-gated experience-edge activation (Algorithm 3 gate)."""

from __future__ import annotations

import math

from vikingrag.domain.models.experience import ExperienceEdge, ExperiencePayload
from vikingrag.domain.models.representation import EmbeddingIdentity


def cosine_similarity(
    a: list[float] | tuple[float, ...], b: list[float] | tuple[float, ...]
) -> float:
    """Cosine similarity in [-1, 1]. Returns 0.0 for zero-norm vectors."""
    if len(a) != len(b):
        raise ValueError(f"vector length mismatch: {len(a)} vs {len(b)}")
    if not a:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def should_activate(
    *,
    current_query_embedding: list[float],
    current_identity: EmbeddingIdentity,
    payload: ExperiencePayload,
    edge: ExperienceEdge,
    gamma: float,
) -> tuple[bool, float]:
    """Return (activate?, query_similarity).

    Activation requires:
    1. Compatible embedding identity (never compare incompatible spaces).
    2. Cosine(current query, historical query) >= gamma.
    3. Edge is active (caller should filter; this is a safety check).
    """
    if not current_identity.compatible_with(payload.embedding_identity):
        return False, 0.0
    if len(current_query_embedding) != current_identity.dimensions:
        return False, 0.0
    if len(payload.query_embedding) != payload.embedding_identity.dimensions:
        return False, 0.0

    from vikingrag.domain.models.experience import ExperienceEdgeStatus

    if edge.status is not ExperienceEdgeStatus.ACTIVE:
        return False, 0.0

    sim = cosine_similarity(current_query_embedding, payload.query_embedding)
    return sim >= gamma, sim
