"""Expansion charges text tokens and honors RetrievalContext deadlines."""

from __future__ import annotations

import asyncio
import time

import pytest

from vikingrag.application.budget import RetrievalContext
from vikingrag.application.experience.expand import (
    estimate_payload_tokens,
    expand_experience_edges,
)
from vikingrag.application.experience.memory_store import InMemoryExperienceStore
from vikingrag.domain.errors import BudgetExhaustedError
from vikingrag.domain.models.experience import (
    ExperienceEdge,
    ExperienceExpansionLimits,
    ExperiencePayload,
    new_experience_edge_id,
    new_experience_payload_id,
    new_query_run_id,
)
from vikingrag.domain.models.representation import EmbeddingIdentity


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(provider="deterministic", model="test", dimensions=4, version="1")


@pytest.mark.asyncio
async def test_max_tokens_truncates_by_text_not_payload_count() -> None:
    store = InMemoryExperienceStore()
    identity = _identity()
    qvec = [1.0, 0.0, 0.0, 0.0]
    # Large query_text so one payload exceeds a tiny budget.
    big = "late fee policy " * 80
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text=big,
        query_embedding=qvec,
        embedding_identity=identity,
        trace_summary="r0:search",
    )
    await store.create_payload(payload)
    await store.create_edges(
        [
            ExperienceEdge(
                id=new_experience_edge_id(),
                payload_id=payload.id,
                source_uri="viking://a",
                target_uri="viking://b",
            )
        ]
    )
    cost = estimate_payload_tokens(payload)
    assert cost > 1
    result = await expand_experience_edges(
        seed_uris=["viking://a"],
        query_embedding=qvec,
        identity=identity,
        edges=store,
        limits=ExperienceExpansionLimits(
            gamma=0.5,
            max_hops=2,
            max_nodes=10,
            max_edges=20,
            max_tokens=max(1, cost - 1),
        ),
    )
    assert result.truncated is True
    assert result.truncation_reason == "max_tokens"
    assert result.edges_activated == 0


class _SlowEdges:
    def __init__(self, inner: InMemoryExperienceStore) -> None:
        self._inner = inner

    async def count_active(self) -> int:
        return await self._inner.count_active()

    async def find_active_by_source(self, source_uri: str, *, identity: EmbeddingIdentity) -> list:
        await asyncio.sleep(2.0)
        return await self._inner.find_active_by_source(source_uri, identity=identity)


@pytest.mark.asyncio
async def test_edge_lookup_respects_retrieval_context_deadline() -> None:
    store = InMemoryExperienceStore()
    identity = _identity()
    qvec = [1.0, 0.0, 0.0, 0.0]
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text="fee",
        query_embedding=qvec,
        embedding_identity=identity,
        trace_summary="t",
    )
    await store.create_payload(payload)
    await store.create_edges(
        [
            ExperienceEdge(
                id=new_experience_edge_id(),
                payload_id=payload.id,
                source_uri="viking://a",
                target_uri="viking://b",
            )
        ]
    )
    ctx = RetrievalContext.create()
    ctx.deadline_monotonic = time.monotonic() + 0.05
    result = await expand_experience_edges(
        seed_uris=["viking://a"],
        query_embedding=qvec,
        identity=identity,
        edges=_SlowEdges(store),  # type: ignore[arg-type]
        limits=ExperienceExpansionLimits(gamma=0.5, max_wall_time_ms=60_000),
        ctx=ctx,
    )
    assert result.truncated is True
    assert result.truncation_reason == "deadline"


@pytest.mark.asyncio
async def test_await_with_deadline_raises_budget_exhausted() -> None:
    ctx = RetrievalContext.create()
    ctx.deadline_monotonic = time.monotonic() - 1.0

    async def _work() -> None:
        await asyncio.sleep(0.01)

    coro = _work()
    try:
        with pytest.raises(BudgetExhaustedError):
            await ctx.await_with_deadline(coro)
    finally:
        coro.close()
