"""Shared budget reservation and deadline tests."""

from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest

from vikingrag.application.budget import BudgetLimits, CancellationError, RetrievalContext
from vikingrag.domain.errors import BudgetExhaustedError
from vikingrag.domain.models.document import DocumentId


@pytest.mark.asyncio
async def test_reserve_accumulates_and_exhausts() -> None:
    ctx = RetrievalContext.create(limits=BudgetLimits(max_embedding_calls=2, max_tool_calls=5))
    await ctx.reserve(embedding_calls=1)
    await ctx.reserve(embedding_calls=1)
    with pytest.raises(BudgetExhaustedError):
        await ctx.reserve(embedding_calls=1)
    assert ctx.usage.embedding_calls == 2


@pytest.mark.asyncio
async def test_concurrent_reservations_do_not_overspend() -> None:
    ctx = RetrievalContext.create(limits=BudgetLimits(max_tool_calls=10))

    async def once() -> None:
        await ctx.reserve(tool_calls=1)

    await asyncio.gather(*[once() for _ in range(10)])
    with pytest.raises(BudgetExhaustedError):
        await ctx.reserve(tool_calls=1)


@pytest.mark.asyncio
async def test_cancellation() -> None:
    ctx = RetrievalContext.create()
    ctx.cancel_event.set()
    with pytest.raises(CancellationError):
        ctx.check_cancelled()


def test_scope_narrowing() -> None:
    a, b, c = DocumentId(uuid4()), DocumentId(uuid4()), DocumentId(uuid4())
    ctx = RetrievalContext.create(permitted_document_ids=frozenset({a, b}))
    assert ctx.narrow_document_ids([a, c]) == (a,)
