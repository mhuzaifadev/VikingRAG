"""Shared retrieval budget accounting - immutable limits, mutable usage ledger."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

from vikingrag.domain.errors import BudgetExhaustedError, DomainError, ScopeDeniedError
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.retrieval import RetrievalBudget


class CancellationError(DomainError):
    def __init__(self, message: str = "Retrieval cancelled") -> None:
        super().__init__(message, code="cancelled")


@dataclass(frozen=True, slots=True)
class BudgetLimits:
    max_tool_calls: int = 12
    max_read_tokens: int = 12_000
    max_wall_time_ms: int = 12_000
    max_embedding_calls: int = 20
    max_vector_searches: int = 20
    max_llm_calls: int = 4
    max_nodes_inspected: int = 200
    max_db_operations: int = 200
    max_estimated_cost_usd: float | None = None

    @classmethod
    def from_retrieval_budget(cls, budget: RetrievalBudget) -> BudgetLimits:
        return cls(
            max_tool_calls=budget.max_tool_calls,
            max_read_tokens=budget.max_read_tokens,
            max_wall_time_ms=budget.max_wall_time_ms,
            max_embedding_calls=budget.max_embedding_calls,
            max_vector_searches=budget.max_vector_searches,
            max_estimated_cost_usd=budget.max_estimated_cost_usd,
        )


@dataclass(slots=True)
class UsageLedger:
    tool_calls: int = 0
    embedding_calls: int = 0
    vector_searches: int = 0
    db_operations: int = 0
    nodes_inspected: int = 0
    read_tokens: int = 0
    evidence_tokens_retained: int = 0
    llm_calls: int = 0
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    provider_attempts: int = 0
    estimated_cost_usd: float | None = None
    cost_known: bool = False

    def snapshot(self) -> dict[str, Any]:
        return {
            "tool_calls": self.tool_calls,
            "embedding_calls": self.embedding_calls,
            "vector_searches": self.vector_searches,
            "db_operations": self.db_operations,
            "nodes_inspected": self.nodes_inspected,
            "read_tokens": self.read_tokens,
            "evidence_tokens_retained": self.evidence_tokens_retained,
            "llm_calls": self.llm_calls,
            "llm_input_tokens": self.llm_input_tokens,
            "llm_output_tokens": self.llm_output_tokens,
            "provider_attempts": self.provider_attempts,
            "estimated_cost_usd": self.estimated_cost_usd,
            "cost_known": self.cost_known,
        }


@dataclass(slots=True)
class RetrievalContext:
    """Per-request shared execution context for all primitives + assessor."""

    query_id: UUID = field(default_factory=uuid4)
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    permitted_document_ids: frozenset[DocumentId] | None = None
    limits: BudgetLimits = field(default_factory=BudgetLimits)
    usage: UsageLedger = field(default_factory=UsageLedger)
    deadline_monotonic: float = field(default_factory=lambda: time.monotonic() + 12.0)
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    _query_embedding_cache: dict[tuple[str, str], list[float]] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        limits: BudgetLimits | None = None,
        permitted_document_ids: frozenset[DocumentId] | None = None,
        query_id: UUID | None = None,
        trace_id: str | None = None,
        budget: RetrievalBudget | None = None,
    ) -> RetrievalContext:
        resolved = limits
        if resolved is None and budget is not None:
            resolved = BudgetLimits.from_retrieval_budget(budget)
        if resolved is None:
            resolved = BudgetLimits()
        ctx = cls(
            query_id=query_id or uuid4(),
            trace_id=trace_id or str(uuid4()),
            permitted_document_ids=permitted_document_ids,
            limits=resolved,
            deadline_monotonic=time.monotonic() + (resolved.max_wall_time_ms / 1000.0),
        )
        return ctx

    def check_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise CancellationError()

    def check_deadline(self) -> None:
        self.check_cancelled()
        if time.monotonic() >= self.deadline_monotonic:
            raise BudgetExhaustedError("wall_time")

    def remaining_ms(self) -> float:
        return max(0.0, (self.deadline_monotonic - time.monotonic()) * 1000.0)

    def document_allowed(self, document_id: DocumentId) -> bool:
        if self.permitted_document_ids is None:
            return True
        return document_id in self.permitted_document_ids

    def narrow_document_ids(
        self, requested: tuple[DocumentId, ...] | list[DocumentId]
    ) -> tuple[DocumentId, ...]:
        """Client may narrow scope, never widen past permitted set."""
        if not requested:
            if self.permitted_document_ids is None:
                return ()
            return tuple(self.permitted_document_ids)
        if self.permitted_document_ids is None:
            return tuple(requested)
        narrowed = tuple(d for d in requested if d in self.permitted_document_ids)
        if not narrowed:
            raise ScopeDeniedError("Requested documents are outside permitted scope")
        return narrowed

    def cache_query_embedding(self, key: tuple[str, str], vector: list[float]) -> None:
        self._query_embedding_cache[key] = vector

    def get_cached_query_embedding(self, key: tuple[str, str]) -> list[float] | None:
        return self._query_embedding_cache.get(key)

    async def reserve(self, **amounts: int) -> None:
        """Atomically reserve budget capacity before an external/db call."""
        async with self._lock:
            self.check_deadline()
            for name, amount in amounts.items():
                if amount < 0:
                    raise ValueError(f"reserve amount for {name} must be >= 0")
                current = getattr(self.usage, name)
                limit_name = f"max_{name}"
                if not hasattr(self.limits, limit_name):
                    # fields without max_* still tracked but not capped here
                    setattr(self.usage, name, current + amount)
                    continue
                limit = getattr(self.limits, limit_name)
                if current + amount > limit:
                    raise BudgetExhaustedError(name)
                setattr(self.usage, name, current + amount)

    async def add_usage(self, **amounts: int) -> None:
        """Reconcile additional usage that does not need pre-reservation."""
        async with self._lock:
            for name, amount in amounts.items():
                if amount == 0:
                    continue
                if amount < 0:
                    raise ValueError("usage increments must be >= 0")
                setattr(self.usage, name, getattr(self.usage, name) + amount)

    @asynccontextmanager
    async def tool_call(self, name: str = "tool") -> AsyncIterator[None]:
        del name
        await self.reserve(tool_calls=1)
        self.check_deadline()
        yield
