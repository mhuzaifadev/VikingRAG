"""Reranker provider contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RerankHit:
    index: int
    score: float
    text: str


@runtime_checkable
class RerankerProvider(Protocol):
    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankHit]: ...
