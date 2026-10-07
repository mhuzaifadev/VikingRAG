"""Embedding provider contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    dimensions: int


@runtime_checkable
class EmbeddingProvider(Protocol):
    async def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult: ...
