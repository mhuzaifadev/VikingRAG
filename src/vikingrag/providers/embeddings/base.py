"""Embedding provider contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class EmbeddingUsage:
    prompt_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    vectors: list[list[float]]
    model: str
    dimensions: int
    usage: EmbeddingUsage | None = None
    latency_ms: float | None = None


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def default_model(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult: ...

    async def embed_text(self, text: str, *, model: str | None = None) -> EmbeddingResult: ...

    async def embed_batch(
        self, texts: list[str], *, model: str | None = None
    ) -> EmbeddingResult: ...
