"""Vector repository contract - concrete pgvector implementation in a later phase."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable
from uuid import UUID


@runtime_checkable
class VectorRepository(Protocol):
    async def upsert(
        self,
        *,
        object_id: UUID,
        embedding: list[float],
        model: str,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...

    async def search(
        self,
        embedding: list[float],
        *,
        top_k: int = 8,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]: ...

    async def delete(self, object_id: UUID) -> None: ...
