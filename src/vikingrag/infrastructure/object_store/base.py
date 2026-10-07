"""Object store contract - S3/MinIO adapters will implement the same Protocol."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ObjectStore(Protocol):
    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> None: ...

    async def get(self, key: str) -> bytes: ...

    async def exists(self, key: str) -> bool: ...

    async def delete(self, key: str) -> None: ...
