"""Local filesystem object store for development and tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from vikingrag.domain.errors import InfrastructureError, NotFoundError


class LocalObjectStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, key: str) -> Path:
        if not key or key.startswith("/") or ".." in Path(key).parts:
            raise InfrastructureError(f"Invalid object key: {key!r}")
        path = (self._root / key).resolve()
        if not str(path).startswith(str(self._root)):
            raise InfrastructureError(f"Object key escapes store root: {key!r}")
        return path

    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        del content_type  # retained for S3-compatible signature
        path = self._resolve(key)

        def _write() -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        await asyncio.to_thread(_write)

    async def get(self, key: str) -> bytes:
        path = self._resolve(key)

        def _read() -> bytes:
            if not path.is_file():
                raise NotFoundError(f"Object not found: {key}")
            return path.read_bytes()

        return await asyncio.to_thread(_read)

    async def exists(self, key: str) -> bool:
        path = self._resolve(key)
        return await asyncio.to_thread(path.is_file)

    async def delete(self, key: str) -> None:
        path = self._resolve(key)

        def _delete() -> None:
            if path.is_file():
                path.unlink()

        await asyncio.to_thread(_delete)
