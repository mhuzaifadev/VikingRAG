from __future__ import annotations

from pathlib import Path

import pytest

from vikingrag.domain.errors import InfrastructureError, NotFoundError
from vikingrag.infrastructure.object_store.local import LocalObjectStore


@pytest.mark.asyncio
async def test_local_object_store_roundtrip(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    await store.put("docs/a.txt", b"hello", content_type="text/plain")
    assert await store.exists("docs/a.txt")
    assert await store.get("docs/a.txt") == b"hello"
    await store.delete("docs/a.txt")
    assert not await store.exists("docs/a.txt")


@pytest.mark.asyncio
async def test_local_object_store_missing(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    with pytest.raises(NotFoundError):
        await store.get("missing.bin")


@pytest.mark.asyncio
async def test_local_object_store_rejects_path_traversal(tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path)
    with pytest.raises(InfrastructureError):
        await store.put("../escape.txt", b"nope")
