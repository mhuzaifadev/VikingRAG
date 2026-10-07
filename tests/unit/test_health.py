from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from vikingrag.api.app import CorrelationMiddleware, create_app
from vikingrag.api.routers import health_router
from vikingrag.infrastructure.object_store.local import LocalObjectStore
from vikingrag.settings.config import Settings


@asynccontextmanager
async def _noop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield


def _build_test_app(settings: Settings, tmp_path: Path, *, db_ok: bool, redis_ok: bool) -> FastAPI:
    """App with health routes but no real external connections."""

    app = FastAPI(lifespan=_noop_lifespan)
    app.state.settings = settings
    app.state.object_store = LocalObjectStore(tmp_path / "objects")

    database = AsyncMock()
    if db_ok:
        database.ping = AsyncMock(return_value=None)
    else:
        database.ping = AsyncMock(side_effect=RuntimeError("db down"))
    app.state.database = database

    redis = AsyncMock()
    if redis_ok:
        redis.ping = AsyncMock(return_value=True)
    else:
        redis.ping = AsyncMock(side_effect=RuntimeError("redis down"))
    app.state.redis = redis

    app.add_middleware(CorrelationMiddleware)
    app.include_router(health_router)
    return app


@pytest.mark.asyncio
async def test_liveness(settings: Settings, tmp_path: Path) -> None:
    app = _build_test_app(settings, tmp_path, db_ok=True, redis_ok=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/health/live")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == settings.app.name
    assert "x-request-id" in response.headers


@pytest.mark.asyncio
async def test_readiness_ok(settings: Settings, tmp_path: Path) -> None:
    app = _build_test_app(settings, tmp_path, db_ok=True, redis_ok=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/v1/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    names = {c["name"] for c in body["checks"]}
    assert {"postgres", "redis", "object_store"} <= names


@pytest.mark.asyncio
async def test_readiness_fails_when_dependency_down(settings: Settings, tmp_path: Path) -> None:
    app = _build_test_app(settings, tmp_path, db_ok=False, redis_ok=True)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "error"
    postgres = next(c for c in body["checks"] if c["name"] == "postgres")
    assert postgres["status"] == "error"


@pytest.mark.asyncio
async def test_create_app_factory(settings: Settings) -> None:
    app = create_app(settings)
    assert app.title == "VikingRAG"
    # Ensure factory does not connect at import/construction time
    assert not hasattr(app.state, "database") or getattr(app.state, "database", None) is None
