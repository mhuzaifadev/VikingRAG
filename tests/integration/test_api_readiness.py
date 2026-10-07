from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from vikingrag.api.app import create_app
from vikingrag.infrastructure.cache.redis import RedisClient
from vikingrag.infrastructure.database.engine import Database
from vikingrag.settings.config import Settings

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_readiness_with_live_dependencies(
    integration_settings: Settings,
    database: Database,
    redis_client: RedisClient,
    tmp_path: Path,
) -> None:
    settings = integration_settings
    settings.object_store.local_root = str(tmp_path / "objects")
    app = create_app(settings)

    # Manually bind resources as lifespan would, without relying on TestClient lifespan quirks
    app.state.database = database
    app.state.redis = redis_client
    from vikingrag.infrastructure.object_store.local import LocalObjectStore

    app.state.object_store = LocalObjectStore(settings.object_store.local_root)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        live = await client.get("/v1/health/live")
        ready = await client.get("/v1/health/ready")

    assert live.status_code == 200
    assert ready.status_code == 200
    assert ready.json()["status"] == "ok"
