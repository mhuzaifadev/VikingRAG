"""Integration fixtures using Docker Compose services or env overrides."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from vikingrag.infrastructure.cache.redis import RedisClient, create_redis_client
from vikingrag.infrastructure.database.engine import Database, create_database
from vikingrag.infrastructure.database.models import Base
from vikingrag.settings.config import (
    DatabaseSettings,
    RedisSettings,
    Settings,
    clear_settings_cache,
)


def _database_url() -> str:
    return os.environ.get(
        "VIKINGRAG_DATABASE_URL",
        "postgresql+asyncpg://vikingrag:vikingrag@localhost:5432/vikingrag",
    )


def _redis_url() -> str:
    return os.environ.get("VIKINGRAG_REDIS_URL", "redis://localhost:6379/0")


@pytest.fixture(scope="session")
def integration_settings() -> Settings:
    clear_settings_cache()
    os.environ.setdefault("VIKINGRAG_APP_ENV", "test")
    os.environ.setdefault("VIKINGRAG_DATABASE_URL", _database_url())
    os.environ.setdefault("VIKINGRAG_REDIS_URL", _redis_url())
    return Settings()


async def _ensure_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        await conn.run_sync(Base.metadata.create_all)


@pytest_asyncio.fixture
async def database(integration_settings: Settings) -> AsyncIterator[Database]:
    db_settings = DatabaseSettings(url=_database_url())
    database = create_database(db_settings)
    try:
        await database.ping()
    except Exception as exc:
        await database.dispose()
        pytest.skip(f"PostgreSQL unavailable: {exc}")
    await _ensure_schema(database.engine)
    try:
        yield database
    finally:
        # Clean documents table between tests
        async with database.engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE documents CASCADE"))
        await database.dispose()


@pytest_asyncio.fixture
async def redis_client(integration_settings: Settings) -> AsyncIterator[RedisClient]:
    client = create_redis_client(RedisSettings(url=_redis_url()))
    try:
        await client.ping()
    except Exception as exc:
        await client.close()
        pytest.skip(f"Redis unavailable: {exc}")
    try:
        yield client
    finally:
        await client.close()
