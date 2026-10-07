"""Integration fixtures using Docker Compose services or env overrides.

When ``VIKINGRAG_DATABASE_URL`` is set (CI / local), prefer Alembic migrations
over ``Base.metadata.create_all`` so schema matches production. Set
``VIKINGRAG_INTEGRATION_REQUIRE_SERVICES=1`` to fail (not skip) if Postgres/Redis
are unavailable.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from sqlalchemy import text

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


def _require_services() -> bool:
    return os.environ.get("VIKINGRAG_INTEGRATION_REQUIRE_SERVICES", "").lower() in {
        "1",
        "true",
        "yes",
    }


@pytest.fixture(scope="session")
def integration_settings() -> Settings:
    clear_settings_cache()
    os.environ.setdefault("VIKINGRAG_APP_ENV", "test")
    os.environ.setdefault("VIKINGRAG_DATABASE_URL", _database_url())
    os.environ.setdefault("VIKINGRAG_REDIS_URL", _redis_url())
    return Settings()


def _alembic_upgrade() -> None:
    from alembic import command
    from alembic.config import Config

    root = Path(__file__).resolve().parents[2]
    cfg = Config(str(root / "alembic.ini"))
    clear_settings_cache()
    command.upgrade(cfg, "head")


@pytest.fixture(scope="session")
def migrated_schema(integration_settings: Settings) -> None:
    """Apply Alembic once per session when DATABASE_URL is configured."""
    del integration_settings
    if not os.environ.get("VIKINGRAG_DATABASE_URL"):
        return
    try:
        _alembic_upgrade()
    except Exception as exc:
        if _require_services():
            raise RuntimeError(f"Alembic upgrade failed: {exc}") from exc
        pytest.skip(f"Alembic upgrade failed: {exc}")


@pytest_asyncio.fixture
async def database(
    integration_settings: Settings, migrated_schema: None
) -> AsyncIterator[Database]:
    del migrated_schema
    db_settings = DatabaseSettings(url=_database_url())
    database = create_database(db_settings)
    try:
        await database.ping()
    except Exception as exc:
        await database.dispose()
        if _require_services():
            raise RuntimeError(f"PostgreSQL required but unavailable: {exc}") from exc
        pytest.skip(f"PostgreSQL unavailable: {exc}")

    # Fallback only when alembic was not run (no DATABASE_URL in env originally)
    if not os.environ.get("VIKINGRAG_DATABASE_URL"):
        async with database.engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await conn.run_sync(Base.metadata.create_all)

    try:
        yield database
    finally:
        async with database.engine.begin() as conn:
            await conn.execute(text("TRUNCATE TABLE documents CASCADE"))
        await database.dispose()


@pytest_asyncio.fixture
async def redis_client(integration_settings: Settings) -> AsyncIterator[RedisClient]:
    del integration_settings
    client = create_redis_client(RedisSettings(url=_redis_url()))
    try:
        await client.ping()
    except Exception as exc:
        await client.close()
        if _require_services():
            raise RuntimeError(f"Redis required but unavailable: {exc}") from exc
        pytest.skip(f"Redis unavailable: {exc}")
    try:
        yield client
    finally:
        await client.close()
