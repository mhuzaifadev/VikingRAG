"""API-key auth and document allowlist semantics."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from vikingrag.api.app import CorrelationMiddleware, create_app
from vikingrag.api.auth import (
    derive_permitted_document_ids,
    parse_allowed_document_ids,
)
from vikingrag.api.errors import register_exception_handlers
from vikingrag.api.routers import documents_router, health_router, search_router
from vikingrag.domain.models.document import DocumentId
from vikingrag.infrastructure.object_store.local import LocalObjectStore
from vikingrag.settings.config import AuthSettings, Settings, clear_settings_cache


def test_parse_allowlist_semantics() -> None:
    assert parse_allowed_document_ids(None) is None
    assert parse_allowed_document_ids("") == frozenset()
    assert parse_allowed_document_ids("   ") == frozenset()
    a = uuid4()
    b = uuid4()
    parsed = parse_allowed_document_ids(f"{a}, {b}")
    assert parsed == frozenset({DocumentId(a), DocumentId(b)})


def test_derive_when_auth_disabled() -> None:
    auth = AuthSettings(enabled=False, allowed_document_ids="")
    assert derive_permitted_document_ids(auth) is None


def test_derive_when_auth_enabled_unrestricted() -> None:
    auth = AuthSettings(enabled=True, api_key="k", allowed_document_ids=None)
    assert derive_permitted_document_ids(auth) is None


def test_derive_when_auth_enabled_deny_all() -> None:
    auth = AuthSettings(enabled=True, api_key="k", allowed_document_ids="")
    assert derive_permitted_document_ids(auth) == frozenset()


def test_production_rejects_fake_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_APP_ENV", "production")
    monkeypatch.setenv("VIKINGRAG_LLM_PROVIDER", "fake")
    monkeypatch.setenv("VIKINGRAG_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VIKINGRAG_RETRIEVAL_ASSESSOR_PROVIDER", "openai_compatible")
    clear_settings_cache()
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        Settings()


def test_production_rejects_scripted_assessor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_APP_ENV", "production")
    monkeypatch.setenv("VIKINGRAG_LLM_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VIKINGRAG_EMBEDDING_PROVIDER", "openai_compatible")
    monkeypatch.setenv("VIKINGRAG_RETRIEVAL_ASSESSOR_PROVIDER", "scripted")
    clear_settings_cache()
    with pytest.raises(Exception):  # noqa: B017
        Settings()


@asynccontextmanager
async def _noop_lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield


def _auth_app(settings: Settings, tmp_path: Path) -> FastAPI:
    app = FastAPI(lifespan=_noop_lifespan)
    app.state.settings = settings
    app.state.object_store = LocalObjectStore(tmp_path / "objects")
    app.state.database = AsyncMock()
    app.state.redis = AsyncMock()
    app.state.embedding_provider = None
    app.state.llm_provider = None
    app.add_middleware(CorrelationMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(documents_router)
    app.include_router(search_router)
    return app


@pytest.mark.asyncio
async def test_health_open_when_auth_enabled(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("VIKINGRAG_APP_ENV", "test")
    monkeypatch.setenv("VIKINGRAG_AUTH_ENABLED", "true")
    monkeypatch.setenv("VIKINGRAG_AUTH_API_KEY", "secret-key")
    clear_settings_cache()
    settings = Settings()
    app = _auth_app(settings, tmp_path)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        live = await client.get("/health/live")
        assert live.status_code == 200
        denied = await client.post("/v1/search", json={"query": "hi"})
        assert denied.status_code == 401
        ok = await client.post(
            "/v1/search",
            json={"query": "hi", "document_ids": []},
            headers={"X-API-Key": "secret-key"},
        )
        # May 422/500 without DB — but must not be 401
        assert ok.status_code != 401


@pytest.mark.asyncio
async def test_create_app_still_builds_with_auth(settings: Settings) -> None:
    app = create_app(settings)
    assert app.title == "VikingRAG"
