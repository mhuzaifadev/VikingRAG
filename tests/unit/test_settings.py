from __future__ import annotations

import pytest

from vikingrag.settings.config import Settings, clear_settings_cache


def test_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIKINGRAG_DATABASE_URL", raising=False)
    monkeypatch.delenv("VIKINGRAG_APP_NAME", raising=False)
    clear_settings_cache()
    settings = Settings()
    assert settings.app.name == "vikingrag"
    assert settings.database.url.startswith("postgresql+asyncpg://")
    assert settings.redis.url.startswith("redis://")
    assert settings.object_store.backend.value == "local"
    assert settings.llm.provider == "unimplemented"
    assert settings.retrieval.initial_top_k == 8


def test_settings_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_APP_NAME", "custom-name")
    monkeypatch.setenv("VIKINGRAG_RETRIEVAL_MAX_ROUNDS", "3")
    monkeypatch.setenv("VIKINGRAG_EMBEDDING_DIMENSIONS", "768")
    clear_settings_cache()
    settings = Settings()
    assert settings.app.name == "custom-name"
    assert settings.retrieval.max_rounds == 3
    assert settings.embedding.dimensions == 768


def test_retrieval_settings_reject_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_RETRIEVAL_INITIAL_TOP_K", "0")
    clear_settings_cache()
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError
        Settings()
