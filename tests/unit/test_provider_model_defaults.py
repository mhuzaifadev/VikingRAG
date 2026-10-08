"""Resolved LLM models must follow provider presets when model env is empty."""

from __future__ import annotations

import os

import pytest

from vikingrag.providers.factory import build_llm_provider
from vikingrag.providers.presets import resolve_llm_preset
from vikingrag.settings.config import Settings, clear_settings_cache


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_cache()
    for key in list(os.environ):
        if key.startswith("VIKINGRAG_LLM_"):
            monkeypatch.delenv(key, raising=False)
    yield
    clear_settings_cache()


def test_gemini_preset_default_model_not_gpt() -> None:
    preset = resolve_llm_preset("gemini")
    assert preset is not None
    assert preset.default_model is not None
    assert "gpt" not in preset.default_model.lower()
    assert "gemini" in preset.default_model.lower()


def test_factory_uses_deepseek_preset_when_model_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("VIKINGRAG_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("VIKINGRAG_LLM_MODEL", "")
    clear_settings_cache()
    settings = Settings()
    assert settings.llm.model == ""
    llm = build_llm_provider(settings)
    assert getattr(llm, "_model", None) == "deepseek-chat" or "deepseek" in str(
        getattr(llm, "model", "")
    )


def test_parse_allowlist_star_is_unrestricted() -> None:
    from vikingrag.settings.config import parse_allowed_document_ids

    assert parse_allowed_document_ids("*") is None
    assert parse_allowed_document_ids("all") is None
    assert parse_allowed_document_ids("") == frozenset()
    assert parse_allowed_document_ids(None) is None


def test_migrate_role_skips_provider_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_APP_ENV", "production")
    monkeypatch.setenv("VIKINGRAG_APP_PROCESS_ROLE", "migrate")
    monkeypatch.setenv("VIKINGRAG_DATABASE_URL", "postgresql+asyncpg://u:p@localhost/db")
    clear_settings_cache()
    settings = Settings()
    assert settings.app.process_role == "migrate"
