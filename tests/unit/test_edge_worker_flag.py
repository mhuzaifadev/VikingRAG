"""Worker settings: disabled in test env."""

from __future__ import annotations

import pytest

from vikingrag.settings.config import Settings, clear_settings_cache


def test_workers_disabled_in_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_APP_ENV", "test")
    monkeypatch.setenv("VIKINGRAG_WORKERS_EDGE_BUILDER_ENABLED", "true")
    clear_settings_cache()
    settings = Settings()
    assert settings.workers.edge_builder_enabled is False


def test_workers_default_enabled_in_development(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_APP_ENV", "development")
    monkeypatch.delenv("VIKINGRAG_WORKERS_EDGE_BUILDER_ENABLED", raising=False)
    clear_settings_cache()
    settings = Settings()
    assert settings.workers.edge_builder_enabled is True
