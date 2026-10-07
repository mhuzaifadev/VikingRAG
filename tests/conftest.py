"""Shared fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from vikingrag.settings.config import Settings, clear_settings_cache


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[Settings]:
    monkeypatch.setenv("VIKINGRAG_APP_ENV", "test")
    monkeypatch.setenv("VIKINGRAG_APP_NAME", "vikingrag-test")
    monkeypatch.setenv(
        "VIKINGRAG_DATABASE_URL",
        os.environ.get(
            "VIKINGRAG_DATABASE_URL",
            "postgresql+asyncpg://vikingrag:vikingrag@localhost:5432/vikingrag",
        ),
    )
    monkeypatch.setenv(
        "VIKINGRAG_REDIS_URL",
        os.environ.get("VIKINGRAG_REDIS_URL", "redis://localhost:6379/0"),
    )
    clear_settings_cache()
    yield Settings()
    clear_settings_cache()
