from __future__ import annotations

import pytest

from vikingrag.application.budget import BudgetLimits
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


def test_production_rejects_fake_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_APP_ENV", "production")
    monkeypatch.setenv("VIKINGRAG_LLM_PROVIDER", "fake")
    monkeypatch.setenv("VIKINGRAG_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("VIKINGRAG_RETRIEVAL_ASSESSOR_PROVIDER", "openai")
    clear_settings_cache()
    with pytest.raises(Exception, match="production forbids"):
        Settings()


def test_paper_profile_applies_k_l_b_gamma(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_PAPER_ENABLED", "true")
    clear_settings_cache()
    settings = Settings()
    assert settings.paper.top_k == 10
    assert settings.paper.chunk_token_upper_bound == 1000
    assert settings.paper.agent_round_budget == 15
    assert settings.paper.activation_gamma == 0.8
    assert settings.retrieval.initial_top_k == 10
    assert settings.retrieval.max_rounds == 15
    assert settings.retrieval.max_read_tokens_per_call == 1000
    assert settings.retrieval.experience_gamma == 0.8
    limits = BudgetLimits.paper_profile(
        agent_round_budget=settings.paper.agent_round_budget,
        top_k=settings.paper.top_k,
        chunk_token_upper_bound=settings.paper.chunk_token_upper_bound,
        activation_gamma=settings.paper.activation_gamma,
    )
    assert limits.max_tool_calls >= 60
    assert limits.max_llm_calls >= 20
