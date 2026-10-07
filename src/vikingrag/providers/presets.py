"""Named provider presets — default base URLs without leaking vendor SDKs."""

from __future__ import annotations

from dataclasses import dataclass

# Gemini OpenAI-compatible Chat Completions endpoint (API key as Bearer).
_GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"


@dataclass(frozen=True, slots=True)
class ProviderPreset:
    """Resolved defaults for a named LLM / embedding provider."""

    name: str
    adapter: str  # openai_compatible | anthropic | deterministic | fake
    default_base_url: str | None = None
    default_model: str | None = None
    default_embedding_model: str | None = None


_LLM_PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        name="openai",
        adapter="openai_compatible",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
    ),
    "openai_compatible": ProviderPreset(
        name="openai_compatible",
        adapter="openai_compatible",
        default_base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
    ),
    "vllm": ProviderPreset(
        name="vllm",
        adapter="openai_compatible",
        default_base_url="http://localhost:8000/v1",
        default_model="default",
    ),
    "deepseek": ProviderPreset(
        name="deepseek",
        adapter="openai_compatible",
        default_base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
    ),
    "gemini": ProviderPreset(
        name="gemini",
        adapter="openai_compatible",
        default_base_url=_GEMINI_OPENAI_BASE,
        default_model="gemini-2.0-flash",
    ),
    "anthropic": ProviderPreset(
        name="anthropic",
        adapter="anthropic",
        default_base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-20250514",
    ),
    "fake": ProviderPreset(name="fake", adapter="fake", default_model="fake-llm-v1"),
    "test": ProviderPreset(name="test", adapter="fake", default_model="fake-llm-v1"),
}

_EMBEDDING_PRESETS: dict[str, ProviderPreset] = {
    "openai": ProviderPreset(
        name="openai",
        adapter="openai_compatible",
        default_base_url="https://api.openai.com/v1",
        default_embedding_model="text-embedding-3-small",
    ),
    "openai_compatible": ProviderPreset(
        name="openai_compatible",
        adapter="openai_compatible",
        default_base_url="https://api.openai.com/v1",
        default_embedding_model="text-embedding-3-small",
    ),
    "vllm": ProviderPreset(
        name="vllm",
        adapter="openai_compatible",
        default_base_url="http://localhost:8000/v1",
        default_embedding_model="default",
    ),
    "deepseek": ProviderPreset(
        name="deepseek",
        adapter="openai_compatible",
        default_base_url="https://api.deepseek.com/v1",
        default_embedding_model="deepseek-embedding",
    ),
    "gemini": ProviderPreset(
        name="gemini",
        adapter="openai_compatible",
        default_base_url=_GEMINI_OPENAI_BASE,
        default_embedding_model="text-embedding-004",
    ),
    "deterministic": ProviderPreset(
        name="deterministic",
        adapter="deterministic",
        default_embedding_model="deterministic-hash-v1",
    ),
    "fake": ProviderPreset(
        name="fake",
        adapter="deterministic",
        default_embedding_model="deterministic-hash-v1",
    ),
    "test": ProviderPreset(
        name="test",
        adapter="deterministic",
        default_embedding_model="deterministic-hash-v1",
    ),
}


def resolve_llm_preset(provider: str) -> ProviderPreset | None:
    key = provider.lower().strip()
    return _LLM_PRESETS.get(key)


def resolve_embedding_preset(provider: str) -> ProviderPreset | None:
    key = provider.lower().strip()
    return _EMBEDDING_PRESETS.get(key)


def llm_base_url(provider: str, configured: str) -> str:
    """Prefer explicit non-default configured URL; else preset default."""
    preset = resolve_llm_preset(provider)
    default_openai = "https://api.openai.com/v1"
    name = provider.lower().strip()
    if preset and preset.default_base_url:
        use_preset = (not configured or configured.rstrip("/") == default_openai.rstrip("/")) and (
            name not in {"openai", "openai_compatible"}
        )
        if use_preset:
            return preset.default_base_url
        if configured:
            return configured.rstrip("/")
        return preset.default_base_url
    return (configured or default_openai).rstrip("/")


def embedding_base_url(provider: str, configured: str) -> str:
    preset = resolve_embedding_preset(provider)
    default_openai = "https://api.openai.com/v1"
    name = provider.lower().strip()
    if preset and preset.default_base_url:
        use_preset = (not configured or configured.rstrip("/") == default_openai.rstrip("/")) and (
            name not in {"openai", "openai_compatible"}
        )
        if use_preset:
            return preset.default_base_url
        if configured:
            return configured.rstrip("/")
        return preset.default_base_url
    return (configured or default_openai).rstrip("/")
