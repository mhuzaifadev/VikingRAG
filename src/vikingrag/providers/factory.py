"""Build configured providers without leaking SDK types into application code."""

from __future__ import annotations

from typing import Any

from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.domain.summary import SummaryGenerator
from vikingrag.providers.embeddings.base import EmbeddingProvider
from vikingrag.providers.embeddings.deterministic import DeterministicEmbeddingProvider
from vikingrag.providers.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from vikingrag.providers.llm.anthropic import AnthropicLLMProvider
from vikingrag.providers.llm.fake import FakeLLMProvider, FakeSummaryGenerator, LLMSummaryGenerator
from vikingrag.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from vikingrag.providers.presets import (
    embedding_base_url,
    llm_base_url,
    resolve_embedding_preset,
    resolve_llm_preset,
)
from vikingrag.providers.reranking.base import RerankerProvider
from vikingrag.providers.reranking.noop import NoOpReranker
from vikingrag.settings.config import Settings


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    name = settings.embedding.provider.lower().strip()
    if name in {"unimplemented", "", "none"}:
        raise NotImplementedCapabilityError("embedding_provider")
    preset = resolve_embedding_preset(name)
    if preset is None:
        raise NotImplementedCapabilityError(f"embedding_provider:{name}")
    if preset.adapter == "deterministic":
        return DeterministicEmbeddingProvider(
            dimensions=settings.embedding.dimensions,
            model=settings.embedding.model
            or preset.default_embedding_model
            or "deterministic-hash-v1",
            provider_name="deterministic",
        )
    if preset.adapter == "openai_compatible":
        return OpenAICompatibleEmbeddingProvider(
            api_key=settings.embedding.api_key,
            base_url=embedding_base_url(name, settings.embedding.base_url),
            model=settings.embedding.model
            or preset.default_embedding_model
            or "text-embedding-3-small",
            dimensions=settings.embedding.dimensions,
            timeout_seconds=settings.embedding.timeout_seconds,
            max_retries=settings.embedding.max_retries,
            batch_size=settings.embedding.batch_size,
        )
    raise NotImplementedCapabilityError(f"embedding_provider:{name}")


def build_llm_provider(settings: Settings) -> Any:
    name = settings.llm.provider.lower().strip()
    if name in {"unimplemented", "", "none"}:
        raise NotImplementedCapabilityError("llm_provider")
    preset = resolve_llm_preset(name)
    if preset is None:
        raise NotImplementedCapabilityError(f"llm_provider:{name}")
    if preset.adapter == "fake":
        return FakeLLMProvider(model=settings.llm.model or preset.default_model or "fake-llm-v1")
    model = settings.llm.model or preset.default_model or "gpt-4o-mini"
    base = llm_base_url(name, settings.llm.base_url)
    if preset.adapter == "anthropic":
        return AnthropicLLMProvider(
            api_key=settings.llm.api_key,
            base_url=base,
            model=model,
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
            default_temperature=settings.llm.temperature,
        )
    if preset.adapter == "openai_compatible":
        return OpenAICompatibleLLMProvider(
            api_key=settings.llm.api_key,
            base_url=base,
            model=model,
            timeout_seconds=settings.llm.timeout_seconds,
            max_retries=settings.llm.max_retries,
            default_temperature=settings.llm.temperature,
        )
    raise NotImplementedCapabilityError(f"llm_provider:{name}")


def build_summary_generator(settings: Settings) -> SummaryGenerator:
    name = settings.llm.provider.lower().strip()
    if name in {"fake", "test", "deterministic"}:
        return FakeSummaryGenerator(
            model=settings.llm.model or "fake-summary-v1",
            version=settings.indexing.summary_version,
        )
    llm = build_llm_provider(settings)
    return LLMSummaryGenerator(
        llm,
        model=settings.llm.model,
        version=settings.indexing.summary_version,
        temperature=settings.llm.temperature,
    )


def build_reranker(settings: Settings) -> RerankerProvider:
    del settings
    return NoOpReranker()
