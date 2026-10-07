"""Build configured providers without leaking SDK types into application code."""

from __future__ import annotations

from typing import Any

from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.domain.summary import SummaryGenerator
from vikingrag.providers.embeddings.base import EmbeddingProvider
from vikingrag.providers.embeddings.deterministic import DeterministicEmbeddingProvider
from vikingrag.providers.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from vikingrag.providers.llm.fake import FakeLLMProvider, FakeSummaryGenerator, LLMSummaryGenerator
from vikingrag.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from vikingrag.providers.reranking.base import RerankerProvider
from vikingrag.providers.reranking.noop import NoOpReranker
from vikingrag.settings.config import Settings


def build_embedding_provider(settings: Settings) -> EmbeddingProvider:
    name = settings.embedding.provider.lower().strip()
    if name in {"unimplemented", "", "none"}:
        raise NotImplementedCapabilityError("embedding_provider")
    if name in {"deterministic", "fake", "test"}:
        return DeterministicEmbeddingProvider(
            dimensions=settings.embedding.dimensions,
            model=settings.embedding.model or "deterministic-hash-v1",
            provider_name="deterministic",
        )
    if name in {"openai", "openai_compatible", "vllm"}:
        return OpenAICompatibleEmbeddingProvider(
            api_key=settings.embedding.api_key,
            base_url=settings.embedding.base_url,
            model=settings.embedding.model,
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
    if name in {"fake", "test"}:
        return FakeLLMProvider(model=settings.llm.model or "fake-llm-v1")
    if name in {"openai", "openai_compatible", "vllm"}:
        return OpenAICompatibleLLMProvider(
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
            model=settings.llm.model,
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
