"""Provider failure and timeout behavior with httpx mocks."""

from __future__ import annotations

import httpx
import pytest

from vikingrag.domain.errors import ProviderError
from vikingrag.providers.embeddings.openai_compatible import OpenAICompatibleEmbeddingProvider
from vikingrag.providers.llm.base import ChatMessage
from vikingrag.providers.llm.openai_compatible import OpenAICompatibleLLMProvider


@pytest.mark.asyncio
async def test_llm_requires_api_key() -> None:
    provider = OpenAICompatibleLLMProvider(api_key=None)
    with pytest.raises(ProviderError, match="API key"):
        await provider.generate([ChatMessage(role="user", content="hi")])


@pytest.mark.asyncio
async def test_llm_http_error_is_structured() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "nope"})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://example.test/v1")
    provider = OpenAICompatibleLLMProvider(api_key="secret", client=client, max_retries=0)
    with pytest.raises(ProviderError, match="HTTP 401"):
        await provider.generate([ChatMessage(role="user", content="hi")])
    await provider.aclose()


@pytest.mark.asyncio
async def test_embedding_success_batch() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "text-embedding-3-small",
                "data": [
                    {"index": 0, "embedding": [0.1] * 8},
                    {"index": 1, "embedding": [0.2] * 8},
                ],
                "usage": {"prompt_tokens": 4, "total_tokens": 4},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://example.test/v1")
    provider = OpenAICompatibleEmbeddingProvider(
        api_key="secret",
        client=client,
        dimensions=8,
        batch_size=10,
    )
    result = await provider.embed_batch(["a", "b"])
    assert len(result.vectors) == 2
    assert result.dimensions == 8
    await provider.aclose()


@pytest.mark.asyncio
async def test_embedding_timeout() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://example.test/v1")
    provider = OpenAICompatibleEmbeddingProvider(
        api_key="secret",
        client=client,
        dimensions=8,
        max_retries=0,
    )
    with pytest.raises(ProviderError, match="timed out"):
        await provider.embed_text("hello")
    await provider.aclose()
