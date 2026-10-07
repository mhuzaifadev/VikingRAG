"""OpenAI-compatible embeddings via httpx."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from vikingrag.domain.errors import ProviderError
from vikingrag.providers.embeddings.base import EmbeddingResult, EmbeddingUsage


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        batch_size: int = 32,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimensions = dimensions
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._batch_size = batch_size
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    @property
    def provider_name(self) -> str:
        return "openai_compatible"

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def embed_text(self, text: str, *, model: str | None = None) -> EmbeddingResult:
        return await self.embed_batch([text], model=model)

    async def embed_batch(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        return await self.embed(texts, model=model)

    async def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(
                vectors=[], model=model or self._model, dimensions=self._dimensions
            )
        if not self._api_key:
            raise ProviderError("Embedding API key is not configured")

        all_vectors: list[list[float]] = []
        total_tokens = 0
        started = time.perf_counter()
        resolved_model = model or self._model

        for offset in range(0, len(texts), self._batch_size):
            batch = texts[offset : offset + self._batch_size]
            result = await self._embed_once(batch, model=resolved_model)
            all_vectors.extend(result.vectors)
            if result.usage and result.usage.total_tokens:
                total_tokens += result.usage.total_tokens

        return EmbeddingResult(
            vectors=all_vectors,
            model=resolved_model,
            dimensions=self._dimensions,
            usage=EmbeddingUsage(
                total_tokens=total_tokens or None, prompt_tokens=total_tokens or None
            ),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    async def _embed_once(self, texts: list[str], *, model: str) -> EmbeddingResult:
        payload: dict[str, Any] = {"model": model, "input": texts}
        # Only send dimensions when the API supports it (OpenAI text-embedding-3-*)
        if self._dimensions > 0:
            payload["dimensions"] = self._dimensions

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post("/embeddings", json=payload, headers=headers)
                if response.status_code >= 500 and attempt < self._max_retries:
                    await asyncio.sleep(0.2 * (2**attempt))
                    continue
                if response.status_code >= 400:
                    raise ProviderError(f"Embedding provider returned HTTP {response.status_code}")
                data = response.json()
                items = sorted(data.get("data") or [], key=lambda x: int(x.get("index", 0)))
                vectors = [list(item["embedding"]) for item in items]
                for vector in vectors:
                    if len(vector) != self._dimensions:
                        raise ProviderError(
                            f"Embedding dimension mismatch: expected {self._dimensions}, "
                            f"got {len(vector)}"
                        )
                usage = data.get("usage") or {}
                return EmbeddingResult(
                    vectors=vectors,
                    model=str(data.get("model") or model),
                    dimensions=self._dimensions,
                    usage=EmbeddingUsage(
                        prompt_tokens=_as_optional_int(usage.get("prompt_tokens")),
                        total_tokens=_as_optional_int(usage.get("total_tokens")),
                    ),
                )
            except ProviderError:
                raise
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise ProviderError("Embedding provider timed out") from exc
                await asyncio.sleep(0.2 * (2**attempt))
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise ProviderError("Embedding provider request failed") from exc
                await asyncio.sleep(0.2 * (2**attempt))
        raise ProviderError(f"Embedding provider failed after retries: {last_error}")


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
