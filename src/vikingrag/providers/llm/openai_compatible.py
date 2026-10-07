"""OpenAI-compatible chat completions via httpx (OpenAI, vLLM, local gateways)."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from vikingrag.domain.errors import ProviderError
from vikingrag.providers.llm.base import ChatMessage, LLMResponse


class OpenAICompatibleLLMProvider:
    """LLM adapter that speaks the OpenAI Chat Completions API shape."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        default_temperature: float = 0.2,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._default_temperature = default_temperature
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(timeout_seconds),
        )

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self._api_key:
            raise ProviderError("LLM API key is not configured")

        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": (self._default_temperature if temperature is None else temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    "/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if response.status_code >= 500 and attempt < self._max_retries:
                    await asyncio.sleep(0.2 * (2**attempt))
                    continue
                if response.status_code >= 400:
                    raise ProviderError(f"LLM provider returned HTTP {response.status_code}")
                data = response.json()
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = str(message.get("content") or "")
                usage = data.get("usage") or {}
                return LLMResponse(
                    content=content,
                    model=str(data.get("model") or payload["model"]),
                    input_tokens=_as_optional_int(usage.get("prompt_tokens")),
                    output_tokens=_as_optional_int(usage.get("completion_tokens")),
                    raw={"latency_ms": (time.perf_counter() - started) * 1000.0},
                )
            except ProviderError:
                raise
            except httpx.TimeoutException as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise ProviderError("LLM provider timed out") from exc
                await asyncio.sleep(0.2 * (2**attempt))
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self._max_retries:
                    raise ProviderError("LLM provider request failed") from exc
                await asyncio.sleep(0.2 * (2**attempt))
        raise ProviderError(f"LLM provider failed after retries: {last_error}")


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
