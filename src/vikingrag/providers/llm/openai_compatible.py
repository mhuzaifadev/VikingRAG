"""OpenAI-compatible chat completions via httpx (OpenAI, vLLM, local gateways)."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import httpx

from vikingrag.domain.errors import ProviderError
from vikingrag.providers.llm.base import (
    ChatMessage,
    FinishReason,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


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
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse:
        if not self._api_key:
            raise ProviderError("LLM API key is not configured")

        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": [_message_to_openai(m) for m in messages],
            "temperature": (self._default_temperature if temperature is None else temperature),
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": response_schema,
            }
        if tools:
            payload["tools"] = [_tool_to_openai(t) for t in tools]
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

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
                return _parse_openai_response(
                    data,
                    default_model=str(payload["model"]),
                    latency_ms=(time.perf_counter() - started) * 1000.0,
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


def _tool_to_openai(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.function.name,
            "description": tool.function.description,
            "parameters": tool.function.parameters,
        },
    }


def _message_to_openai(message: ChatMessage) -> dict[str, Any]:
    payload: dict[str, Any] = {"role": message.role}
    if message.content is not None:
        payload["content"] = message.content
    elif message.role == "assistant" and message.tool_calls:
        # OpenAI allows null/omitted content when tool_calls are present
        payload["content"] = None
    else:
        payload["content"] = message.content or ""

    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": tc.arguments_raw
                    if tc.arguments_raw
                    else json.dumps(tc.arguments, separators=(",", ":")),
                },
            }
            for tc in message.tool_calls
        ]
    if message.tool_call_id is not None:
        payload["tool_call_id"] = message.tool_call_id
    if message.name is not None:
        payload["name"] = message.name
    return payload


def _parse_openai_response(
    data: dict[str, Any],
    *,
    default_model: str,
    latency_ms: float,
) -> LLMResponse:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    raw_content = message.get("content")
    content: str | None = None if raw_content is None else str(raw_content)

    tool_calls = tuple(_parse_tool_call(tc) for tc in (message.get("tool_calls") or []))
    finish = _map_finish_reason(choice.get("finish_reason"), has_tool_calls=bool(tool_calls))
    usage_raw = data.get("usage") or {}
    usage = TokenUsage(
        input_tokens=_as_optional_int(usage_raw.get("prompt_tokens")),
        output_tokens=_as_optional_int(usage_raw.get("completion_tokens")),
        total_tokens=_as_optional_int(usage_raw.get("total_tokens")),
    )
    return LLMResponse(
        content=content,
        model=str(data.get("model") or default_model),
        finish_reason=finish,
        tool_calls=tool_calls,
        usage=usage,
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        raw={"latency_ms": latency_ms, "provider": "openai_compatible"},
    )


def _parse_tool_call(raw: Any) -> ToolCall:
    if not isinstance(raw, dict):
        return ToolCall(
            id="invalid",
            name="unknown",
            arguments={},
            arguments_raw="",
            arguments_valid=False,
        )
    fn = raw.get("function") or {}
    name = str(fn.get("name") or "unknown")
    call_id = str(raw.get("id") or f"call_{name}")
    arguments_raw = fn.get("arguments")
    if arguments_raw is None:
        arguments_raw = "{}"
    elif not isinstance(arguments_raw, str):
        try:
            arguments_raw = json.dumps(arguments_raw)
        except (TypeError, ValueError):
            arguments_raw = str(arguments_raw)
    arguments, valid = _parse_arguments(arguments_raw)
    return ToolCall(
        id=call_id,
        name=name,
        arguments=arguments,
        arguments_raw=arguments_raw,
        arguments_valid=valid,
    )


def _parse_arguments(raw: str) -> tuple[dict[str, Any], bool]:
    text = raw.strip() if raw else "{}"
    if not text:
        return {}, True
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {}, False
    if parsed is None:
        return {}, True
    if isinstance(parsed, dict):
        return parsed, True
    return {}, False


def _map_finish_reason(raw: Any, *, has_tool_calls: bool) -> FinishReason:
    if raw is None:
        return FinishReason.TOOL_CALLS if has_tool_calls else FinishReason.STOP
    value = str(raw).lower()
    mapping = {
        "stop": FinishReason.STOP,
        "tool_calls": FinishReason.TOOL_CALLS,
        "function_call": FinishReason.TOOL_CALLS,
        "length": FinishReason.LENGTH,
        "content_filter": FinishReason.CONTENT_FILTER,
    }
    if value in mapping:
        return mapping[value]
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    return FinishReason.UNKNOWN


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
