"""Anthropic Messages API adapter (native tool_use) via httpx — no SDK import."""

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

_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicLLMProvider:
    """LLM adapter for Anthropic Messages API with tool_use support."""

    def __init__(
        self,
        *,
        api_key: str | None,
        base_url: str = "https://api.anthropic.com",
        model: str = "claude-sonnet-4-20250514",
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

        system, anthropic_messages = _to_anthropic_messages(messages)
        payload: dict[str, Any] = {
            "model": model or self._model,
            "messages": anthropic_messages,
            "max_tokens": max_tokens if max_tokens is not None else 4096,
            "temperature": (self._default_temperature if temperature is None else temperature),
        }
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = [_tool_to_anthropic(t) for t in tools]
            if tool_choice is not None:
                payload["tool_choice"] = _map_tool_choice(tool_choice)
        # response_schema: nudge via system note (Anthropic has no json_schema mode here)
        if response_schema is not None:
            schema_note = "Respond with JSON matching this schema: " + json.dumps(
                response_schema, separators=(",", ":")
            )
            payload["system"] = (
                f"{payload['system']}\n\n{schema_note}" if payload.get("system") else schema_note
            )

        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(
                    "/v1/messages",
                    json=payload,
                    headers=headers,
                )
                if response.status_code >= 500 and attempt < self._max_retries:
                    await asyncio.sleep(0.2 * (2**attempt))
                    continue
                if response.status_code >= 400:
                    raise ProviderError(f"LLM provider returned HTTP {response.status_code}")
                data = response.json()
                return _parse_anthropic_response(
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


def _tool_to_anthropic(tool: ToolDefinition) -> dict[str, Any]:
    return {
        "name": tool.function.name,
        "description": tool.function.description,
        "input_schema": tool.function.parameters or {"type": "object", "properties": {}},
    }


def _map_tool_choice(tool_choice: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(tool_choice, str):
        if tool_choice in {"auto", "any", "none"}:
            return {"type": tool_choice}
        if tool_choice == "required":
            return {"type": "any"}
        return {"type": "auto"}
    # OpenAI-style {"type": "function", "function": {"name": "..."}}
    if tool_choice.get("type") == "function":
        name = (tool_choice.get("function") or {}).get("name")
        if name:
            return {"type": "tool", "name": str(name)}
    return {"type": "auto"}


def _to_anthropic_messages(
    messages: list[ChatMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    out: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def _flush_tool_results() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            out.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for msg in messages:
        if msg.role == "system":
            if msg.content:
                system_parts.append(msg.content)
            continue
        if msg.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id,
                    "content": msg.content or "",
                }
            )
            continue
        _flush_tool_results()
        if msg.role == "user":
            out.append({"role": "user", "content": msg.content or ""})
            continue
        if msg.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if msg.content:
                blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tc.id,
                        "name": tc.name,
                        "input": tc.arguments if tc.arguments_valid else {},
                    }
                )
            if not blocks:
                blocks.append({"type": "text", "text": ""})
            out.append({"role": "assistant", "content": blocks})
            continue
        # Unknown roles treated as user text
        out.append({"role": "user", "content": msg.content or ""})

    _flush_tool_results()
    system = "\n\n".join(system_parts) if system_parts else None
    return system, out


def _parse_anthropic_response(
    data: dict[str, Any],
    *,
    default_model: str,
    latency_ms: float,
) -> LLMResponse:
    content_blocks = data.get("content") or []
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(str(block.get("text") or ""))
        elif btype == "tool_use":
            raw_input = block.get("input")
            if isinstance(raw_input, dict):
                arguments = raw_input
                arguments_raw = json.dumps(raw_input, separators=(",", ":"))
                valid = True
            else:
                arguments, valid = {}, False
                arguments_raw = str(raw_input) if raw_input is not None else "{}"
            tool_calls.append(
                ToolCall(
                    id=str(block.get("id") or "tool_use"),
                    name=str(block.get("name") or "unknown"),
                    arguments=arguments,
                    arguments_raw=arguments_raw,
                    arguments_valid=valid,
                )
            )

    stop = str(data.get("stop_reason") or "").lower()
    finish = _map_stop_reason(stop, has_tool_calls=bool(tool_calls))
    usage_raw = data.get("usage") or {}
    usage = TokenUsage(
        input_tokens=_as_optional_int(usage_raw.get("input_tokens")),
        output_tokens=_as_optional_int(usage_raw.get("output_tokens")),
        total_tokens=None,
    )
    if usage.input_tokens is not None and usage.output_tokens is not None:
        usage = TokenUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            total_tokens=usage.input_tokens + usage.output_tokens,
        )
    content = "".join(text_parts) if text_parts else None
    return LLMResponse(
        content=content,
        model=str(data.get("model") or default_model),
        finish_reason=finish,
        tool_calls=tuple(tool_calls),
        usage=usage,
        raw={"latency_ms": latency_ms, "provider": "anthropic"},
    )


def _map_stop_reason(raw: str, *, has_tool_calls: bool) -> FinishReason:
    mapping = {
        "end_turn": FinishReason.STOP,
        "stop_sequence": FinishReason.STOP,
        "tool_use": FinishReason.TOOL_CALLS,
        "max_tokens": FinishReason.LENGTH,
        "refusal": FinishReason.CONTENT_FILTER,
    }
    if raw in mapping:
        return mapping[raw]
    if has_tool_calls:
        return FinishReason.TOOL_CALLS
    return FinishReason.UNKNOWN if raw else FinishReason.STOP


def _as_optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
