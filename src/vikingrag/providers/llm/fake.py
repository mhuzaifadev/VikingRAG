"""Fake / deterministic LLM and summary generators for offline tests."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from vikingrag.domain.models.representation import SummaryRequest, SummaryResult
from vikingrag.providers.llm.base import (
    ChatMessage,
    FinishReason,
    LLMResponse,
    TokenUsage,
    ToolCall,
    ToolDefinition,
)


class FakeLLMProvider:
    """Returns deterministic content or a scripted sequence of tool/answer steps."""

    def __init__(
        self,
        *,
        model: str = "fake-llm-v1",
        script: Sequence[LLMResponse] | None = None,
    ) -> None:
        self._model = model
        self._script = list(script) if script is not None else None
        self._script_index = 0
        self.calls: list[dict[str, Any]] = []

    @property
    def model(self) -> str:
        return self._model

    async def aclose(self) -> None:
        return None

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
        self.calls.append(
            {
                "messages": list(messages),
                "tools": list(tools) if tools else None,
                "tool_choice": tool_choice,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_schema": response_schema,
            }
        )
        if self._script is not None:
            if self._script_index >= len(self._script):
                # Exhausted script: stop with empty answer
                return LLMResponse(
                    content="",
                    model=model or self._model,
                    finish_reason=FinishReason.STOP,
                    usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
                    input_tokens=1,
                    output_tokens=1,
                )
            response = self._script[self._script_index]
            self._script_index += 1
            return response

        last = ""
        for msg in reversed(messages):
            if msg.content:
                last = msg.content
                break
        content = f"SUMMARY: {last[:400]}"
        return LLMResponse(
            content=content,
            model=model or self._model,
            finish_reason=FinishReason.STOP,
            usage=TokenUsage(
                input_tokens=max(1, len(last.split()) if last else 1),
                output_tokens=max(1, len(content.split())),
            ),
            input_tokens=max(1, len(last.split()) if last else 1),
            output_tokens=max(1, len(content.split())),
        )


def scripted_tool_call(
    *,
    call_id: str,
    name: str,
    arguments: dict[str, Any],
    model: str = "fake-llm-v1",
) -> LLMResponse:
    """Helper to build a tool-calling LLMResponse for FakeLLMProvider scripts."""
    import json

    raw = json.dumps(arguments, separators=(",", ":"))
    return LLMResponse(
        content=None,
        model=model,
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id=call_id,
                name=name,
                arguments=arguments,
                arguments_raw=raw,
                arguments_valid=True,
            ),
        ),
        usage=TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        input_tokens=10,
        output_tokens=5,
    )


def scripted_final_answer(
    content: str,
    *,
    model: str = "fake-llm-v1",
) -> LLMResponse:
    return LLMResponse(
        content=content,
        model=model,
        finish_reason=FinishReason.STOP,
        usage=TokenUsage(
            input_tokens=20,
            output_tokens=max(1, len(content.split())),
            total_tokens=20 + max(1, len(content.split())),
        ),
        input_tokens=20,
        output_tokens=max(1, len(content.split())),
    )


class FakeSummaryGenerator:
    """Bottom-up friendly summarizer that needs no network."""

    def __init__(self, *, model: str = "fake-summary-v1", version: str = "1") -> None:
        self._model = model
        self._version = version
        self.calls: list[SummaryRequest] = []

    @property
    def name(self) -> str:
        return "fake_summary"

    @property
    def model(self) -> str:
        return self._model

    @property
    def version(self) -> str:
        return self._version

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        self.calls.append(request)
        started = time.perf_counter()
        parts: list[str] = []
        if request.title:
            parts.append(f"Title: {request.title}")
        if request.own_content and request.own_content.strip():
            parts.append(request.own_content.strip()[:500])
        for child in request.child_summaries:
            if child.strip():
                parts.append(f"- {child.strip()[:240]}")
        text = " | ".join(parts) if parts else f"Empty {request.node_type.value}"
        if len(text) > 800:
            text = text[:797] + "..."
        latency_ms = (time.perf_counter() - started) * 1000.0
        return SummaryResult(
            text=text,
            model=self._model,
            input_tokens=max(1, len(" ".join(parts).split()) if parts else 1),
            output_tokens=max(1, len(text.split())),
            latency_ms=latency_ms,
        )


class LLMSummaryGenerator:
    """SummaryGenerator backed by an LLMProvider."""

    def __init__(
        self,
        llm: Any,
        *,
        name: str = "llm_summary",
        model: str,
        version: str = "1",
        temperature: float = 0.2,
    ) -> None:
        self._llm = llm
        self._name = name
        self._model = model
        self._version = version
        self._temperature = temperature

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    @property
    def version(self) -> str:
        return self._version

    async def summarize(self, request: SummaryRequest) -> SummaryResult:
        started = time.perf_counter()
        prompt = _build_summary_prompt(request)
        response = await self._llm.generate(
            [
                ChatMessage(
                    role="system",
                    content=(
                        "You write compact hierarchical abstracts for retrieval. "
                        "Preserve important terms and identifiers. Do not invent facts."
                    ),
                ),
                ChatMessage(role="user", content=prompt),
            ],
            model=self._model,
            temperature=self._temperature,
            max_tokens=request.max_output_tokens,
        )
        text = (response.content or "").strip()
        return SummaryResult(
            text=text,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )


def _build_summary_prompt(request: SummaryRequest) -> str:
    lines = [
        f"Node type: {request.node_type.value}",
        f"Title: {request.title or '(none)'}",
        "Own content:",
        (request.own_content or "(none)")[: max(0, request.max_input_tokens * 4)],
        "Child summaries:",
    ]
    if request.child_summaries:
        for child in request.child_summaries:
            lines.append(f"- {child}")
    else:
        lines.append("(none)")
    lines.append("Write a concise abstract for retrieval indexing.")
    return "\n".join(lines)
