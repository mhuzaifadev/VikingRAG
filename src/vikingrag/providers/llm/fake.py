"""Fake / deterministic LLM and summary generators for offline tests."""

from __future__ import annotations

import time
from typing import Any

from vikingrag.domain.models.representation import SummaryRequest, SummaryResult
from vikingrag.providers.llm.base import ChatMessage, LLMResponse


class FakeLLMProvider:
    """Returns a deterministic string derived from the last user message."""

    def __init__(self, *, model: str = "fake-llm-v1") -> None:
        self._model = model
        self.calls: list[list[ChatMessage]] = []

    @property
    def model(self) -> str:
        return self._model

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse:
        del temperature, max_tokens, response_schema
        self.calls.append(list(messages))
        last = messages[-1].content if messages else ""
        content = f"SUMMARY: {last[:400]}"
        return LLMResponse(
            content=content,
            model=model or self._model,
            input_tokens=max(1, len(last.split())),
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
        # Keep within a compact budget
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
        return SummaryResult(
            text=response.content.strip(),
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
