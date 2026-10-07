"""LLM provider contract - no vendor SDK imports."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict[str, Any] | None = None


@runtime_checkable
class LLMProvider(Protocol):
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> LLMResponse: ...
