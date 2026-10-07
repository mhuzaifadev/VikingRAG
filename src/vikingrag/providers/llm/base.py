"""LLM provider contract - no vendor SDK imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable


class FinishReason(StrEnum):
    STOP = "stop"
    TOOL_CALLS = "tool_calls"
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ToolFunctionSpec:
    """JSON-schema function description for tool calling."""

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ToolFunctionSpec.name must be non-empty")


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """Provider-neutral tool definition (OpenAI-style function tool)."""

    function: ToolFunctionSpec
    type: str = "function"

    def __post_init__(self) -> None:
        if self.type != "function":
            raise ValueError("Only function tools are supported")


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One model-requested tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    arguments_raw: str = "{}"
    arguments_valid: bool = True

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("ToolCall.id must be non-empty")
        if not self.name.strip():
            raise ValueError("ToolCall.name must be non-empty")


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """Chat / tool-loop message.

    Assistant messages may carry ``tool_calls``. Tool-result messages use
    ``role="tool"`` with ``tool_call_id``.
    """

    role: str
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None
    name: str | None = None

    def __post_init__(self) -> None:
        if not self.role.strip():
            raise ValueError("ChatMessage.role must be non-empty")
        if self.role == "tool" and not self.tool_call_id:
            raise ValueError("tool messages require tool_call_id")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class LLMResponse:
    content: str | None
    model: str
    finish_reason: FinishReason = FinishReason.STOP
    tool_calls: tuple[ToolCall, ...] = ()
    usage: TokenUsage | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    raw: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        # Keep legacy token fields aligned with structured usage when only one is set.
        object.__setattr__(
            self,
            "input_tokens",
            self.input_tokens
            if self.input_tokens is not None
            else (self.usage.input_tokens if self.usage else None),
        )
        object.__setattr__(
            self,
            "output_tokens",
            self.output_tokens
            if self.output_tokens is not None
            else (self.usage.output_tokens if self.usage else None),
        )


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
        tools: list[ToolDefinition] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
    ) -> LLMResponse: ...
