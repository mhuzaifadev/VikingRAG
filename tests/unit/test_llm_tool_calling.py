"""HTTPX mock tests for OpenAI-compatible tool_calls round-trip."""

from __future__ import annotations

import json

import httpx
import pytest

from vikingrag.domain.errors import ProviderError
from vikingrag.providers.llm.base import (
    ChatMessage,
    FinishReason,
    ToolCall,
    ToolDefinition,
    ToolFunctionSpec,
)
from vikingrag.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from vikingrag.providers.llm.tools import SEARCH_TOOL, retrieval_tool_definitions


def _tool_defs() -> list[ToolDefinition]:
    return retrieval_tool_definitions()


@pytest.mark.asyncio
async def test_multi_tool_calls_round_trip() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        captured["payload"] = body
        return httpx.Response(
            200,
            json={
                "model": "gpt-test",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_search",
                                    "type": "function",
                                    "function": {
                                        "name": "Search",
                                        "arguments": '{"query":"budget","top_k":5}',
                                    },
                                },
                                {
                                    "id": "call_list",
                                    "type": "function",
                                    "function": {
                                        "name": "List",
                                        "arguments": '{"uri":"viking://doc/1"}',
                                    },
                                },
                            ],
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 40,
                    "completion_tokens": 20,
                    "total_tokens": 60,
                },
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://example.test/v1")
    provider = OpenAICompatibleLLMProvider(api_key="secret", client=client, max_retries=0)

    # Prior assistant+tool messages must serialize vendor-side
    history = [
        ChatMessage(role="system", content="retrieve"),
        ChatMessage(role="user", content="What is the budget?"),
        ChatMessage(
            role="assistant",
            content=None,
            tool_calls=(
                ToolCall(
                    id="prev",
                    name="Search",
                    arguments={"query": "x"},
                    arguments_raw='{"query":"x"}',
                ),
            ),
        ),
        ChatMessage(
            role="tool",
            content='{"hits":[]}',
            tool_call_id="prev",
            name="Search",
        ),
    ]
    response = await provider.generate(history, tools=_tool_defs())
    await provider.aclose()

    assert response.finish_reason is FinishReason.TOOL_CALLS
    assert response.content is None
    assert len(response.tool_calls) == 2
    assert response.tool_calls[0].name == "Search"
    assert response.tool_calls[0].arguments["query"] == "budget"
    assert response.tool_calls[1].name == "List"
    assert response.usage is not None
    assert response.usage.input_tokens == 40
    assert response.input_tokens == 40

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert len(payload["tools"]) == len(_tool_defs())
    roles = [m["role"] for m in payload["messages"]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert payload["messages"][2]["content"] is None
    assert payload["messages"][2]["tool_calls"][0]["function"]["name"] == "Search"
    assert payload["messages"][3]["tool_call_id"] == "prev"


@pytest.mark.asyncio
async def test_empty_content_with_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-test",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c1",
                                    "type": "function",
                                    "function": {
                                        "name": "Read",
                                        "arguments": '{"uri":"viking://n/1"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://example.test/v1")
    provider = OpenAICompatibleLLMProvider(api_key="secret", client=client, max_retries=0)
    response = await provider.generate(
        [ChatMessage(role="user", content="read it")],
        tools=[SEARCH_TOOL],
    )
    await provider.aclose()
    assert response.content == ""
    assert response.tool_calls[0].name == "Read"
    assert response.finish_reason is FinishReason.TOOL_CALLS


@pytest.mark.asyncio
async def test_malformed_tool_arguments() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "gpt-test",
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "bad",
                                    "type": "function",
                                    "function": {
                                        "name": "Grep",
                                        "arguments": "{not-json",
                                    },
                                }
                            ],
                        },
                    }
                ],
                "usage": {},
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://example.test/v1")
    provider = OpenAICompatibleLLMProvider(api_key="secret", client=client, max_retries=0)
    response = await provider.generate(
        [ChatMessage(role="user", content="find X")],
        tools=[
            ToolDefinition(
                function=ToolFunctionSpec(
                    name="Grep",
                    description="grep",
                    parameters={"type": "object"},
                )
            )
        ],
    )
    await provider.aclose()
    assert response.tool_calls[0].arguments_valid is False
    assert response.tool_calls[0].arguments == {}
    assert response.tool_calls[0].arguments_raw == "{not-json"


@pytest.mark.asyncio
async def test_provider_failure_on_tool_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport, base_url="https://example.test/v1")
    provider = OpenAICompatibleLLMProvider(api_key="secret", client=client, max_retries=0)
    with pytest.raises(ProviderError, match="HTTP 503"):
        await provider.generate(
            [ChatMessage(role="user", content="hi")],
            tools=_tool_defs(),
        )
    await provider.aclose()
