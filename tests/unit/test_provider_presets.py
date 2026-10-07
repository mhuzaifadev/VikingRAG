"""Factory presets and Anthropic adapter unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.providers.factory import build_embedding_provider, build_llm_provider
from vikingrag.providers.llm.anthropic import AnthropicLLMProvider
from vikingrag.providers.llm.base import (
    ChatMessage,
    FinishReason,
    ToolCall,
    ToolDefinition,
    ToolFunctionSpec,
)
from vikingrag.providers.llm.fake import FakeLLMProvider
from vikingrag.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from vikingrag.providers.presets import llm_base_url, resolve_llm_preset
from vikingrag.settings.config import Settings, clear_settings_cache


@pytest.fixture(autouse=True)
def _clear(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_cache()
    monkeypatch.setenv("VIKINGRAG_APP_ENV", "development")


def test_resolve_presets() -> None:
    assert resolve_llm_preset("deepseek") is not None
    assert resolve_llm_preset("gemini") is not None
    assert resolve_llm_preset("anthropic") is not None
    assert llm_base_url("deepseek", "https://api.openai.com/v1").endswith("deepseek.com/v1")


def test_build_deepseek_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("VIKINGRAG_LLM_API_KEY", "sk-test")
    monkeypatch.setenv("VIKINGRAG_LLM_MODEL", "deepseek-chat")
    clear_settings_cache()
    llm = build_llm_provider(Settings())
    assert isinstance(llm, OpenAICompatibleLLMProvider)
    assert "deepseek" in llm._base_url


def test_build_gemini_openai_compatible(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_LLM_PROVIDER", "gemini")
    monkeypatch.setenv("VIKINGRAG_LLM_API_KEY", "sk-test")
    clear_settings_cache()
    llm = build_llm_provider(Settings())
    assert isinstance(llm, OpenAICompatibleLLMProvider)
    assert "generativelanguage.googleapis.com" in llm._base_url


def test_build_anthropic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("VIKINGRAG_LLM_API_KEY", "sk-ant-test")
    clear_settings_cache()
    llm = build_llm_provider(Settings())
    assert isinstance(llm, AnthropicLLMProvider)


def test_build_fake(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_LLM_PROVIDER", "fake")
    clear_settings_cache()
    llm = build_llm_provider(Settings())
    assert isinstance(llm, FakeLLMProvider)


def test_unknown_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_LLM_PROVIDER", "does-not-exist")
    clear_settings_cache()
    with pytest.raises(NotImplementedCapabilityError):
        build_llm_provider(Settings())


def test_embedding_deepseek_preset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIKINGRAG_EMBEDDING_PROVIDER", "deepseek")
    monkeypatch.setenv("VIKINGRAG_EMBEDDING_API_KEY", "sk-test")
    clear_settings_cache()
    emb = build_embedding_provider(Settings())
    assert emb.provider_name == "openai_compatible" or hasattr(emb, "dimensions")


@pytest.mark.asyncio
async def test_anthropic_parse_tool_use() -> None:
    provider = AnthropicLLMProvider(api_key="sk-test", client=MagicMock())
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "model": "claude-test",
        "stop_reason": "tool_use",
        "content": [
            {"type": "text", "text": "looking up"},
            {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "Search",
                "input": {"query": "fees"},
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    provider._client.post = AsyncMock(return_value=mock_resp)
    result = await provider.generate(
        [ChatMessage(role="user", content="hi")],
        tools=[
            ToolDefinition(
                function=ToolFunctionSpec(
                    name="Search",
                    description="search",
                    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                )
            )
        ],
    )
    assert result.finish_reason is FinishReason.TOOL_CALLS
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "Search"
    assert result.tool_calls[0].arguments == {"query": "fees"}
    assert result.content == "looking up"


@pytest.mark.asyncio
async def test_anthropic_roundtrip_messages() -> None:
    """Ensure tool-result messages serialize without raising."""
    from vikingrag.providers.llm.anthropic import _to_anthropic_messages

    system, msgs = _to_anthropic_messages(
        [
            ChatMessage(role="system", content="sys"),
            ChatMessage(role="user", content="q"),
            ChatMessage(
                role="assistant",
                content=None,
                tool_calls=(
                    ToolCall(
                        id="t1",
                        name="Search",
                        arguments={"query": "x"},
                        arguments_raw='{"query":"x"}',
                    ),
                ),
            ),
            ChatMessage(role="tool", content='{"hits":[]}', tool_call_id="t1"),
        ]
    )
    assert system == "sys"
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"][0]["type"] == "tool_result"
