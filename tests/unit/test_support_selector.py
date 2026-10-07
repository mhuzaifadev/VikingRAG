"""Alg 2 SUPPORT selectors: deterministic + LLM (fake)."""

from __future__ import annotations

import json

import pytest

from vikingrag.application.experience.support_select import (
    DeterministicSupportSelector,
    LLMSupportSelector,
    build_support_selector,
    select_support_uris,
)
from vikingrag.providers.llm.base import FinishReason, LLMResponse, TokenUsage
from vikingrag.providers.llm.fake import FakeLLMProvider


def test_deterministic_prefers_citations() -> None:
    cand = frozenset({"u1", "u2", "u3"})
    got = select_support_uris(cand, citations=["u2", "missing"], max_support=8)
    assert got == frozenset({"u2"})


@pytest.mark.asyncio
async def test_llm_selector_filters_to_cand() -> None:
    payload = json.dumps({"supporting_uris": ["u1", "invented", "u3"]})
    llm = FakeLLMProvider(
        script=[
            LLMResponse(
                content=payload,
                model="fake",
                finish_reason=FinishReason.STOP,
                usage=TokenUsage(input_tokens=1, output_tokens=1, total_tokens=2),
            )
        ]
    )
    sel = LLMSupportSelector(llm, max_support=8)
    got = await sel.select(
        ["u1", "u2", "u3"],
        question="q",
        answer="a",
    )
    assert got == frozenset({"u1", "u3"})
    assert "invented" not in got


@pytest.mark.asyncio
async def test_build_support_selector_modes() -> None:
    d = build_support_selector(mode="deterministic")
    assert isinstance(d, DeterministicSupportSelector)
    llm = FakeLLMProvider(script=[])
    s = build_support_selector(mode="llm", llm=llm)
    assert isinstance(s, LLMSupportSelector)
