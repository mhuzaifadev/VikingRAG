"""Bounded selection of answer-supporting URIs from U_cand (Alg 2 SUPPORT)."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from vikingrag.providers.llm.base import ChatMessage, LLMProvider


def select_support_uris(
    candidates: frozenset[str] | set[str] | Sequence[str],
    *,
    citations: Sequence[str] | None = None,
    answer: str | None = None,
    max_support: int = 16,
) -> frozenset[str]:
    """Deterministic SUPPORT (no LLM) — see DeterministicSupportSelector."""
    return DeterministicSupportSelector(max_support=max_support).select_sync(
        candidates,
        citations=citations,
        answer=answer,
    )


@runtime_checkable
class SupportSelector(Protocol):
    async def select(
        self,
        candidates: frozenset[str] | set[str] | Sequence[str],
        *,
        question: str | None = None,
        answer: str | None = None,
        citations: Sequence[str] | None = None,
        trace_summary: str | None = None,
    ) -> frozenset[str]: ...


class DeterministicSupportSelector:
    """Citation / heuristic URI filter (offline-safe default)."""

    def __init__(self, *, max_support: int = 16) -> None:
        if max_support < 1:
            raise ValueError("max_support must be >= 1")
        self._max = max_support

    def select_sync(
        self,
        candidates: frozenset[str] | set[str] | Sequence[str],
        *,
        citations: Sequence[str] | None = None,
        answer: str | None = None,
    ) -> frozenset[str]:
        cand = frozenset(u for u in candidates if u and u.strip())
        if not cand:
            return frozenset()

        selected: list[str] = []
        seen: set[str] = set()

        if citations:
            for uri in citations:
                if uri in cand and uri not in seen:
                    selected.append(uri)
                    seen.add(uri)
                    if len(selected) >= self._max:
                        return frozenset(selected)

        if answer and len(selected) < self._max:
            for uri in sorted(cand):
                if uri in seen:
                    continue
                tail = uri.rsplit("/", 1)[-1]
                if tail and tail in answer:
                    selected.append(uri)
                    seen.add(uri)
                    if len(selected) >= self._max:
                        return frozenset(selected)

        if not selected:
            for uri in sorted(cand):
                selected.append(uri)
                if len(selected) >= self._max:
                    break

        return frozenset(selected[: self._max])

    async def select(
        self,
        candidates: frozenset[str] | set[str] | Sequence[str],
        *,
        question: str | None = None,
        answer: str | None = None,
        citations: Sequence[str] | None = None,
        trace_summary: str | None = None,
    ) -> frozenset[str]:
        del question, trace_summary
        return self.select_sync(candidates, citations=citations, answer=answer)


class LLMSupportSelector:
    """Paper SUPPORT(·): LLM picks supporting URIs ⊆ U_cand only."""

    def __init__(
        self,
        llm: LLMProvider,
        *,
        model: str | None = None,
        max_support: int = 16,
        fallback: DeterministicSupportSelector | None = None,
    ) -> None:
        self._llm = llm
        self._model = model
        self._max = max_support
        self._fallback = fallback or DeterministicSupportSelector(max_support=max_support)

    async def select(
        self,
        candidates: frozenset[str] | set[str] | Sequence[str],
        *,
        question: str | None = None,
        answer: str | None = None,
        citations: Sequence[str] | None = None,
        trace_summary: str | None = None,
    ) -> frozenset[str]:
        cand = frozenset(u for u in candidates if u and u.strip())
        if not cand:
            return frozenset()
        prompt = (
            "Select URIs from the candidate list that provide useful evidence for "
            'the answer. Return JSON: {"supporting_uris": ["..."]}. '
            "Only use URIs from the candidate list. Do not invent URIs.\n\n"
            f"Question:\n{question or ''}\n\n"
            f"Answer:\n{answer or ''}\n\n"
            f"Trace summary:\n{trace_summary or ''}\n\n"
            f"Citations hint:\n{list(citations or [])}\n\n"
            f"Candidate URIs (U_cand):\n{sorted(cand)}\n"
        )
        try:
            response = await self._llm.generate(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "You select answer-supporting document URIs. "
                            "Respond with JSON only. Never invent URIs."
                        ),
                    ),
                    ChatMessage(role="user", content=prompt),
                ],
                model=self._model,
                temperature=0.0,
                max_tokens=600,
            )
            picked = _parse_supporting_uris(response.content or "")
            filtered = [u for u in picked if u in cand]
            if not filtered:
                return await self._fallback.select(
                    cand,
                    question=question,
                    answer=answer,
                    citations=citations,
                    trace_summary=trace_summary,
                )
            return frozenset(filtered[: self._max])
        except Exception:
            return await self._fallback.select(
                cand,
                question=question,
                answer=answer,
                citations=citations,
                trace_summary=trace_summary,
            )


def build_support_selector(
    *,
    mode: str,
    llm: LLMProvider | None = None,
    model: str | None = None,
    max_support: int = 16,
) -> SupportSelector:
    name = mode.lower().strip()
    if name == "llm":
        if llm is None:
            return DeterministicSupportSelector(max_support=max_support)
        return LLMSupportSelector(llm, model=model, max_support=max_support)
    return DeterministicSupportSelector(max_support=max_support)


def _parse_supporting_uris(content: str) -> list[str]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data: Any = json.loads(text)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        raw = data.get("supporting_uris") or data.get("uris") or []
    elif isinstance(data, list):
        raw = data
    else:
        return []
    if not isinstance(raw, list):
        return []
    return [str(u) for u in raw if u]
