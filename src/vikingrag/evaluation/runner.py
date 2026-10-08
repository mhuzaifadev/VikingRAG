"""Minimal evaluation run — records answers/latency/tokens; never invents scores."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class EvalExample:
    example_id: str
    question: str
    gold_answer: str | None = None
    document_ids: tuple[str, ...] = ()


@dataclass(slots=True)
class MethodResult:
    method: str
    example_id: str
    answer: str | None
    latency_ms: float
    usage: dict[str, Any] = field(default_factory=dict)
    citation_count: int = 0
    error: str | None = None


@dataclass(slots=True)
class EvalRunReport:
    status: str  # completed_unjudged | completed | blocked
    methods: tuple[str, ...]
    results: list[MethodResult]
    measured_scores: dict[str, Any] | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "methods": list(self.methods),
            "results": [asdict(r) for r in self.results],
            "measured_scores": self.measured_scores,
            "notes": self.notes,
        }


class AnswerCallable(Protocol):
    async def __call__(self, question: str, *, document_ids: tuple[str, ...] = ()) -> Any: ...


def load_examples_from_manifest(path: Path) -> list[EvalExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("examples") or data.get("questions") or []
    if not isinstance(raw, list):
        raise ValueError("manifest examples must be a list")
    out: list[EvalExample] = []
    for i, item in enumerate(raw):
        if isinstance(item, str):
            out.append(EvalExample(example_id=f"q{i}", question=item))
            continue
        if not isinstance(item, dict):
            continue
        q = str(item.get("question") or item.get("query") or "").strip()
        if not q:
            continue
        out.append(
            EvalExample(
                example_id=str(item.get("id") or f"q{i}"),
                question=q,
                gold_answer=(str(item["gold"]) if item.get("gold") else None),
                document_ids=tuple(str(x) for x in (item.get("document_ids") or ())),
            )
        )
    return out


async def run_evaluation(
    *,
    examples: list[EvalExample],
    methods: dict[str, AnswerCallable],
    judge: Any | None = None,
) -> EvalRunReport:
    """Execute methods on examples. Judge optional — scores stay null without it."""
    del judge  # claim-level judging reserved; do not invent scores
    results: list[MethodResult] = []
    for name, fn in methods.items():
        for ex in examples:
            started = time.perf_counter()
            try:
                resp = await fn(ex.question, document_ids=ex.document_ids)
                latency = (time.perf_counter() - started) * 1000.0
                answer = getattr(resp, "answer", None)
                if isinstance(resp, dict):
                    answer = resp.get("answer")
                citations = getattr(resp, "citations", ()) or ()
                usage = dict(getattr(resp, "usage", {}) or {})
                results.append(
                    MethodResult(
                        method=name,
                        example_id=ex.example_id,
                        answer=answer if isinstance(answer, str) else None,
                        latency_ms=latency,
                        usage=usage,
                        citation_count=len(citations),
                    )
                )
            except Exception as exc:
                results.append(
                    MethodResult(
                        method=name,
                        example_id=ex.example_id,
                        answer=None,
                        latency_ms=(time.perf_counter() - started) * 1000.0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
    return EvalRunReport(
        status="completed_unjudged",
        methods=tuple(methods.keys()),
        results=results,
        measured_scores=None,
        notes=(
            "Answers recorded with latency/usage/citation_count. "
            "measured_scores is null until an explicit judge is configured."
        ),
    )
