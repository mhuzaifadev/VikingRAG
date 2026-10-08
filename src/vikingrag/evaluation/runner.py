"""Minimal evaluation run — records answers/latency/tokens; never invents scores."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from vikingrag.domain.models.answer import AnswerRequest, AnswerResponse, ExecutionMode
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.experience import LearningPolicy


class ManifestValidationError(ValueError):
    """Malformed explicit scope or examples — fail before provider/DB work."""


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
    status: str  # completed_unjudged | completed | partial | failed | blocked
    methods: tuple[str, ...]
    results: list[MethodResult]
    measured_scores: dict[str, Any] | None
    notes: str
    export_metrics: dict[str, Any] | None = None
    official_baseline: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "methods": list(self.methods),
            "results": [asdict(r) for r in self.results],
            "measured_scores": self.measured_scores,
            "export_metrics": self.export_metrics,
            "official_baseline": self.official_baseline
            or {
                "method": "official_vikingrag_e_plus",
                "status": "not_run",
                "reason": (
                    "Official AGPL runner is not vendored; pin a separate checkout/container "
                    "to measure, or leave as not_run."
                ),
            },
            "notes": self.notes,
            "learning_policy": LearningPolicy.FROZEN.value,
        }


class AnswerCallable(Protocol):
    async def __call__(self, question: str, *, document_ids: tuple[str, ...] = ()) -> Any: ...


VIKINGRAG_METHOD_MODES: dict[str, ExecutionMode] = {
    "vikingrag": ExecutionMode.VIKINGRAG,
    "vikingrag_e": ExecutionMode.VIKINGRAG_E,
    "vikingrag_e_plus": ExecutionMode.VIKINGRAG_E_PLUS,
}

DEFAULT_EVAL_METHODS: tuple[str, ...] = tuple(VIKINGRAG_METHOD_MODES.keys())


def parse_document_ids(
    raw: tuple[str, ...] | list[str],
    *,
    field: str = "document_ids",
    example_id: str | None = None,
) -> tuple[DocumentId, ...]:
    """Parse document UUIDs; reject malformed entries (never silently drop to unrestricted)."""
    out: list[DocumentId] = []
    for value in raw:
        text = str(value).strip()
        if not text:
            where = f"example {example_id!r} " if example_id else ""
            raise ManifestValidationError(
                f"Empty {where}{field} entry is invalid; refusing unrestricted fallback"
            )
        try:
            out.append(DocumentId(UUID(text)))
        except (ValueError, TypeError) as exc:
            where = f"example {example_id!r} " if example_id else ""
            raise ManifestValidationError(
                f"Malformed {where}{field} value {text!r}: expected UUID. "
                "Refusing to broaden scope to unrestricted retrieval."
            ) from exc
    return tuple(out)


# Back-compat alias used by older call sites / tests.
def _parse_document_ids(raw: tuple[str, ...]) -> tuple[DocumentId, ...]:
    return parse_document_ids(raw)


def build_vikingrag_methods(
    client: Any,
    *,
    learning_policy: LearningPolicy = LearningPolicy.FROZEN,
) -> dict[str, AnswerCallable]:
    """Map paper method names to ``AnswerGenerator.generate`` with execution modes.

    Held-out evaluation defaults to ``frozen`` so scoring cannot enqueue edge builds.
    """

    generator = client.answer_generator()

    def _make(mode: ExecutionMode) -> AnswerCallable:
        async def _call(question: str, *, document_ids: tuple[str, ...] = ()) -> AnswerResponse:
            resp = await generator.generate(
                AnswerRequest(
                    question=question,
                    document_ids=parse_document_ids(document_ids, field="document_ids"),
                    execution_mode=mode,
                    learning_policy=learning_policy,
                )
            )
            if not isinstance(resp, AnswerResponse):
                raise TypeError(f"expected AnswerResponse, got {type(resp)!r}")
            return resp

        return _call

    return {name: _make(mode) for name, mode in VIKINGRAG_METHOD_MODES.items()}


def echo_placeholder_method() -> AnswerCallable:
    """Wiring-only echo; keep behind ``--method echo``."""

    async def _echo(question: str, *, document_ids: tuple[str, ...] = ()) -> dict[str, object]:
        del document_ids
        return {
            "answer": None,
            "citations": (),
            "usage": {},
            "note": "echo_placeholder",
            "question": question,
        }

    return _echo


def resolve_methods(
    names: list[str] | tuple[str, ...] | None,
    *,
    client: Any | None = None,
    learning_policy: LearningPolicy = LearningPolicy.FROZEN,
) -> dict[str, AnswerCallable]:
    """Build method callables from a comma-split name list."""
    selected = list(names) if names else list(DEFAULT_EVAL_METHODS)
    out: dict[str, AnswerCallable] = {}
    need_client = any(
        n.strip().lower() in VIKINGRAG_METHOD_MODES or n.strip().lower() == "flat_rag"
        for n in selected
    )
    if need_client and client is None:
        raise ValueError(
            "VikingRAG methods require a VikingRAGClient; "
            "pass client= or construct via VikingRAGClient.from_settings()"
        )
    viking = (
        build_vikingrag_methods(client, learning_policy=learning_policy)
        if client is not None
        else {}
    )
    if client is not None and any(n.strip().lower() == "flat_rag" for n in selected):
        viking["flat_rag"] = _make_flat_rag(client, learning_policy=learning_policy)
    for name in selected:
        key = name.strip().lower()
        if key in {"echo", "echo_placeholder"}:
            out["echo_placeholder"] = echo_placeholder_method()
        elif key in viking:
            out[key] = viking[key]
        else:
            known = ", ".join([*DEFAULT_EVAL_METHODS, "flat_rag", "echo"])
            raise ValueError(f"Unknown method {name!r}; known: {known}")
    return out


def _make_flat_rag(
    client: Any, *, learning_policy: LearningPolicy = LearningPolicy.FROZEN
) -> AnswerCallable:
    """Single-shot Search + Read baseline without agent / Search+."""

    async def _call(question: str, *, document_ids: tuple[str, ...] = ()) -> AnswerResponse:
        # Flat RAG uses ordinary vikingrag mode with learning frozen; agent still runs
        # but without Search+. Callers comparing modes should use identical budgets.
        generator = client.answer_generator()
        resp = await generator.generate(
            AnswerRequest(
                question=question,
                document_ids=parse_document_ids(document_ids, field="document_ids"),
                execution_mode=ExecutionMode.VIKINGRAG,
                learning_policy=learning_policy,
                instructions="Answer from top retrieved chunks only; do not use experience edges.",
            )
        )
        if not isinstance(resp, AnswerResponse):
            raise TypeError(f"expected AnswerResponse, got {type(resp)!r}")
        return resp

    return _call


def citation_validity_metrics(results: list[MethodResult]) -> dict[str, Any]:
    """Deterministic rates: non-empty answers and citation presence. Not LLM accuracy."""
    by_method: dict[str, dict[str, float | int]] = {}
    for r in results:
        bucket = by_method.setdefault(
            r.method,
            {
                "n": 0,
                "answered": 0,
                "with_citations": 0,
                "errors": 0,
            },
        )
        bucket["n"] = int(bucket["n"]) + 1
        if r.error:
            bucket["errors"] = int(bucket["errors"]) + 1
            continue
        if isinstance(r.answer, str) and r.answer.strip():
            bucket["answered"] = int(bucket["answered"]) + 1
        if r.citation_count > 0:
            bucket["with_citations"] = int(bucket["with_citations"]) + 1

    scores: dict[str, Any] = {}
    for method, b in by_method.items():
        n = int(b["n"])
        scored = max(n - int(b["errors"]), 0)
        scores[method] = {
            "n": n,
            "error_rate": (int(b["errors"]) / n) if n else 0.0,
            "non_empty_answer_rate": (int(b["answered"]) / scored) if scored else 0.0,
            "citation_presence_rate": (int(b["with_citations"]) / scored) if scored else 0.0,
        }
    return scores


def _percentile(sorted_vals: list[float], p: float) -> float | None:
    if not sorted_vals:
        return None
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def export_latency_and_usage(results: list[MethodResult]) -> dict[str, Any]:
    """Aggregate latency p50/p95, token sums, and route/error rates per method."""
    by_method: dict[str, list[MethodResult]] = {}
    for r in results:
        by_method.setdefault(r.method, []).append(r)

    out: dict[str, Any] = {}
    for method, rows in by_method.items():
        latencies = sorted(r.latency_ms for r in rows if r.error is None)
        in_tok = sum(int((r.usage or {}).get("input_tokens") or 0) for r in rows)
        out_tok = sum(int((r.usage or {}).get("output_tokens") or 0) for r in rows)
        errors = sum(1 for r in rows if r.error)
        out[method] = {
            "n": len(rows),
            "errors": errors,
            "failure_rate": (errors / len(rows)) if rows else 0.0,
            "latency_ms": {
                "p50": _percentile(latencies, 50),
                "p95": _percentile(latencies, 95),
            },
            "tokens": {"input": in_tok, "output": out_tok},
        }
    return out


def write_export_csv(report: EvalRunReport, path: Path) -> None:
    """Write a flat CSV of per-example results for external analysis."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "method",
                "example_id",
                "latency_ms",
                "citation_count",
                "error",
                "answered",
            ],
        )
        writer.writeheader()
        for r in report.results:
            writer.writerow(
                {
                    "method": r.method,
                    "example_id": r.example_id,
                    "latency_ms": f"{r.latency_ms:.3f}",
                    "citation_count": r.citation_count,
                    "error": r.error or "",
                    "answered": bool(r.answer and r.answer.strip()),
                }
            )


def execution_status(results: list[MethodResult], *, judged: bool) -> str:
    """Derive run status from method outcomes (separate from quality metrics)."""
    if not results:
        return "failed"
    errors = sum(1 for r in results if r.error)
    if errors == len(results):
        return "failed"
    if errors > 0:
        return "partial"
    return "completed" if judged else "completed_unjudged"


def load_examples_from_manifest(path: Path) -> list[EvalExample]:
    data = json.loads(path.read_text(encoding="utf-8"))
    raw = data.get("examples") or data.get("questions") or []
    if not isinstance(raw, list):
        raise ManifestValidationError("manifest examples must be a list")
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
        example_id = str(item.get("id") or f"q{i}")
        raw_ids = tuple(str(x) for x in (item.get("document_ids") or ()))
        # Validate before any provider/DB work; keep string form on the example.
        if raw_ids:
            parse_document_ids(raw_ids, field="document_ids", example_id=example_id)
        out.append(
            EvalExample(
                example_id=example_id,
                question=q,
                gold_answer=(str(item["gold"]) if item.get("gold") else None),
                document_ids=raw_ids,
            )
        )
    return out


async def run_evaluation(
    *,
    examples: list[EvalExample],
    methods: dict[str, AnswerCallable],
    judge: str | None = None,
) -> EvalRunReport:
    """Execute methods on examples.

    ``judge="scripted"`` computes deterministic citation/non-empty rates into
    ``measured_scores``. Anything else leaves scores null (no invented LLM accuracy).
    """
    # Re-validate scopes before any method call (defense in depth).
    for ex in examples:
        if ex.document_ids:
            parse_document_ids(ex.document_ids, field="document_ids", example_id=ex.example_id)

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
                    citations = resp.get("citations") or ()
                    usage = dict(resp.get("usage") or {})
                else:
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

    judge_key = (judge or "").strip().lower()
    judged = judge_key in {"scripted", "citation", "deterministic"}
    status = execution_status(results, judged=judged)
    measured = citation_validity_metrics(results) if judged else None
    export = export_latency_and_usage(results)
    if judged:
        notes = (
            "Deterministic citation_presence_rate / non_empty_answer_rate only. "
            "Not LLM answer accuracy. Held-out learning_policy=frozen. "
            f"execution_status={status}."
        )
    else:
        notes = (
            "Answers recorded with latency/usage/citation_count. "
            "measured_scores is null until --judge scripted (or equivalent) is set. "
            "Held-out learning_policy=frozen. "
            f"execution_status={status}."
        )
    return EvalRunReport(
        status=status,
        methods=tuple(methods.keys()),
        results=results,
        measured_scores=measured,
        export_metrics=export,
        notes=notes,
    )
