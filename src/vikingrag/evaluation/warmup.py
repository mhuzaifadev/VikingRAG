"""Historical-question warm-up (paper §6, M default 1000).

Does not invent questions or scores. Requires an ingested corpus + LLM.
When prerequisites are missing, raises a clear blocked error.
When present, generates document-grounded questions, optionally runs E+/E
to enqueue learning, drains the edge builder, and writes a manifest.
"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vikingrag.domain.errors import DomainError
from vikingrag.domain.models.answer import AnswerRequest, ExecutionMode
from vikingrag.evaluation.base import SourceDocument
from vikingrag.providers.llm.base import ChatMessage, LLMProvider

AnswerFn = Callable[[str], Awaitable[Any]]
DrainFn = Callable[[], Awaitable[tuple[int, int]]]  # (jobs_drained, edges_built)


class WarmupBlockedError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="eval_warmup_blocked")


@dataclass(frozen=True, slots=True)
class WarmupPlan:
    """Declarative plan for generating M document-grounded historical questions."""

    dataset: str
    m: int = 1000
    data_dir: str = "data/eval"
    notes: str = (
        "Generate M distinct document-grounded questions from the ingested corpus only. "
        "Do not use evaluation QA pairs, gold answers, or paraphrases. "
        "Process with VikingRAG-E(+) to materialize experience edges before benchmark runs. "
        "Source text is read only from documents/ or corpus/ (.md/.txt) — never qa.* / gold JSON."
    )


@dataclass(frozen=True, slots=True)
class WarmupResult:
    dataset: str
    m_requested: int
    m_generated: int
    questions: tuple[str, ...]
    manifest_path: str
    status: str  # completed | blocked
    edges_built: int = 0
    jobs_drained: int = 0


def plan_warmup(*, dataset: str, m: int = 1000, data_dir: str = "data/eval") -> WarmupPlan:
    if m < 1:
        raise ValueError("m must be >= 1")
    return WarmupPlan(dataset=dataset, m=m, data_dir=data_dir)


def corpus_excerpt_from_documents(
    documents: Iterator[SourceDocument] | list[SourceDocument],
    *,
    max_chars: int = 12_000,
) -> str:
    """Join source-document text for LLM warm-up prompts."""
    chunks: list[str] = []
    total = 0
    for doc in documents:
        remain = max_chars - total
        if remain <= 0:
            break
        piece = doc.text[:remain]
        label = doc.document_id or (doc.path or "doc")
        chunks.append(f"--- {label} ---\n{piece}")
        total += len(piece)
    return "\n\n".join(chunks)


def _corpus_text_sample(data_root: Path, *, max_chars: int = 12_000) -> str:
    """Sample corpus text from adapter source documents only (no gold JSON)."""
    from vikingrag.evaluation.base import iter_source_documents_from_root

    return corpus_excerpt_from_documents(
        iter_source_documents_from_root(data_root),
        max_chars=max_chars,
    )


async def generate_historical_questions(
    *,
    llm: LLMProvider,
    corpus_excerpt: str,
    m: int,
    model: str | None = None,
) -> list[str]:
    """Ask the LLM for M doc-grounded questions; never invent offline."""
    prompt = (
        f"From the following corpus excerpts only, write exactly {m} distinct "
        "factual questions that can be answered from the text. "
        "Do not use evaluation gold answers or paraphrase known benchmarks. "
        'Return JSON: {"questions": ["..."]}.\n\n'
        f"Corpus:\n{corpus_excerpt}"
    )
    response = await llm.generate(
        [ChatMessage(role="user", content=prompt)],
        model=model,
        temperature=0.2,
        max_tokens=min(4000, 80 * m + 200),
    )
    content = (response.content or "").strip()
    if content.startswith("```"):
        content = content.strip("`")
        if content.startswith("json"):
            content = content[4:].strip()
    try:
        payload: dict[str, Any] = json.loads(content)
    except json.JSONDecodeError as exc:
        raise WarmupBlockedError(f"Warm-up LLM returned non-JSON: {exc}") from exc
    raw = payload.get("questions") or []
    if not isinstance(raw, list):
        raise WarmupBlockedError("Warm-up JSON questions must be a list")
    questions = [str(q).strip() for q in raw if str(q).strip()]
    if not questions:
        raise WarmupBlockedError("Warm-up produced zero questions")
    return questions[:m]


async def materialize_edges_from_questions(
    *,
    questions: list[str],
    answer_fn: AnswerFn,
    drain_fn: DrainFn | None = None,
    execution_mode: ExecutionMode = ExecutionMode.VIKINGRAG_E_PLUS,
    drain_timeout_s: float = 60.0,
) -> tuple[int, int]:
    """Run E+/E answers then drain PENDING edge-builder jobs.

    Returns ``(jobs_drained, edges_built)``. ``edges_built`` is whatever
    ``drain_fn`` reports (0 when drain is a no-op or unavailable).
    """
    del execution_mode  # caller binds mode into answer_fn
    for q in questions:
        await answer_fn(q)

    if drain_fn is None:
        return 0, 0

    jobs_total = 0
    edges_total = 0
    deadline = time.monotonic() + max(drain_timeout_s, 0.1)
    while time.monotonic() < deadline:
        jobs, edges = await drain_fn()
        jobs_total += jobs
        edges_total += edges
        if jobs == 0:
            break
        await _async_sleep(0.25)
    return jobs_total, edges_total


async def _async_sleep(seconds: float) -> None:
    import asyncio

    await asyncio.sleep(seconds)


def _client_answer_fn(client: Any, *, mode: ExecutionMode) -> AnswerFn:
    from vikingrag.domain.models.experience import LearningPolicy

    generator = client.answer_generator()

    async def _answer(question: str) -> Any:
        return await generator.generate(
            AnswerRequest(
                question=question,
                execution_mode=mode,
                learning_policy=LearningPolicy.LEARN,
            )
        )

    return _answer


def _client_drain_fn(client: Any) -> DrainFn:
    from vikingrag.workers.edge_builder import process_pending_edge_jobs

    async def _drain() -> tuple[int, int]:
        jobs = await process_pending_edge_jobs(client.database, batch_size=20)
        # Edge count is not returned by the worker; jobs_drained is authoritative.
        return jobs, 0

    return _drain


def run_warmup(
    plan: WarmupPlan,
    *,
    llm_configured: bool,
    corpus_present: bool,
    llm: LLMProvider | None = None,
    model: str | None = None,
    client: Any | None = None,
    answer_fn: AnswerFn | None = None,
    drain_fn: DrainFn | None = None,
    materialize: bool = True,
    execution_mode: ExecutionMode = ExecutionMode.VIKINGRAG_E_PLUS,
    drain_timeout_s: float = 60.0,
    smoke_max: int | None = None,
) -> WarmupResult:
    """Execute warm-up synchronously via asyncio when LLM is provided.

    Without ``llm``, raises blocked even if flags say configured (caller must wire).
    When ``materialize`` and an answer path exist, runs E+ (or E) then drains jobs.
    """
    if not corpus_present:
        raise WarmupBlockedError(
            f"Corpus for '{plan.dataset}' not present under {plan.data_dir}. "
            f"Use: vikingrag-eval prepare --dataset {plan.dataset} --download "
            "(or place verified files manually). Status: BLOCKED."
        )
    if not llm_configured or llm is None:
        raise WarmupBlockedError(
            "Historical-question generation requires a configured LLM provider. "
            "Set VIKINGRAG_LLM_PROVIDER / API key and pass a live provider. Status: BLOCKED."
        )

    from vikingrag.evaluation.adapters import get_adapter

    adapter = get_adapter(plan.dataset)
    data_root = adapter.data_root(Path(plan.data_dir))
    docs = list(adapter.iter_source_documents(data_root))
    excerpt = corpus_excerpt_from_documents(docs)
    if not excerpt.strip():
        raise WarmupBlockedError(
            f"Corpus under {data_root} has no source documents under "
            f"documents/ or corpus/ (.md/.txt only). Gold QA JSON is ignored."
        )

    import asyncio

    m = plan.m
    if smoke_max is not None:
        m = min(m, max(1, smoke_max))

    bound_answer = answer_fn
    bound_drain = drain_fn
    if materialize and bound_answer is None and client is not None:
        bound_answer = _client_answer_fn(client, mode=execution_mode)
        bound_drain = bound_drain or _client_drain_fn(client)
    if materialize and bound_answer is None:
        raise WarmupBlockedError(
            "materialize=True requires a VikingRAGClient or answer_fn; "
            "cannot complete experience materialization. Status: BLOCKED."
        )

    async def _run_all() -> tuple[list[str], int, int]:
        qs = await generate_historical_questions(llm=llm, corpus_excerpt=excerpt, m=m, model=model)
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for q in qs:
            key = q.casefold()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(q)
        jobs = 0
        edges = 0
        if materialize and bound_answer is not None:
            jobs, edges = await materialize_edges_from_questions(
                questions=deduped,
                answer_fn=bound_answer,
                drain_fn=bound_drain,
                execution_mode=execution_mode,
                drain_timeout_s=drain_timeout_s,
            )
            if materialize and jobs == 0 and edges == 0 and bound_drain is not None:
                # Soft signal — materialization attempted; caller inspects counts
                pass
        return deduped, jobs, edges

    questions, jobs_drained, edges_built = asyncio.run(_run_all())

    out_dir = data_root / "warmup"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"warmup_m{plan.m}.json"
    payload = {
        "dataset": plan.dataset,
        "m_requested": plan.m,
        "m_generated": len(questions),
        "questions": questions,
        "edges_built": edges_built,
        "jobs_drained": jobs_drained,
        "notes": plan.notes,
        "measured_scores": None,
        "status": "completed",
        "execution_mode": execution_mode.value if materialize else None,
        "learning_policy": "learn" if materialize else None,
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return WarmupResult(
        dataset=plan.dataset,
        m_requested=plan.m,
        m_generated=len(questions),
        questions=tuple(questions),
        manifest_path=str(manifest_path),
        status="completed",
        edges_built=edges_built,
        jobs_drained=jobs_drained,
    )


def warmup_result_dict(result: WarmupResult) -> dict[str, Any]:
    return asdict(result)
