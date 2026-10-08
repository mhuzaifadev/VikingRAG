"""Historical-question warm-up (paper §6, M default 1000).

Does not invent questions or scores. Requires an ingested corpus + LLM.
When prerequisites are missing, raises a clear blocked error.
When present, generates document-grounded questions and writes a manifest.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from vikingrag.domain.errors import DomainError
from vikingrag.providers.llm.base import ChatMessage, LLMProvider


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
        "Process with VikingRAG-E(+) to materialize experience edges before benchmark runs."
    )


@dataclass(frozen=True, slots=True)
class WarmupResult:
    dataset: str
    m_requested: int
    m_generated: int
    questions: tuple[str, ...]
    manifest_path: str
    status: str  # completed | blocked


def plan_warmup(*, dataset: str, m: int = 1000, data_dir: str = "data/eval") -> WarmupPlan:
    if m < 1:
        raise ValueError("m must be >= 1")
    return WarmupPlan(dataset=dataset, m=m, data_dir=data_dir)


def _corpus_text_sample(data_root: Path, *, max_chars: int = 12_000) -> str:
    chunks: list[str] = []
    total = 0
    for path in sorted(data_root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".txt", ".json", ".jsonl"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if path.suffix.lower() in {".json", ".jsonl"}:
            # Prefer raw text snippets, not gold QA fields
            text = text[:2000]
        remain = max_chars - total
        if remain <= 0:
            break
        piece = text[:remain]
        chunks.append(f"--- {path.name} ---\n{piece}")
        total += len(piece)
    return "\n\n".join(chunks)


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


def run_warmup(
    plan: WarmupPlan,
    *,
    llm_configured: bool,
    corpus_present: bool,
    llm: LLMProvider | None = None,
    model: str | None = None,
) -> WarmupResult:
    """Execute warm-up synchronously via asyncio when LLM is provided.

    Without ``llm``, raises blocked even if flags say configured (caller must wire).
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
    excerpt = _corpus_text_sample(data_root)
    if not excerpt.strip():
        raise WarmupBlockedError(
            f"Corpus under {data_root} has no readable text files for warm-up."
        )

    import asyncio

    questions = asyncio.run(
        generate_historical_questions(llm=llm, corpus_excerpt=excerpt, m=plan.m, model=model)
    )
    out_dir = data_root / "warmup"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"warmup_m{plan.m}.json"
    payload = {
        "dataset": plan.dataset,
        "m_requested": plan.m,
        "m_generated": len(questions),
        "questions": questions,
        "notes": plan.notes,
        "measured_scores": None,
        "status": "completed",
    }
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return WarmupResult(
        dataset=plan.dataset,
        m_requested=plan.m,
        m_generated=len(questions),
        questions=tuple(questions),
        manifest_path=str(manifest_path),
        status="completed",
    )


def warmup_result_dict(result: WarmupResult) -> dict[str, Any]:
    return asdict(result)
