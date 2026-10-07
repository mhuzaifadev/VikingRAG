"""Historical-question warm-up scaffolding (paper §6, M default 1000).

Does not invent questions or scores. Requires an ingested corpus + LLM.
When prerequisites are missing, raises a clear blocked error.
"""

from __future__ import annotations

from dataclasses import dataclass

from vikingrag.domain.errors import DomainError


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


def plan_warmup(*, dataset: str, m: int = 1000, data_dir: str = "data/eval") -> WarmupPlan:
    if m < 1:
        raise ValueError("m must be >= 1")
    return WarmupPlan(dataset=dataset, m=m, data_dir=data_dir)


def run_warmup(plan: WarmupPlan, *, llm_configured: bool, corpus_present: bool) -> None:
    """Execute warm-up — currently scaffolding only (blocked without setup)."""
    if not corpus_present:
        raise WarmupBlockedError(
            f"Corpus for '{plan.dataset}' not present under {plan.data_dir}. "
            f"Use: vikingrag-eval prepare --dataset {plan.dataset} --download "
            "(or place verified files manually). Status: BLOCKED."
        )
    if not llm_configured:
        raise WarmupBlockedError(
            "Historical-question generation requires a configured LLM provider. "
            "Set VIKINGRAG_LLM_PROVIDER / API key. Status: BLOCKED."
        )
    raise WarmupBlockedError(
        f"Warm-up execution for M={plan.m} on '{plan.dataset}' is scaffolded but not "
        "wired to live generation in this release (avoids inventing questions/scores). "
        f"Plan notes: {plan.notes}"
    )
