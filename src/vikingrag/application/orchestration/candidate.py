"""Draft a one-round candidate answer for Section 5 E+ gate."""

from __future__ import annotations

from typing import Any

from vikingrag.application.agent.finalize import finalize_answer, validate_citations
from vikingrag.application.budget import RetrievalContext
from vikingrag.domain.models.answer import AnswerCitation, AnswerStatus
from vikingrag.domain.models.evidence import EvidenceBundle, RetrievedEvidence
from vikingrag.providers.llm.base import LLMProvider


async def draft_candidate_answer(
    *,
    question: str,
    evidence: EvidenceBundle | list[RetrievedEvidence],
    llm: LLMProvider,
    ctx: RetrievalContext,
    model: str | None = None,
) -> tuple[str | None, list[AnswerCitation], dict[str, Any]]:
    """Produce candidate answer A with validated citations (or None if abstained)."""
    items = list(evidence.items) if isinstance(evidence, EvidenceBundle) else list(evidence)
    status, answer, citations, meta = await finalize_answer(
        question=question,
        evidence=items,
        llm=llm,
        ctx=ctx,
        model=model,
    )
    if status is not AnswerStatus.ANSWERED or not answer:
        return None, [], {**meta, "candidate_status": status.value}
    valid = validate_citations(citations, items)
    if not valid:
        return None, [], {**meta, "candidate_status": "citations_invalid"}
    return answer, valid, {**meta, "candidate_status": "ok"}


def citation_uris_subset_of_evidence(
    citations: list[AnswerCitation],
    evidence: EvidenceBundle | list[RetrievedEvidence],
) -> bool:
    items = list(evidence.items) if isinstance(evidence, EvidenceBundle) else list(evidence)
    allowed = {e.evidence_id for e in items}
    uris = {e.uri for e in items}
    if not citations:
        return False
    for c in citations:
        if c.evidence_id not in allowed:
            return False
        if c.uri and c.uri not in uris:
            return False
    return True
