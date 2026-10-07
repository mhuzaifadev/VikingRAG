"""Finalize a cited answer from collected Read evidence, or abstain."""

from __future__ import annotations

import json
import re
from typing import Any

from vikingrag.application.budget import RetrievalContext
from vikingrag.domain.models.answer import AnswerCitation, AnswerStatus
from vikingrag.domain.models.evidence import RetrievedEvidence
from vikingrag.providers.llm.base import ChatMessage, FinishReason, LLMProvider, LLMResponse


def evidence_from_read_events(
    events: list[dict[str, Any]],
) -> list[RetrievedEvidence]:
    """Build RetrievedEvidence list from Read tool results in order."""
    items: list[RetrievedEvidence] = []
    idx = 0
    for ev in events:
        if ev.get("name") != "Read" or not ev.get("ok"):
            continue
        result = ev.get("result") or {}
        text = str(result.get("text") or "")
        if not text.strip():
            continue
        idx += 1
        eid = f"e{idx}"
        items.append(
            RetrievedEvidence(
                evidence_id=eid,
                uri=str(result.get("uri") or ""),
                text=text,
                token_count=int(result.get("token_count") or 0),
                content_hash=str(result.get("content_hash") or ""),
                start_offset=int(result.get("start_offset") or 0),
                end_offset=int(result.get("end_offset") or 0),
                provenance=("read", "agent"),
                metadata={"tool_call_id": ev.get("tool_call_id")},
            )
        )
    return items


def validate_citations(
    citations: list[AnswerCitation],
    evidence: list[RetrievedEvidence],
) -> list[AnswerCitation]:
    by_id = {e.evidence_id: e for e in evidence}
    valid: list[AnswerCitation] = []
    for c in citations:
        item = by_id.get(c.evidence_id)
        if item is None:
            continue
        if c.quote and c.quote not in item.text:
            continue
        valid.append(
            AnswerCitation(
                evidence_id=c.evidence_id,
                uri=item.uri,
                quote=c.quote,
                start_offset=c.start_offset if c.start_offset is not None else item.start_offset,
                end_offset=c.end_offset if c.end_offset is not None else item.end_offset,
            )
        )
    return valid


async def finalize_answer(
    *,
    question: str,
    evidence: list[RetrievedEvidence],
    llm: LLMProvider,
    ctx: RetrievalContext,
    model: str | None = None,
    draft_content: str | None = None,
) -> tuple[AnswerStatus, str | None, list[AnswerCitation], dict[str, Any]]:
    """Produce a final answer with validated citations, or abstain."""
    if not evidence:
        return (
            AnswerStatus.INSUFFICIENT_EVIDENCE,
            None,
            [],
            {"reason": "no_source_evidence"},
        )

    await ctx.reserve(llm_calls=1, provider_attempts=1)
    evidence_block = "\n\n".join(
        f"[{e.evidence_id}] uri={e.uri} hash={e.content_hash}\n{e.text}" for e in evidence
    )
    prompt = (
        f"Question:\n{question}\n\n"
        f"Source evidence (cite by evidence_id; quotes must appear verbatim):\n"
        f"{evidence_block}\n\n"
        'Return JSON: {"status": "answered|insufficient_evidence|partial", '
        '"answer": string|null, '
        '"citations": [{"evidence_id": "e1", "quote": "..."}], '
        '"abstain_reason": string|null}. '
        "Only use the evidence above. Do not invent facts."
    )
    if draft_content:
        prompt += f"\n\nDraft from retrieval phase (verify against evidence):\n{draft_content}"

    try:
        response: LLMResponse = await ctx.await_with_deadline(
            llm.generate(
                [
                    ChatMessage(
                        role="system",
                        content=(
                            "You finalize grounded answers. Respond with JSON only. "
                            "Never cite evidence IDs that are not provided."
                        ),
                    ),
                    ChatMessage(role="user", content=prompt),
                ],
                model=model,
                temperature=0.0,
                max_tokens=800,
            )
        )
    except Exception as exc:
        return (
            AnswerStatus.PROVIDER_ERROR,
            None,
            [],
            {"error": type(exc).__name__, "message": str(exc)},
        )

    await ctx.add_usage(
        llm_input_tokens=response.input_tokens or 0,
        llm_output_tokens=response.output_tokens or 0,
    )
    parsed = _parse_json(response.content or "")
    status_raw = str(parsed.get("status") or "insufficient_evidence")
    try:
        status = AnswerStatus(status_raw)
    except ValueError:
        status = AnswerStatus.INSUFFICIENT_EVIDENCE
    if status is AnswerStatus.BUDGET_EXHAUSTED:
        status = AnswerStatus.INSUFFICIENT_EVIDENCE

    raw_cites = parsed.get("citations") or []
    citations: list[AnswerCitation] = []
    for c in raw_cites:
        if not isinstance(c, dict):
            continue
        eid = str(c.get("evidence_id") or "")
        if not eid:
            continue
        citations.append(
            AnswerCitation(
                evidence_id=eid,
                uri="",
                quote=str(c["quote"]) if c.get("quote") is not None else None,
            )
        )
    citations = validate_citations(citations, evidence)
    answer = parsed.get("answer")
    answer_s = str(answer).strip() if answer is not None else None
    if status is AnswerStatus.ANSWERED and (not answer_s or not citations):
        status = AnswerStatus.PARTIAL if answer_s else AnswerStatus.INSUFFICIENT_EVIDENCE
    if response.finish_reason is FinishReason.ERROR:
        status = AnswerStatus.PROVIDER_ERROR
    return (
        status,
        answer_s,
        citations,
        {
            "assessor_raw_status": status_raw,
            "abstain_reason": parsed.get("abstain_reason"),
        },
    )


def _parse_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {"status": "insufficient_evidence", "answer": None, "citations": []}
