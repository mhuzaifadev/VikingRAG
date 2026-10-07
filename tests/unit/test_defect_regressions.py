"""Named regression tests for v0.3 → v1 defect repairs."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.application.budget import BudgetLimits, RetrievalContext
from vikingrag.application.evidence_bundle import assemble_evidence_bundle
from vikingrag.application.grep_primitive import _find_literal_positions
from vikingrag.application.read_primitive import _slice_by_tokens
from vikingrag.domain.errors import BudgetExhaustedError, ScopeDeniedError
from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.evidence import RetrievedEvidence
from vikingrag.domain.models.primitives import OffsetSystem
from vikingrag.ingestion.tokenization import ApproxWhitespaceTokenizer, TikTokenTokenizer


@pytest.mark.asyncio
async def test_reserve_is_atomic_no_partial_mutation() -> None:
    """Combined reserve must not charge embedding if tool_calls would overspend."""
    ctx = RetrievalContext.create(limits=BudgetLimits(max_embedding_calls=5, max_tool_calls=1))
    await ctx.reserve(tool_calls=1)
    with pytest.raises(BudgetExhaustedError):
        await ctx.reserve(embedding_calls=1, tool_calls=1)
    assert ctx.usage.embedding_calls == 0
    assert ctx.usage.tool_calls == 1


@pytest.mark.asyncio
async def test_empty_allowlist_is_deny_all_before_work() -> None:
    ctx = RetrievalContext.create(permitted_document_ids=frozenset())
    assert ctx.is_deny_all
    with pytest.raises(ScopeDeniedError):
        ctx.require_scope_not_empty()
    with pytest.raises(ScopeDeniedError):
        ctx.narrow_document_ids([])


def test_none_permit_is_unrestricted() -> None:
    ctx = RetrievalContext.create(permitted_document_ids=None)
    assert ctx.is_unrestricted
    assert ctx.narrow_document_ids([]) == ()


def test_read_token_cap_never_exceeds_budget_emoji() -> None:
    tok = TikTokenTokenizer()
    # Dense emoji / CJK that would blow past max_tokens * 4 char heuristic
    content = "🔥" * 50 + "政策" * 50
    # If one codepoint alone exceeds budget, failure must be explicit (not overshoot)
    first_cost = tok.count(content[0])
    if first_cost > 1:
        from vikingrag.domain.errors import ValidationDomainError

        with pytest.raises(ValidationDomainError):
            _slice_by_tokens(content, start_offset=0, max_tokens=1, tokenizer=tok)
        text, end, truncated = _slice_by_tokens(
            content, start_offset=0, max_tokens=first_cost, tokenizer=tok
        )
        assert tok.count(text) <= first_cost
    else:
        text, end, truncated = _slice_by_tokens(
            content, start_offset=0, max_tokens=1, tokenizer=tok
        )
        assert tok.count(text) <= 1
    assert end <= len(content)
    assert truncated or tok.count(content) <= max(1, first_cost)


def test_read_token_cap_approx_tokenizer_no_char_star4() -> None:
    tok = ApproxWhitespaceTokenizer()
    content = "x" * 500  # no whitespace → count uses chars//4
    text, _end, _trunc = _slice_by_tokens(content, start_offset=0, max_tokens=2, tokenizer=tok)
    assert tok.count(text) <= 2


def test_grep_unicode_casefold_turkish_i() -> None:
    # Turkish İ casefold length differs from lower()
    content = "İ policy handbook"
    positions = _find_literal_positions(content, "i policy", case_sensitive=False)
    assert positions, "expected case-insensitive match for İ→i"
    start, end = positions[0]
    matched = content[start:end]
    assert "policy" in matched.casefold()
    # Offsets must be valid unicode indices into original
    assert 0 <= start < end <= len(content)


def test_overlap_retains_uncovered_tail() -> None:
    nid = NodeId(uuid4())
    did = DocumentId(uuid4())
    first = RetrievedEvidence(
        evidence_id="e1",
        uri="viking://documents/d/nodes/n",
        text="AAAAABBBBBCCCCC",
        token_count=3,
        document_id=did,
        node_id=nid,
        content_hash="h1",
        start_offset=0,
        end_offset=10,
        offset_system=OffsetSystem.UNICODE_CODE_POINT,
    )
    second = RetrievedEvidence(
        evidence_id="e2",
        uri="viking://documents/d/nodes/n",
        text="BBBBBCCCCCDDDDD",
        token_count=3,
        document_id=did,
        node_id=nid,
        content_hash="h1",
        start_offset=5,
        end_offset=20,
        offset_system=OffsetSystem.UNICODE_CODE_POINT,
    )
    bundle = assemble_evidence_bundle([first, second], max_tokens=100)
    # First full span kept; uncovered tail of second (10..20) retained
    spans = {(i.start_offset, i.end_offset) for i in bundle.items}
    assert (0, 10) in spans
    assert any(s[0] >= 10 and s[1] == 20 for s in spans) or (10, 20) in spans


@pytest.mark.asyncio
async def test_add_usage_enforces_caps() -> None:
    ctx = RetrievalContext.create(limits=BudgetLimits(max_nodes_inspected=2))
    await ctx.add_usage(nodes_inspected=2)
    with pytest.raises(BudgetExhaustedError):
        await ctx.add_usage(nodes_inspected=1)
