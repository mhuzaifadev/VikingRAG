"""Bottom-up summary ordering and fake generator tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.domain.models.document import NodeId, NodeType
from vikingrag.domain.models.representation import SummaryRequest
from vikingrag.providers.llm.fake import FakeSummaryGenerator


@pytest.mark.asyncio
async def test_fake_summary_includes_children_and_title() -> None:
    gen = FakeSummaryGenerator()
    result = await gen.summarize(
        SummaryRequest(
            node_id=NodeId(uuid4()),
            node_type=NodeType.SECTION,
            title="Retrieval",
            own_content="Parent notes",
            child_summaries=("Evidence Verification details", "Semantic Search overview"),
        )
    )
    assert "Retrieval" in result.text
    assert "Evidence Verification" in result.text
    assert result.model == gen.model


@pytest.mark.asyncio
async def test_summary_request_depth_order_simulation() -> None:
    """Parents should be summarized after children in indexing (depth descending)."""
    gen = FakeSummaryGenerator()
    child = await gen.summarize(
        SummaryRequest(
            node_id=NodeId(uuid4()),
            node_type=NodeType.SUBSECTION,
            title="Evidence Verification",
            own_content="claim to evidence mapping",
            child_summaries=(),
        )
    )
    parent = await gen.summarize(
        SummaryRequest(
            node_id=NodeId(uuid4()),
            node_type=NodeType.SECTION,
            title="Retrieval",
            own_content=None,
            child_summaries=(child.text,),
        )
    )
    assert gen.calls[0].node_type is NodeType.SUBSECTION
    assert gen.calls[1].node_type is NodeType.SECTION
    assert "Evidence Verification" in parent.text
