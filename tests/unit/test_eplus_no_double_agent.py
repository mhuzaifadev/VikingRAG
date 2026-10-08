"""E+ must not run Algorithm 1 twice on escalation."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from vikingrag.application.agent.loop import AgentRetrieveResult
from vikingrag.application.assessment import ScriptedEvidenceAssessor
from vikingrag.application.orchestration.query import QueryOrchestrator, QueryRequest
from vikingrag.application.search_plus import SearchPlusResponse
from vikingrag.domain.models.answer import AnswerStatus, ExecutionMode
from vikingrag.domain.models.assessment import AssessmentStatus
from vikingrag.domain.models.document import DocumentId, NodeId, NodeType
from vikingrag.domain.models.experience import QueryRunRoute
from vikingrag.domain.models.primitives import OffsetSystem, ReadResponse
from vikingrag.domain.models.representation import (
    EmbeddingIdentity,
    RepresentationType,
    SearchHit,
    SearchResponse,
    SearchTiming,
    SearchUsage,
)


class _CountingAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def retrieve(self, query: str, **kwargs: Any) -> AgentRetrieveResult:
        del query, kwargs
        self.calls += 1
        from vikingrag.application.evidence_bundle import assemble_evidence_bundle

        return AgentRetrieveResult(
            evidence=assemble_evidence_bundle([], max_tokens=1000),
            gaps=("strict_sufficiency_failed",),
            rounds=2,
            tool_calls=3,
            answer="fallback answer",
            citations=[],
            events=[{"name": "Search", "result_uris": []}],
            usage={"tool_calls": 3},
            status=AnswerStatus.ANSWERED,
        )


class _FakeSearchPlus:
    async def search(
        self, request: Any, *, ctx: Any = None, trace: Any = None
    ) -> SearchPlusResponse:
        del request, ctx, trace
        doc_id = DocumentId(uuid4())
        node_id = NodeId(uuid4())
        hits = (
            SearchHit(
                uri="viking://doc/a#chunk-1",
                title=None,
                node_id=node_id,
                document_id=doc_id,
                node_type=NodeType.CHUNK,
                representation_type=RepresentationType.NODE_CONTENT,
                score=0.9,
                similarity=0.9,
                preview="preview",
            ),
        )
        base = SearchResponse(
            query="q",
            query_id=uuid4(),
            embedding_identity=EmbeddingIdentity(
                provider="test", model="test", dimensions=8, version="1"
            ),
            candidates=hits,
            timing=SearchTiming(),
            usage=SearchUsage(),
        )
        return SearchPlusResponse(base=base, cold_path=True)


class _Read:
    async def read(self, request: Any, *, ctx: Any = None) -> ReadResponse:
        del ctx
        text = "The late fee is $25."
        return ReadResponse(
            uri=request.uri,
            document_id=DocumentId(uuid4()),
            node_id=NodeId(uuid4()),
            node_type=NodeType.CHUNK,
            title=None,
            content_hash="abc",
            has_direct_content=True,
            text=text,
            start_offset=0,
            end_offset=len(text),
            offset_system=OffsetSystem.UNICODE_CODE_POINT,
            token_count=6,
            truncated=False,
            next_offset=None,
        )


@pytest.mark.asyncio
async def test_eplus_escalation_calls_agent_once() -> None:
    agent = _CountingAgent()
    orch = QueryOrchestrator(
        search=MagicMock(),
        search_plus=_FakeSearchPlus(),  # type: ignore[arg-type]
        assessor=ScriptedEvidenceAssessor(status=AssessmentStatus.INSUFFICIENT),
        agent=agent,  # type: ignore[arg-type]
        read_service=_Read(),
        llm=None,
    )
    resp = await orch.run(
        QueryRequest(query="What is the late fee?", mode=ExecutionMode.VIKINGRAG_E_PLUS)
    )
    assert resp.route is QueryRunRoute.ESCALATED
    assert agent.calls == 1
    assert resp.answer == "fallback answer"
    assert resp.usage.get("tool_calls") == 3
