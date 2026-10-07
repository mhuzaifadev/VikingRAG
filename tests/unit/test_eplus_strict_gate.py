"""Section 5 E+ strict sufficiency: candidate + assessor gate."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from vikingrag.application.agent.loop import AgentRetrieveResult, StubAgenticRetrievalLoop
from vikingrag.application.assessment import ScriptedEvidenceAssessor
from vikingrag.application.budget import RetrievalContext
from vikingrag.application.orchestration.query import QueryOrchestrator, QueryRequest
from vikingrag.application.search_plus import SearchPlusResponse
from vikingrag.domain.models.answer import ExecutionMode
from vikingrag.domain.models.assessment import AssessmentStatus
from vikingrag.domain.models.document import DocumentId, NodeId, NodeType
from vikingrag.domain.models.evidence import EvidenceBundle
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


class _FakeSearchPlus:
    def __init__(self, uris: tuple[str, ...] = ()) -> None:
        self._uris = uris or ("viking://doc/a#chunk-1",)

    async def search(
        self, request: Any, *, ctx: Any = None, trace: Any = None
    ) -> SearchPlusResponse:
        del request, ctx, trace
        doc_id = DocumentId(uuid4())
        node_id = NodeId(uuid4())
        hits = (
            SearchHit(
                uri=self._uris[0],
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
    def __init__(self, text: str = "The late fee is $25.") -> None:
        self._text = text
        self._doc = DocumentId(uuid4())
        self._node = NodeId(uuid4())

    async def read(self, request: Any, *, ctx: Any = None) -> ReadResponse:
        del ctx
        return ReadResponse(
            uri=request.uri,
            document_id=self._doc,
            node_id=self._node,
            node_type=NodeType.CHUNK,
            title=None,
            content_hash="abc",
            has_direct_content=True,
            text=self._text,
            start_offset=0,
            end_offset=len(self._text),
            offset_system=OffsetSystem.UNICODE_CODE_POINT,
            token_count=6,
            truncated=False,
            next_offset=None,
        )


@pytest.mark.asyncio
async def test_eplus_soft_escalate_without_assessor() -> None:
    orch = QueryOrchestrator(
        search=MagicMock(),
        search_plus=_FakeSearchPlus(),  # type: ignore[arg-type]
        assessor=None,
        agent=StubAgenticRetrievalLoop(),
        read_service=None,
    )
    resp = await orch.run(
        QueryRequest(query="What fee?", mode=ExecutionMode.VIKINGRAG_E_PLUS),
        ctx=RetrievalContext.create(),
    )
    assert "assessor_unavailable" in resp.gaps
    # Stub raises → abstained one-round path
    assert resp.route is QueryRunRoute.ONE_ROUND


@pytest.mark.asyncio
async def test_eplus_one_round_when_scripted_sufficient() -> None:
    orch = QueryOrchestrator(
        search=MagicMock(),
        search_plus=_FakeSearchPlus(),  # type: ignore[arg-type]
        assessor=ScriptedEvidenceAssessor(status=AssessmentStatus.SUFFICIENT),
        agent=StubAgenticRetrievalLoop(),
        read_service=_Read(),
        llm=None,
    )
    resp = await orch.run(
        QueryRequest(query="What is the late fee?", mode=ExecutionMode.VIKINGRAG_E_PLUS),
        ctx=RetrievalContext.create(),
    )
    assert resp.route is QueryRunRoute.ONE_ROUND
    assert resp.gaps == ()
    assert resp.assessment is not None
    assert resp.assessment.status is AssessmentStatus.SUFFICIENT


@pytest.mark.asyncio
async def test_eplus_escalates_when_insufficient() -> None:
    class _Agent:
        async def retrieve(self, *args: Any, **kwargs: Any) -> AgentRetrieveResult:
            return AgentRetrieveResult(
                answer="escalated answer",
                evidence=EvidenceBundle.from_items([]),
                gaps=kwargs.get("gaps") or (),
                rounds=2,
                tool_calls=3,
            )

    orch = QueryOrchestrator(
        search=MagicMock(),
        search_plus=_FakeSearchPlus(),  # type: ignore[arg-type]
        assessor=ScriptedEvidenceAssessor(status=AssessmentStatus.INSUFFICIENT),
        agent=_Agent(),  # type: ignore[arg-type]
        read_service=_Read(text="Unrelated text about parking."),
        llm=None,
    )
    resp = await orch.run(
        QueryRequest(query="What is the late fee?", mode=ExecutionMode.VIKINGRAG_E_PLUS),
        ctx=RetrievalContext.create(),
    )
    assert resp.route is QueryRunRoute.ESCALATED
    assert resp.answer == "escalated answer"
