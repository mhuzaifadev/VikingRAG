"""Unit tests for Algorithm 1 agent loop with scripted FakeLLM tool_calls."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from vikingrag.application.agent.finalize import validate_citations
from vikingrag.application.agent.loop import AgenticRetrievalExecutor
from vikingrag.application.agent.tool_executor import RetrievalToolExecutor, ToolExecutionRecord
from vikingrag.application.budget import BudgetLimits, RetrievalContext
from vikingrag.domain.models.answer import AnswerCitation, AnswerStatus
from vikingrag.domain.models.document import DocumentId, NodeId, NodeType
from vikingrag.domain.models.evidence import RetrievedEvidence
from vikingrag.domain.models.primitives import OffsetSystem, ReadResponse
from vikingrag.domain.models.representation import (
    EmbeddingIdentity,
    RepresentationType,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SearchTiming,
    SearchUsage,
)
from vikingrag.providers.llm.fake import (
    FakeLLMProvider,
    scripted_final_answer,
    scripted_tool_call,
)


class _FakeSearch:
    def __init__(self, hits: tuple[SearchHit, ...]) -> None:
        self.hits = hits
        self.calls = 0

    async def search(self, request: SearchRequest, *, ctx: Any = None) -> SearchResponse:
        self.calls += 1
        return SearchResponse(
            query=request.query,
            query_id=uuid4(),
            candidates=self.hits,
            timing=SearchTiming(),
            usage=SearchUsage(nodes_returned=len(self.hits)),
            embedding_identity=EmbeddingIdentity(provider="fake", model="fake", dimensions=8),
        )


class _FakeRead:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def read(self, request: Any, *, ctx: Any = None) -> ReadResponse:
        self.calls += 1
        return ReadResponse(
            uri=request.uri,
            document_id=DocumentId(uuid4()),
            node_id=NodeId(uuid4()),
            node_type=NodeType.CHUNK,
            title="Policy",
            content_hash="abc",
            has_direct_content=True,
            text=self.text,
            start_offset=0,
            end_offset=len(self.text),
            offset_system=OffsetSystem.UNICODE_CODE_POINT,
            token_count=max(1, len(self.text.split())),
            truncated=False,
            next_offset=None,
        )


class _NoopList:
    async def list(self, request: Any, *, ctx: Any = None) -> Any:
        raise AssertionError("List unexpected")


class _NoopGrep:
    async def grep(self, request: Any, *, ctx: Any = None) -> Any:
        raise AssertionError("Grep unexpected")


@pytest.mark.asyncio
async def test_agent_loop_search_read_stop_then_answer() -> None:
    doc_id = DocumentId(uuid4())
    node_id = NodeId(uuid4())
    uri = f"viking://documents/{doc_id}/nodes/{node_id}"
    hit = SearchHit(
        uri=uri,
        title="Policy",
        node_id=node_id,
        document_id=doc_id,
        node_type=NodeType.CHUNK,
        representation_type=RepresentationType.NODE_CONTENT,
        score=0.9,
        similarity=0.9,
        preview="Travel budget is $500",
    )
    search = _FakeSearch((hit,))
    read = _FakeRead("Travel budget is $500 per trip.")

    class ScriptedExecutor(RetrievalToolExecutor):
        async def execute(  # type: ignore[override]
            self,
            call: Any,
            *,
            ctx: RetrievalContext,
            default_top_k: int = 8,
            default_read_tokens: int = 512,
        ) -> ToolExecutionRecord:
            del default_top_k, default_read_tokens
            if call.name == "Search":
                resp = await search.search(
                    SearchRequest(query=str(call.arguments.get("query") or "q")),
                    ctx=ctx,
                )
                self.state.searched = True
                uris = tuple(h.uri for h in resp.candidates)
                rec = ToolExecutionRecord(
                    tool_call_id=call.id,
                    name="Search",
                    arguments=dict(call.arguments),
                    result={"hits": [{"uri": u} for u in uris]},
                    result_uris=uris,
                )
                self.state.events.append(rec)
                self.state.fingerprints.append(f"Search:{call.arguments}")
                return rec
            if call.name == "Read":
                rr = await read.read(type("R", (), {"uri": call.arguments["uri"]})(), ctx=ctx)
                self.state.read_uris.add(rr.uri)
                rec = ToolExecutionRecord(
                    tool_call_id=call.id,
                    name="Read",
                    arguments=dict(call.arguments),
                    result={
                        "uri": rr.uri,
                        "text": rr.text,
                        "content_hash": rr.content_hash,
                        "start_offset": rr.start_offset,
                        "end_offset": rr.end_offset,
                        "token_count": rr.token_count,
                        "has_direct_content": True,
                    },
                    result_uris=(rr.uri,),
                )
                self.state.events.append(rec)
                self.state.fingerprints.append(f"Read:{call.arguments}")
                return rec
            if call.name == "Stop":
                rec = ToolExecutionRecord(
                    tool_call_id=call.id,
                    name="Stop",
                    arguments=dict(call.arguments),
                    result={"stopped": True, "reason": "enough", "sufficient": True},
                )
                self.state.events.append(rec)
                return rec
            return await super().execute(call, ctx=ctx)

    llm = FakeLLMProvider(
        script=[
            scripted_tool_call(call_id="1", name="Search", arguments={"query": "travel budget"}),
            scripted_tool_call(call_id="2", name="Read", arguments={"uri": uri}),
            scripted_tool_call(
                call_id="3", name="Stop", arguments={"reason": "enough", "sufficient": True}
            ),
            scripted_final_answer(
                '{"status":"answered","answer":"The travel budget is $500 per trip.",'
                '"citations":[{"evidence_id":"e1","quote":"$500"}],"abstain_reason":null}'
            ),
        ]
    )
    executor = AgenticRetrievalExecutor(
        llm=llm,
        tool_executor=ScriptedExecutor(
            search=search,  # type: ignore[arg-type]
            list_service=_NoopList(),  # type: ignore[arg-type]
            grep=_NoopGrep(),  # type: ignore[arg-type]
            read=read,  # type: ignore[arg-type]
        ),
        max_rounds=6,
        finalization_llm_reserve=1,
    )
    ctx = RetrievalContext.create(
        limits=BudgetLimits(
            max_tool_calls=20,
            max_llm_calls=10,
            max_read_tokens=10_000,
            max_wall_time_ms=60_000,
        )
    )
    result = await executor.run("What is the travel budget?", ctx=ctx)

    assert search.calls == 1
    assert read.calls == 1
    assert result.status is AnswerStatus.ANSWERED
    assert result.answer is not None
    assert "500" in result.answer
    assert result.citations
    assert result.rounds_used >= 3


def test_citation_validation_rejects_unknown_id() -> None:
    evidence = [
        RetrievedEvidence(
            uri="u",
            text="hello world",
            token_count=2,
            evidence_id="e1",
            start_offset=0,
            end_offset=11,
            offset_system=OffsetSystem.UNICODE_CODE_POINT,
        )
    ]
    ok = validate_citations(
        [AnswerCitation(evidence_id="e1", uri="u", quote="hello")],
        evidence,
    )
    assert ok[0].evidence_id == "e1"
    bad = validate_citations(
        [AnswerCitation(evidence_id="e99", uri="u", quote="hello")],
        evidence,
    )
    assert bad == []
