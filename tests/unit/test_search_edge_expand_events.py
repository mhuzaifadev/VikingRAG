"""Search+ tool path must emit separate SEARCH and EDGE_EXPAND events."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from vikingrag.application.agent.tool_executor import RetrievalToolExecutor
from vikingrag.application.budget import RetrievalContext
from vikingrag.application.experience.trace_sets import extract_trace_uri_sets
from vikingrag.application.search_plus import ExpansionHit, SearchPlusResponse
from vikingrag.domain.models.document import DocumentId, NodeId, NodeType
from vikingrag.domain.models.experience import (
    RetrievalEvent,
    RetrievalEventType,
    new_query_run_id,
    new_retrieval_event_id,
)
from vikingrag.domain.models.representation import (
    EmbeddingIdentity,
    RepresentationType,
    SearchHit,
    SearchResponse,
    SearchTiming,
    SearchUsage,
)
from vikingrag.providers.llm.base import ToolCall


class _Plus:
    async def search(
        self, request: Any, *, ctx: Any = None, trace: Any = None
    ) -> SearchPlusResponse:
        del request, ctx, trace
        doc = DocumentId(uuid4())
        seed = SearchHit(
            uri="viking://seed",
            title=None,
            node_id=NodeId(uuid4()),
            document_id=doc,
            node_type=NodeType.CHUNK,
            representation_type=RepresentationType.NODE_CONTENT,
            score=0.9,
            similarity=0.9,
            preview="s",
        )
        base = SearchResponse(
            query="q",
            query_id=uuid4(),
            embedding_identity=EmbeddingIdentity(
                provider="t", model="t", dimensions=8, version="1"
            ),
            candidates=(seed,),
            timing=SearchTiming(),
            usage=SearchUsage(),
        )
        expansions = (
            ExpansionHit(
                seed_uri="viking://seed",
                target_uri="viking://edge",
                edge_id=uuid4(),
                similarity=0.8,
                hop=1,
            ),
        )
        return SearchPlusResponse(base=base, expansions=expansions, cold_path=False)


@pytest.mark.asyncio
async def test_search_plus_emits_edge_expand_event() -> None:
    ex = RetrievalToolExecutor(search_plus=_Plus(), use_search_plus=True)
    ctx = RetrievalContext.create()
    call = ToolCall(id="1", name="Search", arguments={"query": "fees"}, arguments_valid=True)
    rec = await ex.execute(call, ctx=ctx)
    assert rec.name == "Search"
    assert rec.result_uris == ("viking://seed",)
    names = [e.name for e in ex.state.events]
    assert "Search" in names
    assert "EDGE_EXPAND" in names
    expand = next(e for e in ex.state.events if e.name == "EDGE_EXPAND")
    assert "viking://edge" in expand.result_uris

    run_id = new_query_run_id()
    events: list[RetrievalEvent] = []
    for i, e in enumerate(ex.state.events):
        if e.name == "EDGE_EXPAND":
            et = RetrievalEventType.EDGE_EXPAND
        elif e.name == "Search":
            et = RetrievalEventType.SEARCH
        else:
            continue
        events.append(
            RetrievalEvent(
                id=new_retrieval_event_id(),
                query_run_id=run_id,
                round_no=i,
                event_type=et,
                arguments=dict(e.arguments),
                result_refs=list(e.result_uris),
            )
        )
    sets = extract_trace_uri_sets(events)
    assert "viking://seed" in sets.u_src
    assert "viking://edge" in sets.u_edge
    assert "viking://edge" not in sets.u_src
