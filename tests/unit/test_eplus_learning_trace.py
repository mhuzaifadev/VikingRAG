"""E+ ONE_ROUND must emit enqueue-shaped Search / EDGE_EXPAND / Read events."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from vikingrag.application.assessment import ScriptedEvidenceAssessor
from vikingrag.application.experience.builder import ExperienceEdgeBuilder
from vikingrag.application.experience.enqueue import _EVENT_MAP
from vikingrag.application.experience.memory_store import InMemoryExperienceStore
from vikingrag.application.experience.trace_sets import extract_trace_uri_sets
from vikingrag.application.orchestration.query import (
    QueryOrchestrator,
    QueryRequest,
    learning_events_from_eplus,
)
from vikingrag.application.search_plus import ExpansionHit, SearchPlusResponse
from vikingrag.domain.models.answer import ExecutionMode
from vikingrag.domain.models.assessment import AssessmentStatus
from vikingrag.domain.models.document import DocumentId, NodeId, NodeType
from vikingrag.domain.models.evidence import EvidenceBundle, RetrievedEvidence
from vikingrag.domain.models.experience import (
    EdgeBuildStatus,
    QueryRun,
    QueryRunRoute,
    QueryRunStatus,
    RetrievalEvent,
    new_query_run_id,
    new_retrieval_event_id,
)
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
    def __init__(self, *, with_expand: bool = True) -> None:
        self._with_expand = with_expand

    async def search(
        self, request: Any, *, ctx: Any = None, trace: Any = None
    ) -> SearchPlusResponse:
        del request, ctx, trace
        doc_id = DocumentId(uuid4())
        node_id = NodeId(uuid4())
        hits = (
            SearchHit(
                uri="viking://seed",
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
        expansions = ()
        if self._with_expand:
            expansions = (
                ExpansionHit(
                    seed_uri="viking://seed",
                    target_uri="viking://edge",
                    edge_id=uuid4(),
                    similarity=0.85,
                    hop=1,
                ),
            )
        return SearchPlusResponse(base=base, expansions=expansions, cold_path=False)


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


def test_learning_events_partition_u_src_u_edge_u_cand() -> None:
    plus = SearchPlusResponse(
        base=SearchResponse(
            query="q",
            query_id=uuid4(),
            embedding_identity=EmbeddingIdentity(
                provider="t", model="t", dimensions=8, version="1"
            ),
            candidates=(
                SearchHit(
                    uri="viking://seed",
                    title=None,
                    node_id=NodeId(uuid4()),
                    document_id=DocumentId(uuid4()),
                    node_type=NodeType.CHUNK,
                    representation_type=RepresentationType.NODE_CONTENT,
                    score=1.0,
                    similarity=1.0,
                    preview="p",
                ),
            ),
            timing=SearchTiming(),
            usage=SearchUsage(),
        ),
        expansions=(
            ExpansionHit(
                seed_uri="viking://seed",
                target_uri="viking://edge",
                edge_id=uuid4(),
                similarity=0.9,
                hop=1,
            ),
        ),
    )
    evidence = EvidenceBundle.from_items(
        [
            RetrievedEvidence(
                evidence_id="e1",
                uri="viking://seed",
                document_id=DocumentId(uuid4()),
                node_id=NodeId(uuid4()),
                text="fee text",
                token_count=2,
                content_hash="h",
                start_offset=0,
                end_offset=8,
                offset_system=OffsetSystem.UNICODE_CODE_POINT,
            )
        ]
    )
    events = learning_events_from_eplus(plus, evidence)
    assert {e["name"] for e in events} >= {"Search", "EDGE_EXPAND", "Read"}
    for e in events:
        assert e["name"] in _EVENT_MAP

    run_id = new_query_run_id()
    domain_events: list[RetrievalEvent] = []
    for i, e in enumerate(events):
        domain_events.append(
            RetrievalEvent(
                id=new_retrieval_event_id(),
                query_run_id=run_id,
                round_no=i,
                event_type=_EVENT_MAP[str(e["name"])],
                arguments=dict(e.get("arguments") or {}),
                result_refs=list(e.get("result_uris") or []),
            )
        )
    sets = extract_trace_uri_sets(domain_events)
    assert "viking://seed" in sets.u_src
    assert "viking://edge" in sets.u_edge
    assert "viking://seed" in sets.u_cand
    assert "viking://edge" not in sets.u_src


@pytest.mark.asyncio
async def test_eplus_one_round_returns_learning_trace_events() -> None:
    orch = QueryOrchestrator(
        search=MagicMock(),
        search_plus=_FakeSearchPlus(),  # type: ignore[arg-type]
        assessor=ScriptedEvidenceAssessor(status=AssessmentStatus.SUFFICIENT),
        read_service=_Read(),
        llm=None,
    )
    resp = await orch.run(
        QueryRequest(query="What is the late fee?", mode=ExecutionMode.VIKINGRAG_E_PLUS)
    )
    assert resp.route is QueryRunRoute.ONE_ROUND
    names = [e["name"] for e in resp.trace_events]
    assert "Search" in names
    assert "Read" in names
    assert "EDGE_EXPAND" in names
    seed = next(e for e in resp.trace_events if e["name"] == "Search")
    assert "viking://seed" in seed["result_uris"]


class _MemRuns:
    def __init__(self) -> None:
        self.runs: dict[object, QueryRun] = {}

    async def create(self, run: QueryRun) -> QueryRun:
        self.runs[run.id] = run
        return run

    async def get(self, run_id: object) -> QueryRun | None:
        return self.runs.get(run_id)

    async def update(self, run: QueryRun) -> QueryRun:
        self.runs[run.id] = run
        return run

    async def claim_pending_edge_jobs(self, *, limit: int = 10) -> list[QueryRun]:
        out: list[QueryRun] = []
        for run in self.runs.values():
            if run.edge_build_status is EdgeBuildStatus.PENDING:
                run.edge_build_status = EdgeBuildStatus.PROCESSING
                out.append(run)
                if len(out) >= limit:
                    break
        return out


class _MemEvents:
    def __init__(self) -> None:
        self.by_run: dict[object, list[RetrievalEvent]] = {}

    async def append(self, event: RetrievalEvent) -> RetrievalEvent:
        self.by_run.setdefault(event.query_run_id, []).append(event)
        return event

    async def list_for_run(self, run_id: object) -> list[RetrievalEvent]:
        return list(self.by_run.get(run_id, []))


@pytest.mark.asyncio
async def test_learning_events_feed_builder_edge() -> None:
    """Citations on a Read target distinct from Search seed → ≥1 experience edge."""
    plus = SearchPlusResponse(
        base=SearchResponse(
            query="late fee",
            query_id=uuid4(),
            embedding_identity=EmbeddingIdentity(
                provider="deterministic", model="test", dimensions=4, version="1"
            ),
            candidates=(
                SearchHit(
                    uri="viking://seed",
                    title=None,
                    node_id=NodeId(uuid4()),
                    document_id=DocumentId(uuid4()),
                    node_type=NodeType.CHUNK,
                    representation_type=RepresentationType.NODE_CONTENT,
                    score=1.0,
                    similarity=1.0,
                    preview="p",
                ),
            ),
            timing=SearchTiming(),
            usage=SearchUsage(),
        ),
        expansions=(),
    )
    evidence = EvidenceBundle.from_items(
        [
            RetrievedEvidence(
                evidence_id="e1",
                uri="viking://target",
                text="The late fee is $25.",
                token_count=6,
            )
        ]
    )
    events = learning_events_from_eplus(plus, evidence)
    runs = _MemRuns()
    mem_events = _MemEvents()
    store = InMemoryExperienceStore()
    identity = EmbeddingIdentity(provider="deterministic", model="test", dimensions=4, version="1")
    run = QueryRun(
        id=new_query_run_id(),
        query_text="late fee",
        status=QueryRunStatus.SUCCEEDED,
        route=QueryRunRoute.ONE_ROUND,
        answer="The late fee is $25.",
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        embedding_identity=identity,
        edge_build_status=EdgeBuildStatus.PROCESSING,
        metadata={"citation_uris": ["viking://target"]},
    )
    await runs.create(run)
    for i, e in enumerate(events):
        await mem_events.append(
            RetrievalEvent(
                id=new_retrieval_event_id(),
                query_run_id=run.id,
                round_no=i,
                event_type=_EVENT_MAP[str(e["name"])],
                arguments=dict(e.get("arguments") or {}),
                result_refs=list(e.get("result_uris") or []),
            )
        )
    builder = ExperienceEdgeBuilder(runs=runs, events=mem_events, edges=store)
    result = await builder.build_for_run(run)
    assert result.skipped is False
    assert len(result.edges_created) >= 1
    assert any(
        e.source_uri == "viking://seed" and e.target_uri == "viking://target"
        for e in result.edges_created
    )
