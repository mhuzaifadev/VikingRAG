"""CI unit: enqueue-shaped events → builder edges → Search+ expansion sees targets."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.application.experience.builder import ExperienceEdgeBuilder
from vikingrag.application.experience.enqueue import _EVENT_MAP
from vikingrag.application.experience.expand import expand_experience_edges
from vikingrag.application.experience.memory_store import InMemoryExperienceStore
from vikingrag.application.orchestration.query import learning_events_from_eplus
from vikingrag.application.search_plus import SearchPlusResponse
from vikingrag.domain.models.document import DocumentId, NodeId, NodeType
from vikingrag.domain.models.evidence import EvidenceBundle, RetrievedEvidence
from vikingrag.domain.models.experience import (
    EdgeBuildStatus,
    ExperienceExpansionLimits,
    QueryRun,
    QueryRunRoute,
    QueryRunStatus,
    RetrievalEvent,
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


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(provider="deterministic", model="test", dimensions=4, version="1")


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


async def _enqueue_in_memory(
    *,
    runs: _MemRuns,
    events: _MemEvents,
    query_text: str,
    answer: str,
    trace_events: list[dict[object, object]] | tuple[dict[str, object], ...],
    citation_uris: list[str],
    query_embedding: list[float],
    identity: EmbeddingIdentity,
) -> QueryRun:
    """In-memory stand-in for ``enqueue_experience_learning`` (no Postgres)."""
    from vikingrag.application.orchestration.query import mark_run_for_edge_build

    run = QueryRun(
        id=new_query_run_id(),
        query_text=query_text,
        status=QueryRunStatus.SUCCEEDED,
        route=QueryRunRoute.ONE_ROUND,
        answer=answer,
        query_embedding=query_embedding,
        embedding_identity=identity,
        metadata={"citation_uris": list(citation_uris)},
    )
    mark_run_for_edge_build(run)
    await runs.create(run)
    for i, ev in enumerate(trace_events):
        name = str(ev.get("name") or "")
        etype = _EVENT_MAP.get(name)
        if etype is None:
            continue
        refs = ev.get("result_uris") or []
        await events.append(
            RetrievalEvent(
                id=new_retrieval_event_id(),
                query_run_id=run.id,
                round_no=i,
                event_type=etype,
                arguments=dict(ev.get("arguments") or {}),  # type: ignore[arg-type]
                result_refs=[str(r) for r in refs],  # type: ignore[union-attr]
            )
        )
    return run


@pytest.mark.asyncio
async def test_e2e_query_enqueue_edges_search_plus_expand() -> None:
    identity = _identity()
    qvec = [1.0, 0.0, 0.0, 0.0]

    # 1) Enqueue-shaped events from E+ learning synthesizer (Search A, Read B)
    plus = SearchPlusResponse(
        base=SearchResponse(
            query="What is the late fee?",
            query_id=uuid4(),
            embedding_identity=identity,
            candidates=(
                SearchHit(
                    uri="viking://A",
                    title=None,
                    node_id=NodeId(uuid4()),
                    document_id=DocumentId(uuid4()),
                    node_type=NodeType.CHUNK,
                    representation_type=RepresentationType.NODE_CONTENT,
                    score=1.0,
                    similarity=1.0,
                    preview="seed",
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
                uri="viking://B",
                text="The late fee is $25.",
                token_count=6,
            )
        ]
    )
    trace_events = learning_events_from_eplus(plus, evidence)
    assert any(e["name"] == "Search" for e in trace_events)
    assert any(e["name"] == "Read" for e in trace_events)

    runs = _MemRuns()
    mem_events = _MemEvents()
    store = InMemoryExperienceStore()

    # 2) Enqueue → claim → builder → edge A→B
    run = await _enqueue_in_memory(
        runs=runs,
        events=mem_events,
        query_text="What is the late fee?",
        answer="The late fee is $25.",
        trace_events=trace_events,
        citation_uris=["viking://B"],
        query_embedding=qvec,
        identity=identity,
    )
    assert run.edge_build_status is EdgeBuildStatus.PENDING

    claimed = await runs.claim_pending_edge_jobs(limit=1)
    assert len(claimed) == 1
    builder = ExperienceEdgeBuilder(runs=runs, events=mem_events, edges=store)
    built = await builder.build_for_run(claimed[0])
    assert built.skipped is False
    assert any(
        e.source_uri == "viking://A" and e.target_uri == "viking://B" for e in built.edges_created
    )
    assert await store.count_active() >= 1

    # 3) Next Search+ activation includes B in expansions
    result = await expand_experience_edges(
        seed_uris=["viking://A"],
        query_embedding=qvec,
        identity=identity,
        edges=store,
        limits=ExperienceExpansionLimits(gamma=0.5, max_hops=2, max_nodes=10, max_edges=20),
    )
    assert "viking://B" in result.expanded_uris
    assert result.edges_activated >= 1
