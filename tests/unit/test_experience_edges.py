"""Unit tests: Alg 2 construction + Alg 3 activation / Search+ cold path."""

from __future__ import annotations

import pytest

from vikingrag.application.experience.activate import cosine_similarity, should_activate
from vikingrag.application.experience.builder import ExperienceEdgeBuilder
from vikingrag.application.experience.expand import expand_experience_edges
from vikingrag.application.experience.memory_store import InMemoryExperienceStore
from vikingrag.application.experience.support_select import select_support_uris
from vikingrag.application.experience.trace_sets import extract_trace_uri_sets
from vikingrag.domain.models.experience import (
    EdgeBuildStatus,
    ExperienceEdge,
    ExperienceExpansionLimits,
    ExperiencePayload,
    QueryRun,
    QueryRunRoute,
    QueryRunStatus,
    RetrievalEvent,
    RetrievalEventType,
    new_experience_edge_id,
    new_experience_payload_id,
    new_query_run_id,
    new_retrieval_event_id,
)
from vikingrag.domain.models.representation import EmbeddingIdentity


def _identity(dims: int = 4) -> EmbeddingIdentity:
    return EmbeddingIdentity(provider="deterministic", model="test", dimensions=dims, version="1")


def _vec(*vals: float) -> list[float]:
    return list(vals)


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


def test_cosine_similarity_identical() -> None:
    v = _vec(1.0, 0.0, 0.0, 0.0)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_gamma_gate_activates_above_threshold() -> None:
    identity = _identity()
    hist = _vec(1.0, 0.0, 0.0, 0.0)
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text="auth flow",
        query_embedding=hist,
        embedding_identity=identity,
        trace_summary="r0:search",
    )
    edge = ExperienceEdge(
        id=new_experience_edge_id(),
        payload_id=payload.id,
        source_uri="viking://a",
        target_uri="viking://b",
    )
    ok, sim = should_activate(
        current_query_embedding=_vec(0.95, 0.05, 0.0, 0.0),
        current_identity=identity,
        payload=payload,
        edge=edge,
        gamma=0.8,
    )
    assert ok is True
    assert sim >= 0.8


def test_gamma_gate_rejects_below_threshold() -> None:
    identity = _identity()
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text="auth flow",
        query_embedding=_vec(1.0, 0.0, 0.0, 0.0),
        embedding_identity=identity,
        trace_summary="r0:search",
    )
    edge = ExperienceEdge(
        id=new_experience_edge_id(),
        payload_id=payload.id,
        source_uri="viking://a",
        target_uri="viking://b",
    )
    ok, sim = should_activate(
        current_query_embedding=_vec(0.0, 1.0, 0.0, 0.0),
        current_identity=identity,
        payload=payload,
        edge=edge,
        gamma=0.8,
    )
    assert ok is False
    assert sim < 0.8


def test_incompatible_embedding_identity_never_activates() -> None:
    a = _identity(4)
    b = EmbeddingIdentity(provider="other", model="test", dimensions=4, version="1")
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text="q",
        query_embedding=_vec(1.0, 0.0, 0.0, 0.0),
        embedding_identity=a,
        trace_summary="x",
    )
    edge = ExperienceEdge(
        id=new_experience_edge_id(),
        payload_id=payload.id,
        source_uri="viking://a",
        target_uri="viking://b",
    )
    ok, sim = should_activate(
        current_query_embedding=_vec(1.0, 0.0, 0.0, 0.0),
        current_identity=b,
        payload=payload,
        edge=edge,
        gamma=0.5,
    )
    assert ok is False
    assert sim == 0.0


@pytest.mark.asyncio
async def test_cycle_termination_in_expansion() -> None:
    store = InMemoryExperienceStore()
    identity = _identity()
    qvec = _vec(1.0, 0.0, 0.0, 0.0)
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text="cycle",
        query_embedding=qvec,
        embedding_identity=identity,
        trace_summary="cycle",
    )
    await store.create_payload(payload)
    # A → B → A cycle
    await store.create_edges(
        [
            ExperienceEdge(
                id=new_experience_edge_id(),
                payload_id=payload.id,
                source_uri="viking://a",
                target_uri="viking://b",
            ),
            ExperienceEdge(
                id=new_experience_edge_id(),
                payload_id=payload.id,
                source_uri="viking://b",
                target_uri="viking://a",
            ),
        ]
    )
    result = await expand_experience_edges(
        seed_uris=["viking://a"],
        query_embedding=qvec,
        identity=identity,
        edges=store,
        limits=ExperienceExpansionLimits(gamma=0.5, max_hops=5, max_nodes=10, max_edges=20),
    )
    assert "viking://b" in result.expanded_uris
    # Cycle must not re-expand A (already visited as seed)
    assert result.expanded_uris.count("viking://a") == 0
    assert len(set(result.all_uris)) == 2


@pytest.mark.asyncio
async def test_empty_edge_store_cold_path() -> None:
    store = InMemoryExperienceStore()
    identity = _identity()
    result = await expand_experience_edges(
        seed_uris=["viking://seed"],
        query_embedding=_vec(1.0, 0.0, 0.0, 0.0),
        identity=identity,
        edges=store,
        limits=ExperienceExpansionLimits(gamma=0.8),
    )
    assert result.expanded_uris == ()
    assert result.activated == ()
    assert result.seed_uris == ("viking://seed",)
    assert await store.count_active() == 0


@pytest.mark.asyncio
async def test_duplicate_active_edge_suppression() -> None:
    runs = _MemRuns()
    events = _MemEvents()
    store = InMemoryExperienceStore()
    identity = _identity()
    qvec = _vec(1.0, 0.0, 0.0, 0.0)

    run = QueryRun(
        id=new_query_run_id(),
        query_text="oauth tokens",
        status=QueryRunStatus.SUCCEEDED,
        route=QueryRunRoute.ESCALATED,
        answer="see viking://tgt",
        query_embedding=qvec,
        embedding_identity=identity,
        edge_build_status=EdgeBuildStatus.PROCESSING,
        metadata={"citation_uris": ["viking://tgt"]},
    )
    await runs.create(run)
    await events.append(
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run.id,
            round_no=0,
            event_type=RetrievalEventType.SEARCH,
            result_refs=["viking://src"],
        )
    )
    await events.append(
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run.id,
            round_no=1,
            event_type=RetrievalEventType.READ,
            arguments={"uri": "viking://tgt"},
            result_refs=["viking://tgt"],
        )
    )

    # Pre-seed a duplicate active edge
    pre_payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text="prior",
        query_embedding=qvec,
        embedding_identity=identity,
        trace_summary="prior",
    )
    await store.create_payload(pre_payload)
    await store.create_edges(
        [
            ExperienceEdge(
                id=new_experience_edge_id(),
                payload_id=pre_payload.id,
                source_uri="viking://src",
                target_uri="viking://tgt",
            )
        ]
    )

    builder = ExperienceEdgeBuilder(runs=runs, events=events, edges=store)
    result = await builder.build_for_run(run)
    assert result.skipped is False
    assert result.edges_suppressed == 1
    assert result.edges_created == ()
    assert await store.count_active() == 1


@pytest.mark.asyncio
async def test_builder_never_learns_from_failed_or_abstained() -> None:
    runs = _MemRuns()
    events = _MemEvents()
    store = InMemoryExperienceStore()
    identity = _identity()

    for status, route in (
        (QueryRunStatus.FAILED, QueryRunRoute.FAILED),
        (QueryRunStatus.ABSTAINED, QueryRunRoute.ABSTAINED),
    ):
        run = QueryRun(
            id=new_query_run_id(),
            query_text="no learn",
            status=status,
            route=route,
            query_embedding=_vec(1.0, 0.0, 0.0, 0.0),
            embedding_identity=identity,
        )
        await runs.create(run)
        builder = ExperienceEdgeBuilder(runs=runs, events=events, edges=store)
        result = await builder.build_for_run(run)
        assert result.skipped is True
        assert run.edge_build_status is EdgeBuildStatus.SKIPPED

    assert await store.count_active() == 0


@pytest.mark.asyncio
async def test_builder_idempotent_and_excludes_u_edge_and_self() -> None:
    runs = _MemRuns()
    events = _MemEvents()
    store = InMemoryExperienceStore()
    identity = _identity()
    qvec = _vec(1.0, 0.0, 0.0, 0.0)
    run = QueryRun(
        id=new_query_run_id(),
        query_text="hierarchy",
        status=QueryRunStatus.SUCCEEDED,
        route=QueryRunRoute.ESCALATED,
        answer="viking://leaf",
        query_embedding=qvec,
        embedding_identity=identity,
        edge_build_status=EdgeBuildStatus.PROCESSING,
        metadata={"citation_uris": ["viking://leaf", "viking://via_edge"]},
    )
    await runs.create(run)
    await events.append(
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run.id,
            round_no=0,
            event_type=RetrievalEventType.SEARCH,
            result_refs=["viking://src", "viking://leaf"],
        )
    )
    await events.append(
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run.id,
            round_no=1,
            event_type=RetrievalEventType.READ,
            arguments={"uri": "viking://leaf"},
            result_refs=["viking://leaf"],
        )
    )
    await events.append(
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run.id,
            round_no=1,
            event_type=RetrievalEventType.EDGE_EXPAND,
            result_refs=["viking://via_edge"],
        )
    )
    await events.append(
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run.id,
            round_no=2,
            event_type=RetrievalEventType.READ,
            arguments={"uri": "viking://via_edge"},
            result_refs=["viking://via_edge"],
        )
    )

    sets = extract_trace_uri_sets(await events.list_for_run(run.id))
    assert "viking://src" in sets.u_src
    assert "viking://via_edge" in sets.u_edge

    builder = ExperienceEdgeBuilder(runs=runs, events=events, edges=store)
    first = await builder.build_for_run(run)
    assert first.skipped is False
    # Self-edge src=leaf→leaf forbidden; via_edge excluded from U_tgt
    for edge in first.edges_created:
        assert edge.source_uri != edge.target_uri
        assert edge.target_uri != "viking://via_edge"
    assert any(e.target_uri == "viking://leaf" for e in first.edges_created)

    second = await builder.build_for_run(run)
    assert second.skipped is True
    assert second.skip_reason == "payload_already_exists"


def test_support_select_bounded_and_citation_preferring() -> None:
    cand = frozenset({"viking://a", "viking://b", "viking://c"})
    selected = select_support_uris(
        cand, citations=["viking://b", "viking://missing"], max_support=2
    )
    assert selected == frozenset({"viking://b"})


def test_self_edge_rejected_at_domain() -> None:
    with pytest.raises(ValueError, match="self-edges"):
        ExperienceEdge(
            id=new_experience_edge_id(),
            payload_id=new_experience_payload_id(),
            source_uri="viking://x",
            target_uri="viking://x",
        )


@pytest.mark.asyncio
async def test_invalidate_by_uris() -> None:
    store = InMemoryExperienceStore()
    identity = _identity()
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text="q",
        query_embedding=_vec(1.0, 0.0, 0.0, 0.0),
        embedding_identity=identity,
        trace_summary="t",
    )
    await store.create_payload(payload)
    await store.create_edges(
        [
            ExperienceEdge(
                id=new_experience_edge_id(),
                payload_id=payload.id,
                source_uri="viking://keep",
                target_uri="viking://gone",
            ),
            ExperienceEdge(
                id=new_experience_edge_id(),
                payload_id=payload.id,
                source_uri="viking://other",
                target_uri="viking://ok",
            ),
        ]
    )
    n = await store.invalidate_by_uris({"viking://gone"})
    assert n == 1
    assert await store.count_active() == 1
