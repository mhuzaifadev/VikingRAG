"""Paper-shaped acceptance scenarios (deterministic / scripted).

Full Postgres-backed scenarios 1-15 belong in integration; these cover
edge activation, Algorithm 2 sets, cold path, and citation/Unicode invariants
that do not require live services.
"""

from __future__ import annotations

import pytest

from vikingrag.application.experience.activate import should_activate
from vikingrag.application.experience.expand import expand_experience_edges
from vikingrag.application.experience.memory_store import InMemoryExperienceStore
from vikingrag.application.experience.support_select import select_support_uris
from vikingrag.application.experience.trace_sets import extract_trace_uri_sets
from vikingrag.application.grep_primitive import _find_literal_positions
from vikingrag.application.read_primitive import _slice_by_tokens
from vikingrag.domain.models.experience import (
    ExperienceEdge,
    ExperienceEdgeStatus,
    ExperienceExpansionLimits,
    ExperiencePayload,
    RetrievalEvent,
    RetrievalEventType,
    new_experience_edge_id,
    new_experience_payload_id,
    new_query_run_id,
    new_retrieval_event_id,
)
from vikingrag.domain.models.representation import EmbeddingIdentity
from vikingrag.ingestion.tokenization import TikTokenTokenizer


def _id() -> EmbeddingIdentity:
    return EmbeddingIdentity(provider="det", model="t", dimensions=4, version="1")


def _payload(qvec: list[float], text: str = "textbook policy") -> ExperiencePayload:
    return ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text=text,
        query_embedding=qvec,
        embedding_identity=_id(),
        trace_summary="search|read",
        support_uris=("viking://documents/d/nodes/policy",),
    )


def test_scenario_3_paraphrase_activates_different_topic_does_not() -> None:
    payload = _payload([1.0, 0.0, 0.0, 0.0], text="What is the textbook?")
    edge = ExperienceEdge(
        id=new_experience_edge_id(),
        payload_id=payload.id,
        source_uri="viking://documents/d/nodes/header",
        target_uri="viking://documents/d/nodes/policy",
        status=ExperienceEdgeStatus.ACTIVE,
    )
    ok, sim = should_activate(
        current_query_embedding=[0.99, 0.1, 0.0, 0.0],
        current_identity=_id(),
        payload=payload,
        edge=edge,
        gamma=0.8,
    )
    assert ok and sim >= 0.8

    no, _ = should_activate(
        current_query_embedding=[0.0, 1.0, 0.0, 0.0],
        current_identity=_id(),
        payload=payload,
        edge=edge,
        gamma=0.8,
    )
    assert not no


@pytest.mark.asyncio
async def test_scenario_7_cycle_terminates_and_incompatible_identity_never_activates() -> None:
    store = InMemoryExperienceStore()
    p = _payload([1.0, 0.0, 0.0, 0.0])
    a = "viking://documents/d/nodes/a"
    b = "viking://documents/d/nodes/b"
    e1 = ExperienceEdge(
        id=new_experience_edge_id(),
        payload_id=p.id,
        source_uri=a,
        target_uri=b,
        status=ExperienceEdgeStatus.ACTIVE,
    )
    e2 = ExperienceEdge(
        id=new_experience_edge_id(),
        payload_id=p.id,
        source_uri=b,
        target_uri=a,
        status=ExperienceEdgeStatus.ACTIVE,
    )
    await store.create_payload(p)
    await store.create_edges([e1, e2])

    result = await expand_experience_edges(
        seed_uris=(a,),
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        identity=_id(),
        edges=store,
        limits=ExperienceExpansionLimits(gamma=0.5, max_hops=5, max_nodes=10, max_edges=10),
    )
    assert a in result.all_uris
    assert b in result.all_uris
    assert len(result.activated) <= 4

    bad_id = EmbeddingIdentity(provider="other", model="t", dimensions=4, version="1")
    ok, _ = should_activate(
        current_query_embedding=[1.0, 0.0, 0.0, 0.0],
        current_identity=bad_id,
        payload=p,
        edge=e1,
        gamma=0.5,
    )
    assert not ok


def test_scenario_8_support_selection_never_invents_uris() -> None:
    cand = frozenset({"u1", "u2"})
    selected = select_support_uris(cand, citations=("u1", "invented"), max_support=5)
    assert selected == frozenset({"u1"})


@pytest.mark.asyncio
async def test_scenario_10_empty_edge_store_cold_path() -> None:
    store = InMemoryExperienceStore()
    result = await expand_experience_edges(
        seed_uris=("viking://documents/d/nodes/seed",),
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        identity=_id(),
        edges=store,
        limits=ExperienceExpansionLimits(gamma=0.8, max_hops=2, max_nodes=8, max_edges=8),
    )
    assert result.seed_uris == ("viking://documents/d/nodes/seed",)
    assert result.expanded_uris == ()
    assert result.edges_activated == 0


def test_scenario_2_trace_sets_u_src_u_cand() -> None:
    run = new_query_run_id()
    events = [
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run,
            round_no=0,
            event_type=RetrievalEventType.SEARCH,
            result_refs=["viking://documents/d/nodes/header"],
        ),
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run,
            round_no=1,
            event_type=RetrievalEventType.READ,
            arguments={"uri": "viking://documents/d/nodes/policy"},
            result_refs=["viking://documents/d/nodes/policy"],
        ),
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run,
            round_no=1,
            event_type=RetrievalEventType.GREP,
            result_refs=["viking://documents/d/nodes/policy"],
        ),
    ]
    sets = extract_trace_uri_sets(events)
    assert "viking://documents/d/nodes/header" in sets.u_src
    assert "viking://documents/d/nodes/policy" in sets.u_cand


def test_scenario_12_unicode_offsets_and_token_caps() -> None:
    content = "İ policy — café 政策"
    positions = _find_literal_positions(content, "policy", case_sensitive=False)
    assert positions
    s, e = positions[0]
    assert "policy" in content[s:e].casefold()

    tok = TikTokenTokenizer()
    text, end, _ = _slice_by_tokens(content, start_offset=0, max_tokens=3, tokenizer=tok)
    assert tok.count(text) <= 3
    assert end <= len(content)


# Scenario 15 (production rejects scripted) covered in test_auth.py
