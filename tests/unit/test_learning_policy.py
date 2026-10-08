"""Learning policy gates enqueue and worker; scoped expansion hide excluded docs."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.application.experience.expand import expand_experience_edges, uri_in_scope
from vikingrag.application.experience.memory_store import InMemoryExperienceStore
from vikingrag.application.experience.policy import (
    apply_learning_policy,
    should_enqueue_edge_build,
    should_persist_run,
    worker_may_build,
)
from vikingrag.application.experience.snapshots import InMemorySnapshotStore
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.experience import (
    EdgeBuildStatus,
    ExperienceEdge,
    ExperienceExpansionLimits,
    ExperiencePayload,
    LearningPolicy,
    QueryRun,
    QueryRunRoute,
    QueryRunStatus,
    SnapshotStatus,
    new_experience_edge_id,
    new_experience_payload_id,
    new_query_run_id,
)
from vikingrag.domain.models.representation import EmbeddingIdentity
from vikingrag.domain.uri.object_uri import ObjectURI, ObjectURIKind, ObjectURIParser


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(provider="deterministic", model="t", dimensions=4, version="1")


def test_policy_semantics() -> None:
    assert should_persist_run(LearningPolicy.OFF) is False
    assert should_persist_run(LearningPolicy.FROZEN) is True
    assert should_enqueue_edge_build(LearningPolicy.LEARN) is True
    assert should_enqueue_edge_build(LearningPolicy.FROZEN) is False
    assert should_enqueue_edge_build(LearningPolicy.RECORD_ONLY) is False


def test_apply_learning_policy_frozen_skips_build() -> None:
    run = QueryRun(
        id=new_query_run_id(),
        query_text="q",
        status=QueryRunStatus.SUCCEEDED,
        route=QueryRunRoute.ONE_ROUND,
    )
    apply_learning_policy(run, LearningPolicy.FROZEN)
    assert run.edge_build_status is EdgeBuildStatus.SKIPPED
    assert worker_may_build(run) is False


def test_uri_scope_fail_closed() -> None:
    doc_a = DocumentId(uuid4())
    doc_b = DocumentId(uuid4())
    uri_a = ObjectURIParser.build(
        ObjectURI(document_id=doc_a, path_components=(), kind=ObjectURIKind.DOCUMENT)
    )
    assert uri_in_scope(uri_a, frozenset({doc_a})) is True
    assert uri_in_scope(uri_a, frozenset({doc_b})) is False
    assert uri_in_scope("viking://opaque", frozenset({doc_a})) is False
    assert uri_in_scope(uri_a, None) is True


@pytest.mark.asyncio
async def test_expansion_respects_document_scope() -> None:
    store = InMemoryExperienceStore()
    identity = _identity()
    qvec = [1.0, 0.0, 0.0, 0.0]
    doc_a = DocumentId(uuid4())
    doc_b = DocumentId(uuid4())
    uri_a = ObjectURIParser.build(
        ObjectURI(document_id=doc_a, path_components=(uuid4(),), kind=ObjectURIKind.DIRECTORY)
    )
    uri_b = ObjectURIParser.build(
        ObjectURI(document_id=doc_b, path_components=(uuid4(),), kind=ObjectURIKind.DIRECTORY)
    )
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text="fee",
        query_embedding=qvec,
        embedding_identity=identity,
        trace_summary="t",
    )
    await store.create_payload(payload)
    await store.create_edges(
        [
            ExperienceEdge(
                id=new_experience_edge_id(),
                payload_id=payload.id,
                source_uri=uri_a,
                target_uri=uri_b,
            )
        ]
    )
    result = await expand_experience_edges(
        seed_uris=[uri_a],
        query_embedding=qvec,
        identity=identity,
        edges=store,
        limits=ExperienceExpansionLimits(gamma=0.5),
        permitted_document_ids=frozenset({doc_a}),
    )
    assert uri_b not in result.expanded_uris
    assert result.edges_activated == 0


@pytest.mark.asyncio
async def test_snapshot_freeze_and_rollback() -> None:
    store = InMemoryExperienceStore()
    identity = _identity()
    qvec = [1.0, 0.0, 0.0, 0.0]
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=new_query_run_id(),
        query_text="q",
        query_embedding=qvec,
        embedding_identity=identity,
        trace_summary="t",
    )
    await store.create_payload(payload)
    e1 = ExperienceEdge(
        id=new_experience_edge_id(),
        payload_id=payload.id,
        source_uri="viking://objects/" + str(uuid4()),
        target_uri="viking://objects/" + str(uuid4()) + "/a",
    )
    await store.create_edges([e1])
    snaps = InMemorySnapshotStore()
    snap = await snaps.create(name="warm", edges=[e1])
    await snaps.freeze(snap.id)
    assert (await snaps.get(snap.id)).status is SnapshotStatus.FROZEN  # type: ignore[union-attr]

    class _Src:
        async def list_active_edges(self):
            return [e for e in store.edges.values() if e.status.value == "active"]

        async def set_edge_status(self, edge_id, status):
            store.edges[edge_id].status = status

    e2 = ExperienceEdge(
        id=new_experience_edge_id(),
        payload_id=payload.id,
        source_uri="viking://objects/" + str(uuid4()),
        target_uri="viking://objects/" + str(uuid4()) + "/b",
    )
    await store.create_edges([e2])
    await snaps.rollback(snap.id, edges=_Src())
    assert store.edges[e1.id].status.value == "active"
    assert store.edges[e2.id].status.value == "superseded"
