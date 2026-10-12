"""Builder skips missing endpoints and stamps revisions when a resolver is set."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

import pytest

from vikingrag.application.experience.builder import ExperienceEdgeBuilder
from vikingrag.application.experience.memory_store import InMemoryExperienceStore
from vikingrag.domain.models.document import AbstractStatus, DocumentId, NodeId, NodeType
from vikingrag.domain.models.experience import (
    EdgeBuildStatus,
    QueryRun,
    QueryRunRoute,
    QueryRunStatus,
    RetrievalEvent,
    RetrievalEventType,
    new_query_run_id,
    new_retrieval_event_id,
)
from vikingrag.domain.models.node import DocumentNode
from vikingrag.domain.models.representation import EmbeddingIdentity


@dataclass
class _MemRuns:
    runs: dict

    async def update(self, run: QueryRun) -> QueryRun:
        self.runs[run.id] = run
        return run


@dataclass
class _MemEvents:
    by_run: dict

    async def list_for_run(self, run_id):  # type: ignore[no-untyped-def]
        return list(self.by_run.get(run_id, []))


@dataclass
class _NodeMap:
    by_uri: dict[str, DocumentNode]

    async def get_by_uri(self, uri: str) -> DocumentNode | None:
        return self.by_uri.get(uri)


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(provider="deterministic", model="t", dimensions=4, version="1")


def _node(uri: str, content_hash: str) -> DocumentNode:
    nid = NodeId(uuid4())
    did = DocumentId(uuid4())
    return DocumentNode(
        id=nid,
        document_id=did,
        parent_id=None,
        node_type=NodeType.SECTION,
        title="s",
        ordinal=0,
        depth=1,
        uri=uri,
        content="x",
        content_hash=content_hash,
        token_count=1,
        path_ids=(uuid4(),),
        abstract_status=AbstractStatus.NONE,
    )


@pytest.mark.asyncio
async def test_builder_stamps_revisions_when_nodes_live() -> None:
    store = InMemoryExperienceStore()
    run_id = new_query_run_id()
    src = f"viking://documents/{uuid4()}/nodes/{uuid4()}"
    tgt = f"viking://documents/{uuid4()}/nodes/{uuid4()}"
    run = QueryRun(
        id=run_id,
        query_text="q",
        status=QueryRunStatus.SUCCEEDED,
        route=QueryRunRoute.ONE_ROUND,
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        embedding_identity=_identity(),
        edge_build_status=EdgeBuildStatus.PENDING,
        metadata={"citation_uris": [tgt]},
    )
    events = [
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run_id,
            round_no=0,
            event_type=RetrievalEventType.SEARCH,
            result_refs=[src],
        ),
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run_id,
            round_no=1,
            event_type=RetrievalEventType.READ,
            result_refs=[tgt],
        ),
    ]
    nodes = _NodeMap(
        {
            src: _node(src, "hash-src"),
            tgt: _node(tgt, "hash-tgt"),
        }
    )
    builder = ExperienceEdgeBuilder(
        runs=_MemRuns({run_id: run}),  # type: ignore[arg-type]
        events=_MemEvents({run_id: events}),  # type: ignore[arg-type]
        edges=store,
        nodes=nodes,
    )
    result = await builder.build_for_run(run)
    assert result.skipped is False
    assert len(result.edges_created) == 1
    edge = result.edges_created[0]
    assert edge.source_revision == "hash-src"
    assert edge.target_revision == "hash-tgt"


@pytest.mark.asyncio
async def test_builder_skips_missing_endpoints() -> None:
    store = InMemoryExperienceStore()
    run_id = new_query_run_id()
    src = f"viking://documents/{uuid4()}/nodes/{uuid4()}"
    tgt = f"viking://documents/{uuid4()}/nodes/{uuid4()}"
    run = QueryRun(
        id=run_id,
        query_text="q",
        status=QueryRunStatus.SUCCEEDED,
        route=QueryRunRoute.ONE_ROUND,
        query_embedding=[1.0, 0.0, 0.0, 0.0],
        embedding_identity=_identity(),
        edge_build_status=EdgeBuildStatus.PENDING,
        metadata={"citation_uris": [tgt]},
    )
    events = [
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run_id,
            round_no=0,
            event_type=RetrievalEventType.SEARCH,
            result_refs=[src],
        ),
        RetrievalEvent(
            id=new_retrieval_event_id(),
            query_run_id=run_id,
            round_no=1,
            event_type=RetrievalEventType.READ,
            result_refs=[tgt],
        ),
    ]
    builder = ExperienceEdgeBuilder(
        runs=_MemRuns({run_id: run}),  # type: ignore[arg-type]
        events=_MemEvents({run_id: events}),  # type: ignore[arg-type]
        edges=store,
        nodes=_NodeMap({}),  # nothing live
    )
    result = await builder.build_for_run(run)
    assert result.skipped is True
    assert result.skip_reason == "endpoints_missing_after_document_change"
    assert run.edge_build_status is EdgeBuildStatus.SKIPPED
