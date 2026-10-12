"""Postgres integration: REPLACE/DELETE retire experience edges transactionally."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from vikingrag.application.documents import DocumentService
from vikingrag.application.experience.invalidate import (
    prepare_document_experience_retirement_session,
)
from vikingrag.domain.models.document import DocumentId, DocumentStatus, IngestionStrategy
from vikingrag.domain.models.experience import (
    EdgeBuildStatus,
    ExperienceEdge,
    ExperienceEdgeStatus,
    ExperiencePayload,
    LearningPolicy,
    QueryRun,
    QueryRunRoute,
    QueryRunStatus,
    new_experience_edge_id,
    new_experience_payload_id,
    new_query_run_id,
)
from vikingrag.domain.models.representation import EmbeddingIdentity
from vikingrag.domain.uri import VikingURIParser
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.models import (
    DEFAULT_VECTOR_DIMENSIONS,
    ExperienceEdgeRow,
)
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.experience import (
    SqlExperienceEdgeRepository,
    SqlQueryRunRepository,
)
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.infrastructure.object_store.local import LocalObjectStore
from vikingrag.ingestion.pipeline import IngestionPipeline
from vikingrag.ingestion.types import IngestionInput
from vikingrag.settings.config import IngestionSettings

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _identity() -> EmbeddingIdentity:
    return EmbeddingIdentity(
        provider="deterministic",
        model="test",
        dimensions=DEFAULT_VECTOR_DIMENSIONS,
        version="1",
    )


def _unit_embedding() -> list[float]:
    """pgvector columns are fixed at DEFAULT_VECTOR_DIMENSIONS (1536)."""
    vec = [0.0] * DEFAULT_VECTOR_DIMENSIONS
    vec[0] = 1.0
    return vec


def _md(text: str) -> bytes:
    return text.encode("utf-8")


def _distinct_target(tree: list, root: str) -> str:
    """Pick a non-root node URI, or a synthetic distinct URI under the document root.

    Small markdown fixtures may only materialize a document-root node; self-edges
    are forbidden by the ExperienceEdge domain model.
    """
    for node in tree:
        uri = getattr(node, "uri", None)
        if uri and uri != root:
            return str(uri)
    return f"{root}/nodes/{uuid4()}"


async def _seed_edge(
    session,
    *,
    source_uri: str,
    target_uri: str,
    document_id: DocumentId | None = None,
) -> ExperienceEdge:
    if source_uri == target_uri:
        target_uri = f"{source_uri}/nodes/{uuid4()}"
    runs = SqlQueryRunRepository(session)
    edges = SqlExperienceEdgeRepository(session)
    emb = _unit_embedding()
    identity = _identity()
    run = QueryRun(
        id=new_query_run_id(),
        query_text="seed",
        status=QueryRunStatus.SUCCEEDED,
        route=QueryRunRoute.ONE_ROUND,
        query_embedding=emb,
        embedding_identity=identity,
        edge_build_status=EdgeBuildStatus.COMPLETED,
        learning_policy=LearningPolicy.LEARN,
        document_ids=(document_id,) if document_id else (),
        metadata={"citation_uris": [target_uri]},
    )
    await runs.create(run)
    payload = ExperiencePayload(
        id=new_experience_payload_id(),
        query_run_id=run.id,
        query_text="seed",
        query_embedding=list(emb),
        embedding_identity=identity,
        trace_summary="seed",
        support_uris=(target_uri,),
    )
    await edges.create_payload(payload)
    edge = ExperienceEdge(
        id=new_experience_edge_id(),
        payload_id=payload.id,
        source_uri=source_uri,
        target_uri=target_uri,
        status=ExperienceEdgeStatus.ACTIVE,
    )
    created = await edges.create_edges([edge])
    return created[0]


async def _ingest_service(
    database: Database,
    store: LocalObjectStore,
    *,
    content: bytes,
    filename: str = "doc.md",
    external_id: str | None = None,
    strategy: IngestionStrategy = IngestionStrategy.CREATE_VERSION,
) -> DocumentId:
    svc = DocumentService(
        database=database,
        object_store=store,
        ingestion_settings=IngestionSettings(),
    )
    result = await svc.ingest(
        filename=filename,
        mime_type="text/markdown",
        content=content,
        external_id=external_id,
        strategy=strategy,
    )
    return result.document_id


@pytest.mark.asyncio
async def test_replace_by_external_id_invalidates_incident_edges(
    database: Database, tmp_path: Path
) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    doc_id = await _ingest_service(
        database,
        store,
        content=_md("# A\n\nHello world.\n"),
        external_id="policy-a",
        strategy=IngestionStrategy.CREATE_VERSION,
    )
    root = VikingURIParser.build_document_uri(doc_id)
    other_doc = DocumentId(uuid4())
    other_uri = VikingURIParser.build_document_uri(other_doc)

    async with database.session() as session:
        nodes = SqlNodeRepository(session)
        tree = await nodes.list_by_document(doc_id)
        section_uri = _distinct_target(tree, root)
        edge_out = await _seed_edge(
            session, source_uri=root, target_uri=section_uri, document_id=doc_id
        )
        edge_in = await _seed_edge(
            session, source_uri=other_uri, target_uri=section_uri, document_id=doc_id
        )
        edge_unrelated = await _seed_edge(
            session, source_uri=other_uri, target_uri=other_uri + "/x"
        )
        edge_ids = (edge_out.id, edge_in.id, edge_unrelated.id)

    # REPLACE via external_id through DocumentService
    new_id = await _ingest_service(
        database,
        store,
        content=_md("# A\n\nReplaced body.\n"),
        external_id="policy-a",
        strategy=IngestionStrategy.REPLACE,
    )
    assert new_id == doc_id

    async with database.session() as session:
        for eid, expect in (
            (edge_ids[0], ExperienceEdgeStatus.INVALIDATED),
            (edge_ids[1], ExperienceEdgeStatus.INVALIDATED),
            (edge_ids[2], ExperienceEdgeStatus.ACTIVE),
        ):
            row = await session.get(ExperienceEdgeRow, eid)
            assert row is not None
            assert row.status == expect.value
        nodes = SqlNodeRepository(session)
        tree = await nodes.list_by_document(doc_id)
        assert VikingURIParser.build_document_uri(doc_id) == root
        assert len(tree) >= 1
        # Prior section URI should be gone after hierarchy rebuild
        assert section_uri not in {n.uri for n in tree if n.uri != root}
        docs = SqlDocumentRepository(session)
        record = await docs.get(doc_id)
        assert record is not None
        assert record.content_hash
        assert f"/{record.content_hash}/" in (record.object_key or "")
        assert await store.exists(record.object_key or "")


@pytest.mark.asyncio
async def test_replace_by_content_hash_without_external_id(
    database: Database, tmp_path: Path
) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    content = _md("# HashDoc\n\nOriginal.\n")
    doc_id = await _ingest_service(
        database,
        store,
        content=content,
        filename="hash.md",
        strategy=IngestionStrategy.CREATE_VERSION,
    )
    root = VikingURIParser.build_document_uri(doc_id)
    async with database.session() as session:
        nodes = SqlNodeRepository(session)
        tree = await nodes.list_by_document(doc_id)
        tgt = _distinct_target(tree, root)
        edge = await _seed_edge(session, source_uri=root, target_uri=tgt, document_id=doc_id)
        edge_id = edge.id

    # REPLACE resolved by content hash of the *incoming* content matching existing
    # Note: REPLACE by content_hash finds the existing doc with that hash, then replaces
    # with new content — so first find same hash, then write different body.
    # Pipeline resolve uses loaded.content_hash of inbound; for REPLACE without
    # external_id it looks up by content hash. To hit the existing doc we must
    # send the same hash first... Actually REPLACE lookup uses inbound hash, so
    # to replace a document you typically use external_id. Content-hash REPLACE
    # means: find doc with same hash as inbound, then replace its hierarchy with
    # that same content (re-ingest). Use identical content for resolve, which
    # rebuilds hierarchy and must still invalidate.
    await _ingest_service(
        database,
        store,
        content=content,
        filename="hash.md",
        strategy=IngestionStrategy.REPLACE,
    )
    async with database.session() as session:
        row = await session.get(ExperienceEdgeRow, edge_id)
        assert row is not None
        assert row.status == ExperienceEdgeStatus.INVALIDATED.value


@pytest.mark.asyncio
async def test_direct_pipeline_replace_invalidates(database: Database, tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    content = _md("# Pipe\n\nOne.\n")
    async with database.session() as session:
        docs = SqlDocumentRepository(session)
        nodes = SqlNodeRepository(session)
        pipeline = IngestionPipeline(
            documents=docs, nodes=nodes, object_store=store, session=session
        )
        result = await pipeline.run(
            IngestionInput(
                filename="pipe.md",
                mime_type="text/markdown",
                content=content,
                external_id="pipe-1",
                strategy=IngestionStrategy.CREATE_VERSION,
            )
        )
        doc_id = result.document_id
        root = VikingURIParser.build_document_uri(doc_id)
        tree = await nodes.list_by_document(doc_id)
        tgt = _distinct_target(tree, root)
        edge = await _seed_edge(session, source_uri=root, target_uri=tgt, document_id=doc_id)
        edge_id = edge.id

    async with database.session() as session:
        docs = SqlDocumentRepository(session)
        nodes = SqlNodeRepository(session)
        pipeline = IngestionPipeline(
            documents=docs, nodes=nodes, object_store=store, session=session
        )
        await pipeline.run(
            IngestionInput(
                filename="pipe.md",
                mime_type="text/markdown",
                content=_md("# Pipe\n\nTwo.\n"),
                external_id="pipe-1",
                strategy=IngestionStrategy.REPLACE,
            )
        )

    async with database.session() as session:
        row = await session.get(ExperienceEdgeRow, edge_id)
        assert row is not None
        assert row.status == ExperienceEdgeStatus.INVALIDATED.value


@pytest.mark.asyncio
async def test_failed_replace_preserves_hierarchy_bytes_and_edges(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    content = _md("# Fail\n\nBody.\n")
    doc_id = await _ingest_service(
        database,
        store,
        content=content,
        external_id="fail-1",
        strategy=IngestionStrategy.CREATE_VERSION,
    )
    async with database.session() as session:
        docs = SqlDocumentRepository(session)
        record = await docs.get(doc_id)
        assert record is not None
        old_key = record.object_key
        old_hash = record.content_hash
        nodes = SqlNodeRepository(session)
        tree_before = await nodes.list_by_document(doc_id)
        uris_before = {n.uri for n in tree_before}
        root = VikingURIParser.build_document_uri(doc_id)
        tgt = next((u for u in uris_before if u != root), f"{root}/nodes/{uuid4()}")
        edge = await _seed_edge(session, source_uri=root, target_uri=tgt, document_id=doc_id)
        edge_id = edge.id

    assert old_key
    old_bytes = await store.get(old_key)

    async with database.session() as session:
        docs = SqlDocumentRepository(session)
        nodes = SqlNodeRepository(session)

        async def _boom(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            raise RuntimeError("simulated persist failure")

        monkeypatch.setattr(SqlNodeRepository, "bulk_create", _boom)
        pipeline = IngestionPipeline(
            documents=docs, nodes=nodes, object_store=store, session=session
        )
        with pytest.raises(RuntimeError, match="simulated persist failure"):
            await pipeline.run(
                IngestionInput(
                    filename="fail.md",
                    mime_type="text/markdown",
                    content=_md("# Fail\n\nNew body.\n"),
                    external_id="fail-1",
                    strategy=IngestionStrategy.REPLACE,
                )
            )

    async with database.session() as session:
        docs = SqlDocumentRepository(session)
        nodes = SqlNodeRepository(session)
        record = await docs.get(doc_id)
        assert record is not None
        assert record.content_hash == old_hash
        assert record.object_key == old_key
        assert await store.get(old_key) == old_bytes
        tree = await nodes.list_by_document(doc_id)
        assert {n.uri for n in tree} == uris_before
        row = await session.get(ExperienceEdgeRow, edge_id)
        assert row is not None
        assert row.status == ExperienceEdgeStatus.ACTIVE.value


@pytest.mark.asyncio
async def test_soft_delete_invalidates_and_failed_delete_preserves(
    database: Database, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    doc_id = await _ingest_service(
        database,
        store,
        content=_md("# Del\n\nX.\n"),
        strategy=IngestionStrategy.CREATE_VERSION,
    )
    root = VikingURIParser.build_document_uri(doc_id)
    async with database.session() as session:
        nodes = SqlNodeRepository(session)
        tree = await nodes.list_by_document(doc_id)
        edge = await _seed_edge(
            session,
            source_uri=root,
            target_uri=_distinct_target(tree, root),
            document_id=doc_id,
        )
        edge_id = edge.id

    svc = DocumentService(
        database=database,
        object_store=store,
        ingestion_settings=IngestionSettings(),
    )
    await svc.delete_document(doc_id)

    async with database.session() as session:
        docs = SqlDocumentRepository(session)
        record = await docs.get(doc_id)
        assert record is not None
        assert record.status is DocumentStatus.DELETED
        row = await session.get(ExperienceEdgeRow, edge_id)
        assert row is not None
        assert row.status == ExperienceEdgeStatus.INVALIDATED.value

    # Second document: failed delete rolls back
    doc2 = await _ingest_service(
        database,
        store,
        content=_md("# Del2\n\nY.\n"),
        filename="del2.md",
        strategy=IngestionStrategy.CREATE_VERSION,
    )
    root2 = VikingURIParser.build_document_uri(doc2)
    async with database.session() as session:
        nodes = SqlNodeRepository(session)
        tree = await nodes.list_by_document(doc2)
        edge2 = await _seed_edge(
            session,
            source_uri=root2,
            target_uri=_distinct_target(tree, root2),
            document_id=doc2,
        )
        edge2_id = edge2.id

    async def _boom_update(self, record):  # type: ignore[no-untyped-def]
        raise RuntimeError("simulated delete failure")

    monkeypatch.setattr(SqlDocumentRepository, "update", _boom_update)
    with pytest.raises(RuntimeError, match="simulated delete failure"):
        await svc.delete_document(doc2)

    async with database.session() as session:
        docs = SqlDocumentRepository(session)
        record = await docs.get(doc2)
        assert record is not None
        assert record.status is DocumentStatus.READY
        row = await session.get(ExperienceEdgeRow, edge2_id)
        assert row is not None
        assert row.status == ExperienceEdgeStatus.ACTIVE.value


@pytest.mark.asyncio
async def test_skip_identical_and_create_version_preserve_experience(
    database: Database, tmp_path: Path
) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    content = _md("# Skip\n\nStable.\n")
    doc_id = await _ingest_service(
        database,
        store,
        content=content,
        filename="skip.md",
        strategy=IngestionStrategy.CREATE_VERSION,
    )
    root = VikingURIParser.build_document_uri(doc_id)
    async with database.session() as session:
        nodes = SqlNodeRepository(session)
        tree = await nodes.list_by_document(doc_id)
        edge = await _seed_edge(
            session,
            source_uri=root,
            target_uri=_distinct_target(tree, root),
            document_id=doc_id,
        )
        edge_id = edge.id

    # SKIP_IDENTICAL
    again = await _ingest_service(
        database,
        store,
        content=content,
        filename="skip.md",
        strategy=IngestionStrategy.SKIP_IDENTICAL,
    )
    assert again == doc_id
    async with database.session() as session:
        row = await session.get(ExperienceEdgeRow, edge_id)
        assert row is not None
        assert row.status == ExperienceEdgeStatus.ACTIVE.value

    # CREATE_VERSION — new doc; old edges untouched
    new_id = await _ingest_service(
        database,
        store,
        content=content,
        filename="skip.md",
        strategy=IngestionStrategy.CREATE_VERSION,
    )
    assert new_id != doc_id
    async with database.session() as session:
        row = await session.get(ExperienceEdgeRow, edge_id)
        assert row is not None
        assert row.status == ExperienceEdgeStatus.ACTIVE.value
        edges = SqlExperienceEdgeRepository(session)
        assert await edges.count_active() >= 1


@pytest.mark.asyncio
async def test_replace_skips_pending_jobs(database: Database, tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    doc_id = await _ingest_service(
        database,
        store,
        content=_md("# Job\n\nA.\n"),
        external_id="job-1",
        strategy=IngestionStrategy.CREATE_VERSION,
    )
    root = VikingURIParser.build_document_uri(doc_id)
    async with database.session() as session:
        runs = SqlQueryRunRepository(session)
        run = QueryRun(
            id=new_query_run_id(),
            query_text="pending",
            status=QueryRunStatus.SUCCEEDED,
            route=QueryRunRoute.ONE_ROUND,
            query_embedding=_unit_embedding(),
            embedding_identity=_identity(),
            edge_build_status=EdgeBuildStatus.PENDING,
            learning_policy=LearningPolicy.LEARN,
            document_ids=(doc_id,),
            metadata={"citation_uris": [root]},
        )
        await runs.create(run)
        run_id = run.id

    await _ingest_service(
        database,
        store,
        content=_md("# Job\n\nB.\n"),
        external_id="job-1",
        strategy=IngestionStrategy.REPLACE,
    )
    async with database.session() as session:
        runs = SqlQueryRunRepository(session)
        got = await runs.get(run_id)
        assert got is not None
        assert got.edge_build_status is EdgeBuildStatus.SKIPPED
        assert got.edge_build_error == "document_replaced_or_deleted"


@pytest.mark.asyncio
async def test_prepare_retirement_session_helper(database: Database, tmp_path: Path) -> None:
    store = LocalObjectStore(tmp_path / "objects")
    doc_id = await _ingest_service(
        database,
        store,
        content=_md("# Prep\n\nZ.\n"),
        strategy=IngestionStrategy.CREATE_VERSION,
    )
    root = VikingURIParser.build_document_uri(doc_id)
    async with database.session() as session:
        nodes = SqlNodeRepository(session)
        tree = await nodes.list_by_document(doc_id)
        await _seed_edge(
            session,
            source_uri=root,
            target_uri=_distinct_target(tree, root),
            document_id=doc_id,
        )
        edges_n, jobs_n = await prepare_document_experience_retirement_session(session, doc_id)
        assert edges_n >= 1
        assert jobs_n == 0
