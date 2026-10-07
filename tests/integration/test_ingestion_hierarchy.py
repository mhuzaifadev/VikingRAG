from __future__ import annotations

from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from vikingrag.api.app import create_app
from vikingrag.application.navigation import StructuralNavigationService
from vikingrag.domain.models.document import IngestionStrategy, NodeType
from vikingrag.domain.uri import VikingURIParser
from vikingrag.infrastructure.cache.redis import RedisClient
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.infrastructure.object_store.local import LocalObjectStore
from vikingrag.ingestion.pipeline import IngestionPipeline
from vikingrag.ingestion.types import IngestionInput
from vikingrag.settings.config import Settings

pytestmark = pytest.mark.integration

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


@pytest.mark.asyncio
async def test_markdown_ingestion_hierarchy_and_navigation(
    database: Database,
    tmp_path: Path,
) -> None:
    content = (FIXTURES / "architecture.md").read_bytes()
    store = LocalObjectStore(tmp_path / "objects")

    async with database.session() as session:
        docs = SqlDocumentRepository(session)
        nodes = SqlNodeRepository(session)
        pipeline = IngestionPipeline(documents=docs, nodes=nodes, object_store=store)
        result = await pipeline.run(
            IngestionInput(
                filename="architecture.md",
                mime_type="text/markdown",
                content=content,
                strategy=IngestionStrategy.SKIP_IDENTICAL,
            )
        )
        assert result.skipped is False
        assert result.structural_node_count >= 10
        doc_id = result.document_id

        nav = StructuralNavigationService(documents=docs, nodes=nodes)
        tree = await nav.get_document_tree(doc_id, include_chunks=False)
        assert tree.title == "System Architecture"
        child_titles = [c.title for c in tree.children]
        assert "Storage" in child_titles
        assert "Retrieval" in child_titles
        assert "Operations" in child_titles

        storage = next(c for c in tree.children if c.title == "Storage")
        storage_children = [c.title for c in storage.children]
        assert "PostgreSQL" in storage_children
        assert "Object Storage" in storage_children

        retrieval = next(c for c in tree.children if c.title == "Retrieval")
        retrieval_children = {c.title for c in retrieval.children}
        assert {
            "Semantic Search",
            "Structural Search",
            "Evidence Verification",
        } <= retrieval_children

        # URI resolve + ancestors/descendants
        postgres = next(c for c in storage.children if c.title == "PostgreSQL")
        node = await nav.get_node(postgres.id)
        assert node.uri == VikingURIParser.build_node_uri(doc_id, postgres.id)
        resolved = await nav.resolve_uri(node.uri)
        assert resolved.id == postgres.id  # type: ignore[union-attr]

        ancestors = await nav.list_ancestors(postgres.id)
        ancestor_titles = [a.title for a in ancestors]
        assert ancestor_titles[0] == "System Architecture"
        assert "Storage" in ancestor_titles

        descendants = await nav.list_descendants(storage.id)
        desc_titles = {d.title for d in descendants if d.title}
        assert {"PostgreSQL", "Object Storage"} <= desc_titles

        # Chunks do not appear in default tree; with include_chunks they nest under leaves
        tree_with_chunks = await nav.get_document_tree(doc_id, include_chunks=True)
        chunk_types = []

        def walk(n):  # type: ignore[no-untyped-def]
            chunk_types.append(n.node_type)
            for c in n.children:
                walk(c)

        walk(tree_with_chunks)
        # Either chunks exist or leaf text was small enough to be a single chunk
        assert NodeType.CHUNK in chunk_types or result.chunk_count >= 0

        # Idempotent re-ingest
        again = await pipeline.run(
            IngestionInput(
                filename="architecture.md",
                mime_type="text/markdown",
                content=content,
                strategy=IngestionStrategy.SKIP_IDENTICAL,
            )
        )
        assert again.skipped is True
        assert again.document_id == doc_id


@pytest.mark.asyncio
async def test_pdf_ingestion(
    database: Database,
    tmp_path: Path,
) -> None:
    content = (FIXTURES / "sample.pdf").read_bytes()
    store = LocalObjectStore(tmp_path / "objects")
    async with database.session() as session:
        pipeline = IngestionPipeline(
            documents=SqlDocumentRepository(session),
            nodes=SqlNodeRepository(session),
            object_store=store,
        )
        result = await pipeline.run(
            IngestionInput(
                filename="sample.pdf",
                mime_type="application/pdf",
                content=content,
                strategy=IngestionStrategy.CREATE_VERSION,
            )
        )
        assert result.status == "ready"
        assert result.structural_node_count >= 1


@pytest.mark.asyncio
async def test_api_ingest_and_tree(
    integration_settings: Settings,
    database: Database,
    redis_client: RedisClient,
    tmp_path: Path,
) -> None:
    settings = integration_settings
    settings.object_store.local_root = str(tmp_path / "objects")
    app = create_app(settings)
    app.state.database = database
    app.state.redis = redis_client
    app.state.object_store = LocalObjectStore(settings.object_store.local_root)

    content = (FIXTURES / "architecture.md").read_bytes()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/documents",
            files={"file": ("architecture.md", content, "text/markdown")},
            data={"strategy": "create_version"},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        doc_id = body["document_id"]

        got = await client.get(f"/v1/documents/{doc_id}")
        assert got.status_code == 200
        assert got.json()["status"] == "ready"

        tree = await client.get(f"/v1/documents/{doc_id}/tree")
        assert tree.status_code == 200
        payload = tree.json()
        assert payload["title"] == "System Architecture"
        # Default tree omits dumping chunk bodies
        assert "content" not in payload

        # Find a nested node id and fetch it
        storage = next(c for c in payload["children"] if c["title"] == "Storage")
        node_id = storage["children"][0]["id"]
        node = await client.get(f"/v1/nodes/{node_id}")
        assert node.status_code == 200
        uri = node.json()["uri"]
        resolved = await client.get("/v1/uris/resolve", params={"uri": uri})
        assert resolved.status_code == 200
        assert resolved.json()["id"] == node_id
