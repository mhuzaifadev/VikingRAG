"""Integration: ingest → index (fake/deterministic) → semantic Search."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from vikingrag.application.indexing import DocumentIndexingService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.models.document import DocumentId, IngestionStrategy, NodeType
from vikingrag.domain.models.representation import SearchRequest
from vikingrag.domain.uri.viking_uri import VikingURIParser
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.infrastructure.object_store.local import LocalObjectStore
from vikingrag.ingestion.chunking import ChunkingConfig
from vikingrag.ingestion.pipeline import IngestionPipeline
from vikingrag.ingestion.types import IngestionInput
from vikingrag.providers.embeddings.deterministic import DeterministicEmbeddingProvider
from vikingrag.providers.llm.fake import FakeSummaryGenerator
from vikingrag.settings.config import EmbeddingSettings, IndexingSettings, RetrievalSettings

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "system_architecture.md"


@pytest_asyncio.fixture
async def object_store(tmp_path_factory: pytest.TempPathFactory) -> LocalObjectStore:
    root = tmp_path_factory.mktemp("obj")
    return LocalObjectStore(str(root))


@pytest.mark.integration
@pytest.mark.asyncio
async def test_evidence_verification_ranks_highly(
    database: Database,
    object_store: LocalObjectStore,
) -> None:
    content = FIXTURE.read_bytes()
    async with database.session() as session:
        pipeline = IngestionPipeline(
            documents=SqlDocumentRepository(session),
            nodes=SqlNodeRepository(session),
            object_store=object_store,
            chunking_config=ChunkingConfig(target_tokens=200, max_tokens=400, overlap_tokens=20),
        )
        ingested = await pipeline.run(
            IngestionInput(
                filename="system_architecture.md",
                mime_type="text/markdown",
                content=content,
                strategy=IngestionStrategy.REPLACE,
            )
        )
    document_id = ingested.document_id

    embedding_provider = DeterministicEmbeddingProvider(
        dimensions=1536,
        model="deterministic-hash-v1",
    )
    indexing = DocumentIndexingService(
        database=database,
        summary_generator=FakeSummaryGenerator(model="fake-summary-v1", version="1"),
        embedding_provider=embedding_provider,
        indexing_settings=IndexingSettings(summary_concurrency=2),
        embedding_settings=EmbeddingSettings(
            provider="deterministic",
            model="deterministic-hash-v1",
            dimensions=1536,
            identity_version="1",
            batch_size=16,
        ),
    )
    result = await indexing.index_document(DocumentId(document_id))
    assert result.stage.value == "indexed"
    assert result.metrics.embeddings_generated > 0

    # Idempotent second pass reuses work
    again = await indexing.index_document(DocumentId(document_id))
    assert again.metrics.embeddings_reused >= again.metrics.embeddings_generated

    status = await indexing.get_status(DocumentId(document_id))
    assert status.embedding_count > 0
    assert status.representation_count > 0

    search = SemanticSearchService(
        database=database,
        embedding_provider=embedding_provider,
        retrieval_settings=RetrievalSettings(initial_top_k=8, candidate_pool_size=30),
        embedding_settings=EmbeddingSettings(
            provider="deterministic",
            model="deterministic-hash-v1",
            dimensions=1536,
        ),
    )
    response = await search.search(
        SearchRequest(query="How do we verify retrieved evidence?", top_k=8)
    )
    assert response.candidates, "expected at least one search hit"
    titles = " ".join((c.title or "") + " " + c.preview for c in response.candidates).lower()
    assert "evidence" in titles and "verif" in titles

    top = response.candidates[0]
    # Prefer the Evidence Verification region among top hits
    top_n_titles = [(c.title or "").lower() for c in response.candidates[:5]]
    assert any("evidence verification" in t for t in top_n_titles)

    parsed = VikingURIParser.parse(top.uri)
    assert parsed.document_id is not None
    async with database.session() as session:
        nodes = SqlNodeRepository(session)
        node = await nodes.get_by_uri(top.uri)
        assert node is not None
        assert node.node_type in {
            NodeType.DOCUMENT,
            NodeType.SECTION,
            NodeType.SUBSECTION,
            NodeType.CHUNK,
        }


@pytest.mark.integration
@pytest.mark.asyncio
async def test_search_document_filter_and_min_score(
    database: Database,
    object_store: LocalObjectStore,
) -> None:
    content = FIXTURE.read_bytes()
    async with database.session() as session:
        pipeline = IngestionPipeline(
            documents=SqlDocumentRepository(session),
            nodes=SqlNodeRepository(session),
            object_store=object_store,
            chunking_config=ChunkingConfig(target_tokens=200, max_tokens=400, overlap_tokens=20),
        )
        ingested = await pipeline.run(
            IngestionInput(
                filename="system_architecture.md",
                mime_type="text/markdown",
                content=content,
                strategy=IngestionStrategy.REPLACE,
            )
        )

    embedding_provider = DeterministicEmbeddingProvider(dimensions=1536)
    indexing = DocumentIndexingService(
        database=database,
        summary_generator=FakeSummaryGenerator(),
        embedding_provider=embedding_provider,
        indexing_settings=IndexingSettings(),
        embedding_settings=EmbeddingSettings(dimensions=1536, batch_size=16),
    )
    await indexing.index_document(DocumentId(ingested.document_id))

    search = SemanticSearchService(
        database=database,
        embedding_provider=embedding_provider,
        retrieval_settings=RetrievalSettings(),
        embedding_settings=EmbeddingSettings(dimensions=1536),
    )
    response = await search.search(
        SearchRequest(
            query="PostgreSQL hierarchy store",
            top_k=5,
            document_ids=(DocumentId(ingested.document_id),),
            min_score=0.01,
        )
    )
    assert all(c.document_id == DocumentId(ingested.document_id) for c in response.candidates)
    assert response.usage.embedding_calls == 1
    assert response.usage.vector_searches == 1
