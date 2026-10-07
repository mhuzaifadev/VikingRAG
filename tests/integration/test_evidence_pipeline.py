"""Integration: Search → List/Grep/Read → bundle → scripted assessment."""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio

from vikingrag.application.assessment import ScriptedEvidenceAssessor
from vikingrag.application.budget import RetrievalContext
from vikingrag.application.evidence import EvidenceRetrievalService
from vikingrag.application.evidence_collector import EvidenceCollectionSettings, EvidenceCollector
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.indexing import DocumentIndexingService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.models.assessment import AssessmentStatus
from vikingrag.domain.models.document import DocumentId, IngestionStrategy
from vikingrag.domain.models.primitives import GrepRequest, ListRequest, ReadRequest
from vikingrag.domain.models.representation import SearchRequest
from vikingrag.domain.uri import VikingURIParser
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
    return LocalObjectStore(str(tmp_path_factory.mktemp("obj")))


async def _ingest_and_index(
    database: Database, object_store: LocalObjectStore
) -> tuple[DocumentId, DeterministicEmbeddingProvider]:
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
    doc_id = DocumentId(ingested.document_id)
    embedding = DeterministicEmbeddingProvider(dimensions=1536, model="deterministic-hash-v1")
    indexing = DocumentIndexingService(
        database=database,
        summary_generator=FakeSummaryGenerator(),
        embedding_provider=embedding,
        indexing_settings=IndexingSettings(summary_concurrency=2),
        embedding_settings=EmbeddingSettings(
            provider="deterministic",
            model="deterministic-hash-v1",
            dimensions=1536,
            batch_size=16,
        ),
    )
    await indexing.index_document(doc_id)
    return doc_id, embedding


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_evidence_pipeline(database: Database, object_store: LocalObjectStore) -> None:
    doc_id, embedding = await _ingest_and_index(database, object_store)
    search = SemanticSearchService(
        database=database,
        embedding_provider=embedding,
        retrieval_settings=RetrievalSettings(),
        embedding_settings=EmbeddingSettings(dimensions=1536, model="deterministic-hash-v1"),
    )
    list_service = ListService(database=database)
    read_service = ReadService(database=database)
    grep_service = GrepService(database=database)

    # Search
    hits = await search.search(
        SearchRequest(query="How do we verify retrieved evidence?", top_k=5),
    )
    assert hits.candidates

    # List a structural hit if present, else document root
    top = hits.candidates[0]
    listing = await list_service.list(ListRequest(uri=top.uri, limit=10))
    assert listing.document_id == top.document_id

    # Grep exact phrase in document scope
    doc_uri = VikingURIParser.build_document_uri(doc_id)
    grepped = await grep_service.grep(
        GrepRequest(uri=doc_uri, pattern="Evidence Verification", case_sensitive=True)
    )
    assert grepped.matches
    match = grepped.matches[0]

    # Read authoritative range
    read = await read_service.read(
        ReadRequest(
            uri=match.uri,
            start_offset=match.start_offset,
            max_tokens=200,
            expected_content_hash=match.content_hash,
        )
    )
    assert read.has_direct_content
    assert "Evidence" in read.text or "evidence" in read.text.lower()

    from vikingrag.domain.errors import StaleSourceError

    with pytest.raises(StaleSourceError):
        await read_service.read(ReadRequest(uri=match.uri, expected_content_hash="not-the-hash"))

    # Shared budget across repeated searches
    ctx = RetrievalContext.create()
    await search.search(SearchRequest(query="evidence verification", top_k=3), ctx=ctx)
    await search.search(SearchRequest(query="evidence verification", top_k=3), ctx=ctx)
    assert ctx.usage.tool_calls >= 2
    assert ctx.usage.embedding_calls == 1  # query embedding cached
    assert ctx.usage.vector_searches == 2

    # Collect + assess
    collector = EvidenceCollector(
        search=search,
        list_service=list_service,
        read_service=read_service,
        grep_service=grep_service,
    )
    service = EvidenceRetrievalService(
        collector=collector,
        assessor=ScriptedEvidenceAssessor(),
    )
    result = await service.retrieve_evidence(
        "How do we verify retrieved evidence?",
        document_ids=(doc_id,),
        settings=EvidenceCollectionSettings(
            search_top_k=5,
            max_candidates=5,
            max_list_children=5,
            max_descent_depth=1,
            include_grep=True,
            grep_pattern="Evidence Verification",
        ),
    )
    assert result.collection.bundle.items
    assert result.assessment.status in {
        AssessmentStatus.SUFFICIENT,
        AssessmentStatus.INSUFFICIENT,
        AssessmentStatus.UNKNOWN,
    }
    # Scripted default should be sufficient when evidence present
    assert result.assessment.status is AssessmentStatus.SUFFICIENT
    assert result.usage["tool_calls"] >= 1
    assert result.collection.bundle.total_tokens == sum(
        i.token_count for i in result.collection.bundle.items
    )
