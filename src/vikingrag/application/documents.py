"""Document use-cases: ingest and read."""

from __future__ import annotations

from vikingrag.application.experience.invalidate import (
    prepare_document_experience_retirement_session,
)
from vikingrag.domain.errors import DocumentNotFound
from vikingrag.domain.models.document import DocumentId, DocumentStatus, IngestionStrategy
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.base import DocumentRecord
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.infrastructure.object_store.base import ObjectStore
from vikingrag.ingestion.chunking import ChunkingConfig
from vikingrag.ingestion.pipeline import IngestionPipeline
from vikingrag.ingestion.types import IngestionInput, IngestionResult
from vikingrag.observability.logging import get_logger
from vikingrag.settings.config import IngestionSettings

logger = get_logger(__name__)


class DocumentService:
    def __init__(
        self,
        *,
        database: Database,
        object_store: ObjectStore,
        ingestion_settings: IngestionSettings,
    ) -> None:
        self._database = database
        self._object_store = object_store
        self._ingestion_settings = ingestion_settings

    async def ingest(
        self,
        *,
        filename: str,
        mime_type: str,
        content: bytes,
        external_id: str | None = None,
        strategy: IngestionStrategy | None = None,
    ) -> IngestionResult:
        """Ingest a document.

        REPLACE invalidation lives in ``IngestionPipeline`` (same DB session as
        hierarchy replace) so external_id and content-hash resolve paths both apply.
        """
        resolved_strategy = strategy or IngestionStrategy(self._ingestion_settings.default_strategy)
        async with self._database.session() as session:
            docs = SqlDocumentRepository(session)
            nodes = SqlNodeRepository(session)
            pipeline = IngestionPipeline(
                documents=docs,
                nodes=nodes,
                object_store=self._object_store,
                session=session,
                chunking_config=ChunkingConfig(
                    target_tokens=self._ingestion_settings.chunk_target_tokens,
                    max_tokens=self._ingestion_settings.chunk_max_tokens,
                    overlap_tokens=self._ingestion_settings.chunk_overlap_tokens,
                ),
            )
            return await pipeline.run(
                IngestionInput(
                    filename=filename,
                    mime_type=mime_type,
                    content=content,
                    external_id=external_id,
                    strategy=resolved_strategy,
                )
            )

    async def get_document(self, document_id: DocumentId) -> DocumentRecord:
        async with self._database.session() as session:
            docs = SqlDocumentRepository(session)
            record = await docs.get(document_id)
            if record is None:
                raise DocumentNotFound(f"Document not found: {document_id}")
            return record

    async def delete_document(self, document_id: DocumentId) -> None:
        """Soft-delete a document and invalidate incident experience edges (one txn)."""
        async with self._database.session() as session:
            docs = SqlDocumentRepository(session)
            record = await docs.get(document_id)
            if record is None:
                raise DocumentNotFound(f"Document not found: {document_id}")
            edges, jobs = await prepare_document_experience_retirement_session(session, document_id)
            record.status = DocumentStatus.DELETED
            await docs.update(record)
        logger.info(
            "document_deleted",
            document_id=str(document_id),
            edges_invalidated=edges,
            jobs_skipped=jobs,
        )
