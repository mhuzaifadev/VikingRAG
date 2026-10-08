"""Library facade — wire Settings → providers → retrieval/answer services without FastAPI.

Example::

    async with VikingRAGClient.from_settings() as rag:
        doc = await rag.ingest("policy.md")
        await rag.index(doc.id)
        answer = await rag.ask(
            "What is the cancellation policy?",
            document_ids=[doc.id],
            mode="vikingrag_e_plus",
            learning_policy="record_only",
        )
"""

from __future__ import annotations

import asyncio
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Self
from uuid import UUID

from vikingrag.application.answer.generate import AnswerGenerator
from vikingrag.application.documents import DocumentService
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.indexing import DocumentIndexingService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.domain.models.answer import AnswerRequest, AnswerResponse, ExecutionMode
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.experience import ExperienceSnapshotId, LearningPolicy
from vikingrag.infrastructure.database.engine import Database, create_database
from vikingrag.infrastructure.object_store.factory import build_object_store
from vikingrag.observability.logging import get_logger
from vikingrag.providers.embeddings.base import EmbeddingProvider
from vikingrag.providers.factory import (
    build_embedding_provider,
    build_llm_provider,
    build_reranker,
    build_summary_generator,
)
from vikingrag.providers.llm.base import LLMProvider
from vikingrag.settings.config import Settings, get_settings

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DocumentRef:
    id: DocumentId
    content_hash: str
    status: str
    skipped: bool = False


@dataclass
class VikingRAGClient:
    """Composable SDK entry point for ingest, index, ask, and explain."""

    settings: Settings
    database: Database
    embedding: EmbeddingProvider
    llm: LLMProvider | None
    object_store: Any
    search: SemanticSearchService
    list_service: ListService
    grep_service: GrepService
    read_service: ReadService
    search_plus: Any | None = None
    search_plus_error: str | None = None
    _owns_database: bool = False
    _closed: bool = field(default=False, init=False, repr=False)

    @classmethod
    def from_settings(cls, settings: Settings | None = None) -> VikingRAGClient:
        cfg = settings or get_settings()
        database = create_database(cfg.database)
        embedding = build_embedding_provider(cfg)
        llm: LLMProvider | None
        try:
            llm = build_llm_provider(cfg)
        except NotImplementedCapabilityError:
            llm = None
        object_store = build_object_store(cfg.object_store)
        reranker = build_reranker(cfg) if cfg.retrieval.rerank_enabled else None
        search = SemanticSearchService(
            database=database,
            embedding_provider=embedding,
            retrieval_settings=cfg.retrieval,
            embedding_settings=cfg.embedding,
            reranker=reranker,
        )
        search_plus = None
        search_plus_error: str | None = None
        try:
            from vikingrag.application.search_plus import build_search_plus

            search_plus = build_search_plus(
                cfg,
                database,
                search=search,
                embedding_provider=embedding,
            )
        except Exception as exc:
            search_plus_error = f"{type(exc).__name__}: {exc}"
            logger.warning("search_plus_unavailable", error=search_plus_error)
            search_plus = None
        return cls(
            settings=cfg,
            database=database,
            embedding=embedding,
            llm=llm,
            object_store=object_store,
            search=search,
            list_service=ListService(database=database),
            grep_service=GrepService(database=database),
            read_service=ReadService(database=database),
            search_plus=search_plus,
            search_plus_error=search_plus_error,
            _owns_database=True,
        )

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()

    def require_search_plus(self, mode: ExecutionMode | str) -> Any:
        """Return Search+ or raise — never silently degrade E/E+ to ordinary Search."""
        mode_value = mode.value if isinstance(mode, ExecutionMode) else str(mode)
        if mode_value not in {
            ExecutionMode.VIKINGRAG_E.value,
            ExecutionMode.VIKINGRAG_E_PLUS.value,
        }:
            return self.search_plus
        if self.search_plus is not None:
            return self.search_plus
        detail = self.search_plus_error or "Search+ was not constructed"
        raise NotImplementedCapabilityError(
            f"search_plus (required for execution_mode={mode_value}): {detail}"
        )

    def answer_generator(self) -> AnswerGenerator:
        if self.llm is None:
            raise NotImplementedCapabilityError("llm_provider")
        return AnswerGenerator(
            llm=self.llm,
            search=self.search,
            list_service=self.list_service,
            grep_service=self.grep_service,
            read_service=self.read_service,
            settings=self.settings,
            search_plus=self.search_plus,
            database=self.database,
        )

    def document_service(self) -> DocumentService:
        return DocumentService(
            database=self.database,
            object_store=self.object_store,
            ingestion_settings=self.settings.ingestion,
        )

    def indexing_service(self) -> DocumentIndexingService:
        return DocumentIndexingService(
            database=self.database,
            summary_generator=build_summary_generator(self.settings),
            embedding_provider=self.embedding,
            indexing_settings=self.settings.indexing,
            embedding_settings=self.settings.embedding,
        )

    async def ingest(
        self,
        source: str | Path | bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
        external_id: str | None = None,
    ) -> DocumentRef:
        """Ingest a file path or raw bytes into the document store."""
        if isinstance(source, bytes):
            content = source
            name = filename or "upload.bin"
            mime = mime_type or "application/octet-stream"
        else:
            path = Path(source)
            content = await asyncio.to_thread(path.read_bytes)
            name = filename or path.name
            guessed, _ = mimetypes.guess_type(name)
            mime = mime_type or guessed or "application/octet-stream"
        result = await self.document_service().ingest(
            filename=name,
            mime_type=mime,
            content=content,
            external_id=external_id,
        )
        return DocumentRef(
            id=result.document_id,
            content_hash=result.content_hash,
            status=result.status,
            skipped=result.skipped,
        )

    async def index(self, document_id: DocumentId | UUID | str, **kwargs: Any) -> Any:
        """Build summaries + embeddings for an ingested document."""
        doc_id = DocumentId(UUID(str(document_id)))
        return await self.indexing_service().index_document(doc_id, **kwargs)

    async def ask(
        self,
        question: str,
        *,
        document_ids: list[DocumentId | UUID | str] | tuple[DocumentId, ...] | None = None,
        mode: ExecutionMode | str = ExecutionMode.VIKINGRAG,
        learning_policy: LearningPolicy | str = LearningPolicy.LEARN,
        snapshot_id: ExperienceSnapshotId | UUID | str | None = None,
        instructions: str | None = None,
    ) -> AnswerResponse:
        """Answer a question with explicit mode and learning policy."""
        exec_mode = mode if isinstance(mode, ExecutionMode) else ExecutionMode(str(mode))
        policy = (
            learning_policy
            if isinstance(learning_policy, LearningPolicy)
            else LearningPolicy(str(learning_policy))
        )
        self.require_search_plus(exec_mode)
        docs: tuple[DocumentId, ...] = ()
        if document_ids:
            docs = tuple(DocumentId(UUID(str(d))) for d in document_ids)
        snap: ExperienceSnapshotId | None = None
        if snapshot_id is not None:
            snap = ExperienceSnapshotId(UUID(str(snapshot_id)))
        return await self.answer_generator().generate(
            AnswerRequest(
                question=question,
                document_ids=docs,
                execution_mode=exec_mode,
                learning_policy=policy,
                snapshot_id=snap,
                instructions=instructions,
            )
        )

    async def explain(
        self,
        query_id: UUID | str,
        *,
        permitted_document_ids: frozenset[DocumentId] | None = None,
    ) -> dict[str, Any]:
        """Load a stored answer/run explanation (Milestone C)."""
        from vikingrag.application.experience.explain import explain_query_run

        return await explain_query_run(
            self.database,
            query_id=UUID(str(query_id)),
            permitted_document_ids=permitted_document_ids,
        )

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.llm, "aclose", None)
        if close is not None:
            await close()
        close_e = getattr(self.embedding, "aclose", None)
        if close_e is not None:
            await close_e()
        close_o = getattr(self.object_store, "aclose", None)
        if close_o is not None:
            await close_o()
        if self._owns_database:
            await self.database.dispose()
