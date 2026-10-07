"""Library facade — wire Settings → providers → retrieval/answer services without FastAPI.

Example::

    from vikingrag.client import VikingRAGClient

    client = VikingRAGClient.from_settings()
    # client.search / client.answer / client.database ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vikingrag.application.answer.generate import AnswerGenerator
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.infrastructure.database.engine import Database, create_database
from vikingrag.infrastructure.object_store.factory import build_object_store
from vikingrag.providers.embeddings.base import EmbeddingProvider
from vikingrag.providers.factory import (
    build_embedding_provider,
    build_llm_provider,
    build_reranker,
)
from vikingrag.providers.llm.base import LLMProvider
from vikingrag.settings.config import Settings, get_settings


@dataclass
class VikingRAGClient:
    """Composable SDK entry point for embedders, search, and answers."""

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
    _owns_database: bool = False

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
        try:
            from vikingrag.application.search_plus import build_search_plus

            search_plus = build_search_plus(cfg, database)
        except Exception:
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
            _owns_database=True,
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

    async def aclose(self) -> None:
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
