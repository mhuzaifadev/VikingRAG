"""Library facade — wire Settings → providers → retrieval/answer services without FastAPI.

Example::

    from vikingrag.client import VikingRAGClient

    client = VikingRAGClient.from_settings()
    # client.search / client.answer / client.database ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vikingrag.application.answer.generate import AnswerGenerator
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.domain.models.answer import ExecutionMode
from vikingrag.infrastructure.database.engine import Database, create_database
from vikingrag.infrastructure.object_store.factory import build_object_store
from vikingrag.observability.logging import get_logger
from vikingrag.providers.embeddings.base import EmbeddingProvider
from vikingrag.providers.factory import (
    build_embedding_provider,
    build_llm_provider,
    build_reranker,
)
from vikingrag.providers.llm.base import LLMProvider
from vikingrag.settings.config import Settings, get_settings

logger = get_logger(__name__)


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

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        close = getattr(self.llm, "aclose", None)
        if close is not None:
            await close()
        # Search+ reuses the same embedding provider; close once via self.embedding.
        close_e = getattr(self.embedding, "aclose", None)
        if close_e is not None:
            await close_e()
        close_o = getattr(self.object_store, "aclose", None)
        if close_o is not None:
            await close_o()
        if self._owns_database:
            await self.database.dispose()
