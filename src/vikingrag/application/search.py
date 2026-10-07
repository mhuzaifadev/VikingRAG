"""Semantic Search primitive - discovery only, compact URI-addressable hits."""

from __future__ import annotations

import time
from uuid import uuid4

from vikingrag.application.budget import RetrievalContext
from vikingrag.application.navigation import StructuralNavigationService
from vikingrag.domain.errors import ScopeDeniedError
from vikingrag.domain.models.document import NodeId, NodeType
from vikingrag.domain.models.representation import (
    EmbeddingIdentity,
    SearchHit,
    SearchRequest,
    SearchResponse,
    SearchTiming,
    SearchUsage,
)
from vikingrag.domain.models.retrieval import RetrievalBudget, RetrievalTrace
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.embedding import SqlEmbeddingRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.observability.logging import get_logger
from vikingrag.providers.embeddings.base import EmbeddingProvider
from vikingrag.providers.reranking.base import RerankerProvider
from vikingrag.providers.reranking.noop import NoOpReranker
from vikingrag.settings.config import EmbeddingSettings, RetrievalSettings

logger = get_logger(__name__)


class SemanticSearchService:
    def __init__(
        self,
        *,
        database: Database,
        embedding_provider: EmbeddingProvider,
        retrieval_settings: RetrievalSettings,
        embedding_settings: EmbeddingSettings,
        reranker: RerankerProvider | None = None,
    ) -> None:
        self._database = database
        self._embeddings = embedding_provider
        self._retrieval = retrieval_settings
        self._embedding_settings = embedding_settings
        self._reranker = reranker or NoOpReranker()
        self._rerank_enabled = retrieval_settings.rerank_enabled and reranker is not None

    def embedding_identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            provider=self._embeddings.provider_name,
            model=self._embeddings.default_model,
            dimensions=self._embeddings.dimensions,
            version=self._embedding_settings.identity_version,
        )

    def granularity_weight(self, node_type: NodeType) -> float:
        weights = {
            NodeType.DOCUMENT: self._retrieval.weight_document,
            NodeType.SECTION: self._retrieval.weight_section,
            NodeType.SUBSECTION: self._retrieval.weight_subsection,
            NodeType.CHUNK: self._retrieval.weight_chunk,
        }
        return weights.get(node_type, 1.0)

    async def search(
        self,
        request: SearchRequest,
        *,
        budget: RetrievalBudget | None = None,
        trace: RetrievalTrace | None = None,
        ctx: RetrievalContext | None = None,
    ) -> SearchResponse:
        context = ctx or RetrievalContext.create(budget=budget)
        query_id = context.query_id if ctx is not None else uuid4()
        total_started = time.perf_counter()
        identity = self.embedding_identity()
        top_k = request.top_k or self._retrieval.initial_top_k
        pool = request.candidate_pool_size or self._retrieval.candidate_pool_size
        pool = max(pool, top_k)
        min_score = (
            request.min_score if request.min_score is not None else self._retrieval.min_score
        )

        # Scope: empty permit set = deny-all (raise before embed/SQL).
        # None = unrestricted; nonempty = SQL filter to those docs.
        context.require_scope_not_empty()
        scoped_docs = context.narrow_document_ids(list(request.document_ids))

        async with context.tool_call("search"):
            context.check_deadline()
            scope_node_id: NodeId | None = None
            if request.scope_uri is not None:
                async with self._database.session() as session:
                    await context.reserve(db_operations=1)
                    nav = StructuralNavigationService(
                        documents=SqlDocumentRepository(session),
                        nodes=SqlNodeRepository(session),
                    )
                    scope_node = await nav.resolve_uri(request.scope_uri.strip())
                    if not context.document_allowed(scope_node.document_id):
                        raise ScopeDeniedError(
                            f"scope_uri document {scope_node.document_id} outside permitted scope"
                        )
                    scope_node_id = scope_node.id
                    # When client sent no document_ids, restrict SQL to the scope's document.
                    if not scoped_docs:
                        scoped_docs = context.narrow_document_ids((scope_node.document_id,))

            cache_key = (request.query, identity.key())
            cached = context.get_cached_query_embedding(cache_key)
            embed_started = time.perf_counter()
            if cached is not None:
                query_vector = cached
                embedding_ms = 0.0
                embedding_calls = 0
            else:
                await context.reserve(embedding_calls=1)
                embedded = await context.await_with_deadline(
                    self._embeddings.embed_text(request.query)
                )
                query_vector = embedded.vectors[0]
                context.cache_query_embedding(cache_key, query_vector)
                embedding_ms = (time.perf_counter() - embed_started) * 1000.0
                embedding_calls = 1

            await context.reserve(vector_searches=1, db_operations=1)
            vector_started = time.perf_counter()
            # When unrestricted and client sent no ids, pass empty → no SQL doc filter.
            # When permitted set exists, scoped_docs is always nonempty (or raised).
            sql_doc_ids = scoped_docs if scoped_docs else ()
            async with self._database.session() as session:
                repo = SqlEmbeddingRepository(session)
                raw_hits = await repo.search_cosine(
                    query_vector,
                    identity=identity,
                    top_k=pool,
                    document_ids=sql_doc_ids,
                    node_types=request.node_types,
                    representation_types=request.representation_types,
                    min_score=min_score,
                    scope_node_id=scope_node_id,
                )
            vector_ms = (time.perf_counter() - vector_started) * 1000.0
            await context.add_usage(nodes_inspected=len(raw_hits))

            # Defense in depth
            if context.permitted_document_ids is not None:
                raw_hits = [h for h in raw_hits if h.document_id in context.permitted_document_ids]

            scored: list[SearchHit] = []
            for hit in raw_hits:
                weight = self.granularity_weight(hit.node_type)
                score = hit.similarity * weight
                if min_score is not None and score < min_score:
                    continue
                scored.append(
                    SearchHit(
                        uri=hit.uri,
                        title=hit.title,
                        node_id=hit.node_id,
                        document_id=hit.document_id,
                        node_type=hit.node_type,
                        representation_type=hit.representation_type,
                        score=score,
                        similarity=hit.similarity,
                        preview=hit.preview,
                        metadata={
                            "provider": hit.provider,
                            "model": hit.model,
                            "granularity_weight": weight,
                        },
                    )
                )
            scored.sort(key=lambda h: h.score, reverse=True)

            rerank_ms = 0.0
            reranked = False
            rerank_calls = 0
            # Only claim reranked when a non-NoOp reranker actually ran
            from vikingrag.providers.reranking.noop import NoOpReranker as _NoOp

            real_rerank = self._rerank_enabled and not isinstance(self._reranker, _NoOp)
            if real_rerank and scored:
                rerank_started = time.perf_counter()
                docs = [f"{h.title or ''}\n{h.preview}" for h in scored]
                rerank_hits = await self._reranker.rerank(request.query, docs, top_n=top_k)
                rerank_ms = (time.perf_counter() - rerank_started) * 1000.0
                rerank_calls = 1
                reranked = True
                ordered: list[SearchHit] = []
                for rh in rerank_hits:
                    if 0 <= rh.index < len(scored):
                        base = scored[rh.index]
                        ordered.append(
                            SearchHit(
                                uri=base.uri,
                                title=base.title,
                                node_id=base.node_id,
                                document_id=base.document_id,
                                node_type=base.node_type,
                                representation_type=base.representation_type,
                                score=float(rh.score),
                                similarity=base.similarity,
                                preview=base.preview,
                                metadata={**base.metadata, "rerank_score": rh.score},
                            )
                        )
                scored = ordered
            else:
                scored = scored[:top_k]

            total_ms = (time.perf_counter() - total_started) * 1000.0
            usage = SearchUsage(
                embedding_calls=embedding_calls,
                vector_searches=1,
                candidates_inspected=len(raw_hits),
                nodes_returned=len(scored),
                rerank_calls=rerank_calls,
            )
            timing = SearchTiming(
                embedding_ms=embedding_ms,
                vector_ms=vector_ms,
                rerank_ms=rerank_ms,
                total_ms=total_ms,
            )

            if trace is not None:
                trace.record(
                    "search",
                    query_id=str(query_id),
                    top_k=top_k,
                    pool=pool,
                    returned=len(scored),
                    embedding_ms=embedding_ms,
                    vector_ms=vector_ms,
                    rerank_ms=rerank_ms,
                )

            logger.info(
                "search_completed",
                query_id=str(query_id),
                trace_id=context.trace_id,
                query_length=len(request.query),
                top_k=top_k,
                candidate_pool_size=pool,
                returned_results=len(scored),
                embedding_ms=embedding_ms,
                vector_ms=vector_ms,
                rerank_ms=rerank_ms,
                total_ms=total_ms,
                provider=identity.provider,
                model=identity.model,
                reranked=reranked,
            )

            return SearchResponse(
                query=request.query,
                query_id=query_id,
                candidates=tuple(scored),
                timing=timing,
                usage=usage,
                embedding_identity=identity,
                reranked=reranked,
            )
