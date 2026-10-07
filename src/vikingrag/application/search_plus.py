"""Experience-augmented Search (Algorithm 3 / Search+)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from vikingrag.application.budget import RetrievalContext
from vikingrag.application.experience.expand import EdgeLookup, expand_experience_edges
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.models.experience import ExpansionResult, ExperienceExpansionLimits
from vikingrag.domain.models.representation import (
    EmbeddingIdentity,
    SearchHit,
    SearchRequest,
    SearchResponse,
)
from vikingrag.domain.models.retrieval import RetrievalBudget, RetrievalTrace
from vikingrag.infrastructure.database.repositories.experience import (
    SqlExperienceEdgeRepository,
)
from vikingrag.observability.logging import get_logger
from vikingrag.providers.embeddings.base import EmbeddingProvider

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ExpansionHit:
    seed_uri: str
    target_uri: str
    edge_id: UUID
    similarity: float
    hop: int
    historical_query: str | None = None
    payload_id: UUID | None = None


@dataclass(frozen=True, slots=True)
class ExpansionBundle:
    """URI set used by E+ bounded reads."""

    seed_uris: tuple[str, ...]
    expanded_uris: tuple[str, ...]

    @property
    def all_uris(self) -> tuple[str, ...]:
        seen: set[str] = set()
        out: list[str] = []
        for u in (*self.seed_uris, *self.expanded_uris):
            if u not in seen:
                seen.add(u)
                out.append(u)
        return tuple(out)


@dataclass(slots=True)
class SearchPlusResponse:
    base: SearchResponse
    expansions: tuple[ExpansionHit, ...] = ()
    truncated: bool = False
    hops_used: int = 0
    edges_activated: int = 0
    cold_path: bool = True
    query_embedding: list[float] = field(default_factory=list)
    embedding_identity: EmbeddingIdentity | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def candidates(self) -> tuple[SearchHit, ...]:
        """Seed hits — tool executor / callers expecting Search-like shape."""
        return self.base.candidates

    @property
    def seed_hits(self) -> tuple[SearchHit, ...]:
        return self.base.candidates

    @property
    def expansion(self) -> ExpansionBundle:
        seeds = tuple(h.uri for h in self.base.candidates)
        expanded = tuple(e.target_uri for e in self.expansions)
        return ExpansionBundle(seed_uris=seeds, expanded_uris=expanded)

    @property
    def query(self) -> str:
        return self.base.query

    @property
    def query_id(self) -> UUID:
        return self.base.query_id


class _SessionEdgeLookup:
    """Adapts a SQL repo factory + Database into EdgeLookup for one expansion."""

    def __init__(self, database: Any, factory: Any) -> None:
        self._database = database
        self._factory = factory

    async def find_active_by_source(
        self,
        source_uri: str,
        *,
        identity: EmbeddingIdentity,
    ) -> list[tuple[Any, Any]]:
        async with self._database.session() as session:
            repo = self._factory(session)
            rows = await repo.find_active_by_source(source_uri, identity=identity)
            return list(rows)

    async def count_active(self) -> int:
        async with self._database.session() as session:
            repo = self._factory(session)
            return int(await repo.count_active())


class ExperienceAugmentedSearch:
    """Seed Search -> gamma-gated multi-hop frontier expansion.

    Empty active edge store → cold path (seeds only). Expansion is not a second
    semantic Search over targets.
    """

    def __init__(
        self,
        *,
        search: SemanticSearchService,
        edges: EdgeLookup | None = None,
        edge_repo_factory: Any | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        limits: ExperienceExpansionLimits | None = None,
    ) -> None:
        self._search = search
        if edges is not None:
            self._edges: EdgeLookup = edges
        elif edge_repo_factory is not None:
            self._edges = _SessionEdgeLookup(search._database, edge_repo_factory)
        else:
            raise ValueError("Provide edges (EdgeLookup) or edge_repo_factory")
        self._embeddings = embedding_provider
        self._limits = limits or ExperienceExpansionLimits()

    async def search(
        self,
        request: SearchRequest,
        *,
        ctx: RetrievalContext | None = None,
        query_embedding: list[float] | None = None,
        budget: RetrievalBudget | None = None,
        trace: RetrievalTrace | None = None,
    ) -> SearchPlusResponse:
        context = ctx or RetrievalContext.create(budget=budget)
        base = await self._search.search(request, ctx=context, budget=budget, trace=trace)
        identity = base.embedding_identity

        cache_key = (request.query, identity.key())
        qvec = query_embedding or context.get_cached_query_embedding(cache_key)
        if qvec is None:
            provider = self._embeddings
            if provider is None:
                provider = self._search._embeddings
            embedded = await context.await_with_deadline(provider.embed_text(request.query))
            qvec = embedded.vectors[0]
            context.cache_query_embedding(cache_key, qvec)

        seed_uris = tuple(h.uri for h in base.candidates)
        active = await self._edges.count_active()
        cold = active == 0

        result: ExpansionResult = await expand_experience_edges(
            seed_uris=seed_uris,
            query_embedding=list(qvec),
            identity=identity,
            edges=self._edges,
            limits=self._limits,
        )

        hits: list[ExpansionHit] = []
        for act in result.activated:
            hits.append(
                ExpansionHit(
                    seed_uri=act.edge.source_uri,
                    target_uri=act.edge.target_uri,
                    edge_id=act.edge.id,
                    similarity=act.query_similarity,
                    hop=act.hop,
                    historical_query=act.payload.query_text,
                    payload_id=act.payload.id,
                )
            )

        if trace is not None:
            trace.record(
                "edge_expand",
                seed_count=len(seed_uris),
                expanded=len(result.expanded_uris),
                activated=result.edges_activated,
                cold_path=cold,
                hops=result.hops_taken,
            )

        logger.info(
            "search_plus_completed",
            seeds=len(seed_uris),
            expansions=len(hits),
            edges_activated=result.edges_activated,
            cold_path=cold,
            truncated=result.truncated,
            query_id=str(base.query_id),
        )
        return SearchPlusResponse(
            base=base,
            expansions=tuple(hits),
            truncated=result.truncated,
            hops_used=result.hops_taken,
            edges_activated=result.edges_activated,
            cold_path=cold,
            query_embedding=list(qvec),
            embedding_identity=identity,
            metadata={
                "visited_expanded": list(result.expanded_uris),
                "truncation_reason": result.truncation_reason,
            },
        )


def build_search_plus(settings: Any, database: Any) -> ExperienceAugmentedSearch:
    """Factory used by the answers router."""
    from vikingrag.providers.factory import build_embedding_provider, build_reranker

    emb = build_embedding_provider(settings)
    search = SemanticSearchService(
        database=database,
        embedding_provider=emb,
        retrieval_settings=settings.retrieval,
        embedding_settings=settings.embedding,
        reranker=build_reranker(settings) if settings.retrieval.rerank_enabled else None,
    )
    limits = ExperienceExpansionLimits(
        gamma=settings.retrieval.experience_gamma,
        max_hops=settings.retrieval.experience_max_hops,
        max_nodes=getattr(settings.retrieval, "experience_max_nodes", 32),
        max_edges=settings.retrieval.experience_max_edges,
    )
    return ExperienceAugmentedSearch(
        search=search,
        edge_repo_factory=SqlExperienceEdgeRepository,
        embedding_provider=emb,
        limits=limits,
    )
