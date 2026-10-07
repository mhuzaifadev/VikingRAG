"""Disabled / no-op reranker."""

from __future__ import annotations

from vikingrag.providers.reranking.base import RerankHit


class NoOpReranker:
    """Returns the input order unchanged (optionally truncated)."""

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankHit]:
        del query
        limit = len(documents) if top_n is None else min(top_n, len(documents))
        return [RerankHit(index=i, score=float(limit - i), text=documents[i]) for i in range(limit)]
