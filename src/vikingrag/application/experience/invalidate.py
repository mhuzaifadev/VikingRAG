"""Invalidate experience edges when document endpoints are deleted or replaced."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.uri.viking_uri import VikingURIParser
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.experience import (
    SqlExperienceEdgeRepository,
    SqlQueryRunRepository,
)
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


async def invalidate_edges_for_uris_session(
    session: AsyncSession,
    uris: set[str],
) -> int:
    """Mark active edges whose source or target is in ``uris`` as INVALIDATED."""
    if not uris:
        return 0
    repo = SqlExperienceEdgeRepository(session)
    return await repo.invalidate_by_uris(uris)


async def collect_document_uris(
    session: AsyncSession,
    document_id: DocumentId,
) -> set[str]:
    """All node URIs for a document, always including the stable document-root URI."""
    nodes = SqlNodeRepository(session)
    tree_nodes = await nodes.list_by_document(document_id)
    uris = {n.uri for n in tree_nodes if n.uri}
    uris.add(VikingURIParser.build_document_uri(document_id))
    return uris


async def invalidate_edges_for_document_session(
    session: AsyncSession,
    document_id: DocumentId,
) -> int:
    """Invalidate incident edges for a document on the caller's session."""
    uris = await collect_document_uris(session, document_id)
    n = await invalidate_edges_for_uris_session(session, uris)
    logger.info(
        "experience_edges_invalidated_for_document",
        document_id=str(document_id),
        uri_count=len(uris),
        edges=n,
    )
    return n


async def skip_pending_edge_jobs_for_document_session(
    session: AsyncSession,
    document_id: DocumentId,
    *,
    uris: set[str] | None = None,
) -> int:
    """Mark PENDING edge-build jobs that touch this document as SKIPPED.

    Matches runs whose ``document_ids`` contain the document, or whose
    ``metadata.citation_uris`` intersect the document URI set.
    """
    uri_set = uris if uris is not None else await collect_document_uris(session, document_id)
    runs = SqlQueryRunRepository(session)
    return await runs.skip_pending_edge_jobs_for_document(
        document_id,
        citation_uris=uri_set,
    )


async def prepare_document_experience_retirement_session(
    session: AsyncSession,
    document_id: DocumentId,
) -> tuple[int, int]:
    """Invalidate incident edges and skip stale PENDING jobs (same session).

    Returns ``(edges_invalidated, jobs_skipped)``.
    """
    uris = await collect_document_uris(session, document_id)
    edges = await invalidate_edges_for_uris_session(session, uris)
    jobs = await skip_pending_edge_jobs_for_document_session(session, document_id, uris=uris)
    return edges, jobs


async def invalidate_edges_for_uris(
    database: Database,
    uris: set[str],
) -> int:
    """Mark active edges whose source or target is in ``uris`` as INVALIDATED."""
    if not uris:
        return 0
    async with database.session() as session:
        n = await invalidate_edges_for_uris_session(session, uris)
    logger.info("experience_edges_invalidated", uri_count=len(uris), edges=n)
    return n


async def invalidate_edges_for_document(
    database: Database,
    document_id: DocumentId,
) -> int:
    """Collect all node URIs for a document and invalidate incident edges."""
    async with database.session() as session:
        n = await invalidate_edges_for_document_session(session, document_id)
    return n
