"""Invalidate experience edges when document endpoints are deleted or replaced."""

from __future__ import annotations

from vikingrag.domain.models.document import DocumentId
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.experience import SqlExperienceEdgeRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


async def invalidate_edges_for_uris(
    database: Database,
    uris: set[str],
) -> int:
    """Mark active edges whose source or target is in ``uris`` as INVALIDATED."""
    if not uris:
        return 0
    async with database.session() as session:
        repo = SqlExperienceEdgeRepository(session)
        n = await repo.invalidate_by_uris(uris)
    logger.info("experience_edges_invalidated", uri_count=len(uris), edges=n)
    return n


async def invalidate_edges_for_document(
    database: Database,
    document_id: DocumentId,
) -> int:
    """Collect all node URIs for a document and invalidate incident edges."""
    async with database.session() as session:
        nodes = SqlNodeRepository(session)
        tree_nodes = await nodes.list_by_document(document_id)
        uris = {n.uri for n in tree_nodes if n.uri}
        if not uris:
            return 0
        repo = SqlExperienceEdgeRepository(session)
        n = await repo.invalidate_by_uris(uris)
    logger.info(
        "experience_edges_invalidated_for_document",
        document_id=str(document_id),
        uri_count=len(uris),
        edges=n,
    )
    return n
