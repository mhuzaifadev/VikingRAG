"""Scoped lexical Grep - literal substring match on authoritative node content."""

from __future__ import annotations

import time
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from vikingrag.application.budget import RetrievalContext
from vikingrag.application.cursors import decode_cursor, encode_cursor, require_uuid
from vikingrag.application.navigation import StructuralNavigationService
from vikingrag.domain.errors import ScopeDeniedError, ValidationDomainError
from vikingrag.domain.models.document import DocumentId, NodeId, NodeType
from vikingrag.domain.models.primitives import (
    GrepMatch,
    GrepRequest,
    GrepResponse,
    OffsetSystem,
)
from vikingrag.domain.uri import VikingURIParser
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)

_EXCERPT_RADIUS = 80


class GrepService:
    def __init__(self, *, database: Database) -> None:
        self._database = database

    async def grep(
        self,
        request: GrepRequest,
        *,
        ctx: RetrievalContext | None = None,
    ) -> GrepResponse:
        context = ctx or RetrievalContext.create()
        async with context.tool_call("grep"):
            context.check_deadline()
            started = time.perf_counter()
            parsed = VikingURIParser.parse(request.uri)

            async with self._database.session() as session:
                await context.reserve(db_operations=1)
                nav = StructuralNavigationService(
                    documents=SqlDocumentRepository(session),
                    nodes=SqlNodeRepository(session),
                )
                scope_node = await nav.resolve_uri(request.uri)
                if not context.document_allowed(scope_node.document_id):
                    raise ScopeDeniedError(
                        f"Document {scope_node.document_id} outside permitted scope"
                    )
                if parsed.document_id != scope_node.document_id:
                    raise ValidationDomainError("URI document_id mismatch")

                after_node_id: UUID | None = None
                after_offset = -1
                if request.cursor:
                    cursor = decode_cursor(request.cursor)
                    if str(cursor.get("scope_uri")) != request.uri:
                        raise ValidationDomainError("Cursor scope does not match request URI")
                    if cursor.get("pattern") != request.pattern:
                        raise ValidationDomainError("Cursor pattern mismatch")
                    after_node_id = require_uuid(cursor["node_id"], field="node_id")
                    after_offset = int(cursor["start_offset"])

                rows, nodes_inspected = await _scoped_literal_grep(
                    session,
                    scope_node_id=scope_node.id,
                    document_id=scope_node.document_id,
                    pattern=request.pattern,
                    case_sensitive=request.case_sensitive,
                    max_descendants=request.max_descendants,
                )
                await context.add_usage(nodes_inspected=nodes_inspected)

            matches: list[GrepMatch] = []
            for row in rows:
                content = str(row["content"] or "")
                positions = _find_literal_positions(
                    content, request.pattern, case_sensitive=request.case_sensitive
                )
                for start, end in positions:
                    if after_node_id is not None:
                        nid = UUID(str(row["id"]))
                        if nid < after_node_id or (nid == after_node_id and start <= after_offset):
                            continue
                    excerpt = _excerpt(content, start, end)
                    title_val = row.get("title")
                    title = str(title_val) if title_val is not None else None
                    matches.append(
                        GrepMatch(
                            uri=str(row["uri"]),
                            document_id=DocumentId(UUID(str(row["document_id"]))),
                            node_id=NodeId(UUID(str(row["id"]))),
                            node_type=NodeType(str(row["node_type"])),
                            title=title,
                            excerpt=excerpt,
                            start_offset=start,
                            end_offset=end,
                            offset_system=OffsetSystem.UNICODE_CODE_POINT,
                            content_hash=str(row["content_hash"]),
                        )
                    )
                    if len(matches) > request.max_matches:
                        break
                if len(matches) > request.max_matches:
                    break

            truncated = len(matches) > request.max_matches
            page = matches[: request.max_matches]
            next_cursor = None
            if truncated and page:
                last = page[-1]
                next_cursor = encode_cursor(
                    {
                        "scope_uri": request.uri,
                        "pattern": request.pattern,
                        "node_id": str(last.node_id),
                        "start_offset": last.start_offset,
                    }
                )

            elapsed = (time.perf_counter() - started) * 1000.0
            logger.info(
                "grep_completed",
                query_id=str(context.query_id),
                trace_id=context.trace_id,
                matches=len(page),
                truncated=truncated,
                nodes_inspected=nodes_inspected,
                latency_ms=elapsed,
            )
            return GrepResponse(
                uri=request.uri,
                pattern=request.pattern,
                case_sensitive=request.case_sensitive,
                matches=tuple(page),
                next_cursor=next_cursor,
                truncated=truncated,
                nodes_inspected=nodes_inspected,
            )


async def _scoped_literal_grep(
    session: AsyncSession,
    *,
    scope_node_id: NodeId,
    document_id: DocumentId,
    pattern: str,
    case_sensitive: bool,
    max_descendants: int,
) -> tuple[list[dict[str, object]], int]:
    """Load candidate nodes with content inside scope; matching done in Python for offsets.

    Scope restriction happens in SQL via recursive CTE. Statement limited by max_descendants.
    """
    # Include the scope node itself if it has content, plus descendants.
    sql = text(
        """
        WITH RECURSIVE scope AS (
            SELECT id FROM document_nodes WHERE id = :scope_id AND document_id = :document_id
            UNION ALL
            SELECT c.id
            FROM document_nodes c
            INNER JOIN scope s ON c.parent_id = s.id
            WHERE c.document_id = :document_id
        )
        SELECT n.id, n.document_id, n.uri, n.node_type, n.title, n.content, n.content_hash
        FROM document_nodes n
        INNER JOIN scope s ON n.id = s.id
        WHERE n.content IS NOT NULL
          AND length(n.content) > 0
          AND (
            CASE WHEN :case_sensitive THEN
              strpos(n.content, :pattern) > 0
            ELSE
              strpos(lower(n.content), lower(:pattern)) > 0
            END
          )
        ORDER BY n.depth ASC, n.ordinal ASC, n.id ASC
        LIMIT :limit
        """
    )
    # Set a local statement timeout for this transaction (ms)
    await session.execute(text("SET LOCAL statement_timeout = '5000'"))
    result = await session.execute(
        sql,
        {
            "scope_id": scope_node_id,
            "document_id": document_id,
            "pattern": pattern,
            "case_sensitive": case_sensitive,
            "limit": max_descendants,
        },
    )
    rows = [dict(r) for r in result.mappings().all()]
    return rows, len(rows)


def _find_literal_positions(
    content: str, pattern: str, *, case_sensitive: bool
) -> list[tuple[int, int]]:
    if not pattern:
        return []
    haystack = content if case_sensitive else content.lower()
    needle = pattern if case_sensitive else pattern.lower()
    positions: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        positions.append((idx, idx + len(pattern)))
        start = idx + max(1, len(needle))
    return positions


def _excerpt(content: str, start: int, end: int) -> str:
    left = max(0, start - _EXCERPT_RADIUS)
    right = min(len(content), end + _EXCERPT_RADIUS)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(content) else ""
    return f"{prefix}{content[left:right]}{suffix}"
