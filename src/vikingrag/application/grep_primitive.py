"""Scoped lexical Grep - literal (and bounded pattern) match on authoritative content."""

from __future__ import annotations

import re
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
# Reject catastrophic / unsupported regex constructs for pattern mode
_UNSAFE_PATTERN = re.compile(r"(\(\?[<!=]|\\[1-9]|\{(?:\d+,\d{3,}|\d{4,})\}|\*\*|\[\^[^\]]{20,}\])")


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
        context.require_scope_not_empty()
        async with context.tool_call("grep"):
            context.check_deadline()
            started = time.perf_counter()
            parsed = VikingURIParser.parse(request.uri)
            mode = getattr(request, "mode", "literal") or "literal"
            if mode not in ("literal", "pattern"):
                raise ValidationDomainError("grep mode must be 'literal' or 'pattern'")
            if mode == "pattern":
                _validate_safe_pattern(request.pattern)

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

                after_depth = -1
                after_ordinal = -1
                after_node_id: UUID | None = None
                after_offset = -1
                if request.cursor:
                    cursor = decode_cursor(request.cursor)
                    if str(cursor.get("scope_uri")) != request.uri:
                        raise ValidationDomainError("Cursor scope does not match request URI")
                    if cursor.get("pattern") != request.pattern:
                        raise ValidationDomainError("Cursor pattern mismatch")
                    after_depth = int(cursor.get("depth", -1))
                    after_ordinal = int(cursor.get("ordinal", -1))
                    after_node_id = require_uuid(cursor["node_id"], field="node_id")
                    after_offset = int(cursor["start_offset"])

                rows, nodes_visited, incomplete_scan = await _scoped_content_nodes(
                    session,
                    scope_node_id=scope_node.id,
                    document_id=scope_node.document_id,
                    pattern=request.pattern,
                    case_sensitive=request.case_sensitive,
                    mode=mode,
                    max_descendants=request.max_descendants,
                    after_depth=after_depth,
                    after_ordinal=after_ordinal,
                    after_node_id=after_node_id,
                    after_offset=after_offset,
                )
                await context.add_usage(nodes_inspected=nodes_visited)

            matches: list[GrepMatch] = []
            for row in rows:
                content = str(row["content"] or "")
                if mode == "pattern":
                    positions = _find_pattern_positions(
                        content, request.pattern, case_sensitive=request.case_sensitive
                    )
                else:
                    positions = _find_literal_positions(
                        content, request.pattern, case_sensitive=request.case_sensitive
                    )
                depth = int(str(row["depth"]))
                ordinal = int(str(row["ordinal"]))
                nid = UUID(str(row["id"]))
                for start, end in positions:
                    # Keyset filter matching SQL order (depth, ordinal, id, start_offset)
                    if after_node_id is not None and (depth, ordinal, nid, start) <= (
                        after_depth,
                        after_ordinal,
                        after_node_id,
                        after_offset,
                    ):
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
            if (truncated or incomplete_scan) and page:
                last = page[-1]
                # Resolve depth/ordinal from last match's row
                last_row = next(r for r in rows if UUID(str(r["id"])) == last.node_id)
                next_cursor = encode_cursor(
                    {
                        "scope_uri": request.uri,
                        "pattern": request.pattern,
                        "depth": int(str(last_row["depth"])),
                        "ordinal": int(str(last_row["ordinal"])),
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
                incomplete_scan=incomplete_scan,
                nodes_inspected=nodes_visited,
                latency_ms=elapsed,
            )
            return GrepResponse(
                uri=request.uri,
                pattern=request.pattern,
                case_sensitive=request.case_sensitive,
                matches=tuple(page),
                next_cursor=next_cursor,
                truncated=truncated or incomplete_scan,
                nodes_inspected=nodes_visited,
            )


def _validate_safe_pattern(pattern: str) -> None:
    if not pattern or len(pattern) > 256:
        raise ValidationDomainError("pattern must be 1..256 characters")
    if _UNSAFE_PATTERN.search(pattern):
        raise ValidationDomainError("Unsupported or unsafe regex pattern")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise ValidationDomainError(f"Invalid regex pattern: {exc}") from exc


async def _scoped_content_nodes(
    session: AsyncSession,
    *,
    scope_node_id: NodeId,
    document_id: DocumentId,
    pattern: str,
    case_sensitive: bool,
    mode: str,
    max_descendants: int,
    after_depth: int,
    after_ordinal: int,
    after_node_id: UUID | None,
    after_offset: int,
) -> tuple[list[dict[str, object]], int, bool]:
    """Load candidate nodes; cap CTE visitation; signal incomplete_scan."""
    # Cap visited nodes in the CTE, not only matching rows.
    # Cursor keyset clause is omitted when after_node_id is None so asyncpg
    # never sees an untyped NULL uuid bind.
    cursor_clause = ""
    params: dict[str, object] = {
        "scope_id": scope_node_id,
        "document_id": document_id,
        "pattern": pattern,
        "case_sensitive": case_sensitive,
        "mode": mode,
        "visit_cap": max(max_descendants, 1),
        "limit": max_descendants,
    }
    if after_node_id is not None:
        cursor_clause = """
          AND (
            (n.depth, n.ordinal, n.id) > (:after_depth, :after_ordinal, CAST(:after_node_id AS uuid))
            OR n.id = CAST(:after_node_id AS uuid)
          )
        """
        params["after_depth"] = after_depth
        params["after_ordinal"] = after_ordinal
        params["after_node_id"] = str(after_node_id)

    sql = text(
        f"""
        WITH RECURSIVE scope AS (
            SELECT id, depth, ordinal, 1 AS visited
            FROM document_nodes
            WHERE id = :scope_id AND document_id = :document_id
            UNION ALL
            SELECT c.id, c.depth, c.ordinal, s.visited + 1
            FROM document_nodes c
            INNER JOIN scope s ON c.parent_id = s.id
            WHERE c.document_id = :document_id
              AND s.visited < :visit_cap
        )
        SELECT n.id, n.document_id, n.uri, n.node_type, n.title, n.content,
               n.content_hash, n.depth, n.ordinal,
               (SELECT COUNT(*) FROM scope) AS visited_count
        FROM document_nodes n
        INNER JOIN scope s ON n.id = s.id
        WHERE n.content IS NOT NULL
          AND length(n.content) > 0
          AND (
            CASE WHEN :mode = 'pattern' THEN TRUE
            WHEN :case_sensitive THEN
              strpos(n.content, :pattern) > 0
            ELSE
              strpos(lower(n.content), lower(:pattern)) > 0
            END
          )
          {cursor_clause}
        ORDER BY n.depth ASC, n.ordinal ASC, n.id ASC
        LIMIT :limit
        """
    )
    await session.execute(text("SET LOCAL statement_timeout = '5000'"))
    visit_cap = max(max_descendants, 1)
    result = await session.execute(sql, params)
    rows = [dict(r) for r in result.mappings().all()]
    visited = int(rows[0]["visited_count"]) if rows else 0
    # incomplete if visit cap likely hit (visited == visit_cap) or row limit filled
    incomplete = visited >= visit_cap or len(rows) >= max_descendants
    # Filter same-node offsets in Python (SQL can't know match offsets for pattern)
    if after_node_id is not None:
        filtered: list[dict[str, object]] = []
        for row in rows:
            nid = UUID(str(row["id"]))
            if nid == after_node_id:
                filtered.append(row)  # offset filter applied in match loop
            else:
                filtered.append(row)
        rows = filtered
    del after_offset  # applied in match loop
    return rows, visited or len(rows), incomplete


def _casefold_index_map(text: str) -> tuple[str, list[int]]:
    """Casefold with NFKD; drop combining marks so İ≈i without length drift bugs.

    Returns folded string and map from folded index → original code-point index.
    """
    import unicodedata

    folded_chars: list[str] = []
    index_map: list[int] = []
    for i, ch in enumerate(text):
        for fc in unicodedata.normalize("NFKD", ch).casefold():
            if unicodedata.combining(fc):
                continue
            folded_chars.append(fc)
            index_map.append(i)
    return "".join(folded_chars), index_map


def _fold_needle(pattern: str) -> str:
    import unicodedata

    return "".join(
        fc
        for ch in pattern
        for fc in unicodedata.normalize("NFKD", ch).casefold()
        if not unicodedata.combining(fc)
    )


def _find_literal_positions(
    content: str, pattern: str, *, case_sensitive: bool
) -> list[tuple[int, int]]:
    if not pattern:
        return []
    if case_sensitive:
        positions: list[tuple[int, int]] = []
        start = 0
        while True:
            idx = content.find(pattern, start)
            if idx < 0:
                break
            positions.append((idx, idx + len(pattern)))
            start = idx + max(1, len(pattern))
        return positions

    haystack, index_map = _casefold_index_map(content)
    needle = _fold_needle(pattern)
    if not needle or not index_map:
        return []
    positions = []
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        end_folded = idx + len(needle) - 1
        orig_start = index_map[idx]
        orig_end = index_map[end_folded] + 1
        positions.append((orig_start, orig_end))
        start = idx + max(1, len(needle))
    return positions


def _find_pattern_positions(
    content: str, pattern: str, *, case_sensitive: bool
) -> list[tuple[int, int]]:
    flags = 0 if case_sensitive else re.IGNORECASE
    compiled = re.compile(pattern, flags)
    return [(m.start(), m.end()) for m in compiled.finditer(content)]


def _excerpt(content: str, start: int, end: int) -> str:
    left = max(0, start - _EXCERPT_RADIUS)
    right = min(len(content), end + _EXCERPT_RADIUS)
    prefix = "…" if left > 0 else ""
    suffix = "…" if right < len(content) else ""
    return f"{prefix}{content[left:right]}{suffix}"
