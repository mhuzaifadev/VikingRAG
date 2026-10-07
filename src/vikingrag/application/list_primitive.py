"""Structural List primitive - direct children only, compact metadata."""

from __future__ import annotations

import time

from vikingrag.application.budget import RetrievalContext
from vikingrag.application.cursors import decode_cursor, encode_cursor, require_uuid
from vikingrag.application.navigation import StructuralNavigationService
from vikingrag.domain.errors import ScopeDeniedError, ValidationDomainError
from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.primitives import ListItem, ListRequest, ListResponse
from vikingrag.domain.uri import VikingURIKind, VikingURIParser
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


class ListService:
    def __init__(self, *, database: Database) -> None:
        self._database = database

    async def list(
        self,
        request: ListRequest,
        *,
        ctx: RetrievalContext | None = None,
    ) -> ListResponse:
        context = ctx or RetrievalContext.create()
        async with context.tool_call("list"):
            context.check_deadline()
            started = time.perf_counter()
            parsed = VikingURIParser.parse(request.uri)

            async with self._database.session() as session:
                await context.reserve(db_operations=1)
                nav = StructuralNavigationService(
                    documents=SqlDocumentRepository(session),
                    nodes=SqlNodeRepository(session),
                )
                parent = await nav.resolve_uri(request.uri)
                document_id = parent.document_id
                if not context.document_allowed(document_id):
                    raise ScopeDeniedError(f"Document {document_id} outside permitted scope")

                if parsed.kind is VikingURIKind.NODE and parsed.document_id != document_id:
                    raise ValidationDomainError("URI document_id mismatch")

                children = await nav.list_children(parent.id)
                await context.add_usage(nodes_inspected=len(children) + 1)

            children = sorted(children, key=lambda n: (n.ordinal, str(n.id)))
            start_index = 0
            if request.cursor:
                cursor = decode_cursor(request.cursor)
                if str(cursor.get("parent_id")) != str(parent.id):
                    raise ValidationDomainError("Cursor parent does not match request URI")
                if cursor.get("ordering") != "ordinal,id":
                    raise ValidationDomainError("Cursor ordering mismatch")
                after_ordinal = int(cursor["ordinal"])
                after_id = require_uuid(cursor["node_id"], field="node_id")
                for i, child in enumerate(children):
                    if child.ordinal > after_ordinal or (
                        child.ordinal == after_ordinal and child.id > NodeId(after_id)
                    ):
                        start_index = i
                        break
                else:
                    start_index = len(children)

            page = children[start_index : start_index + request.limit]
            truncated = start_index + len(page) < len(children)
            next_cursor = None
            if truncated and page:
                last = page[-1]
                next_cursor = encode_cursor(
                    {
                        "parent_id": str(parent.id),
                        "ordering": "ordinal,id",
                        "ordinal": last.ordinal,
                        "node_id": str(last.id),
                    }
                )

            parent_uri = (
                None
                if parent.parent_id is None
                else VikingURIParser.build_node_uri(parent.document_id, parent.parent_id)
            )
            # List returns children; parent_uri on items is the listed parent
            listed_parent_uri = parent.uri
            items = tuple(
                ListItem(
                    node_id=c.id,
                    document_id=DocumentId(c.document_id),
                    uri=c.uri,
                    parent_uri=listed_parent_uri,
                    node_type=c.node_type,
                    title=c.title,
                    ordinal=c.ordinal,
                    has_content=bool(c.content and c.content.strip()),
                    token_count=c.token_count,
                )
                for c in page
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            logger.info(
                "list_completed",
                query_id=str(context.query_id),
                trace_id=context.trace_id,
                returned=len(items),
                total_children=len(children),
                truncated=truncated,
                latency_ms=elapsed,
            )
            del parent_uri
            return ListResponse(
                uri=request.uri,
                parent_node_id=parent.id,
                document_id=document_id,
                items=items,
                next_cursor=next_cursor,
                truncated=truncated,
                total_children=len(children),
            )
