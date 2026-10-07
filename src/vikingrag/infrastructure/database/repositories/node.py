"""SQLAlchemy async document node repository with CTE traversal."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from uuid import UUID

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from vikingrag.domain.models.document import AbstractStatus, DocumentId, NodeId, NodeType
from vikingrag.domain.models.node import DocumentNode, TreeNode
from vikingrag.infrastructure.database.models import DocumentNodeRow


def _to_node(row: DocumentNodeRow) -> DocumentNode:
    path = tuple(UUID(str(x)) for x in (row.path_ids or []))
    return DocumentNode(
        id=NodeId(row.id if isinstance(row.id, UUID) else UUID(str(row.id))),
        document_id=DocumentId(
            row.document_id if isinstance(row.document_id, UUID) else UUID(str(row.document_id))
        ),
        parent_id=(
            None
            if row.parent_id is None
            else NodeId(
                row.parent_id if isinstance(row.parent_id, UUID) else UUID(str(row.parent_id))
            )
        ),
        node_type=NodeType(row.node_type),
        title=row.title,
        ordinal=row.ordinal,
        depth=row.depth,
        uri=row.uri,
        content=row.content,
        content_hash=row.content_hash,
        token_count=row.token_count,
        path_ids=path,
        metadata=dict(row.metadata_ or {}),
        abstract_text=row.abstract_text,
        abstract_status=AbstractStatus(row.abstract_status),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _mapping_to_node(row: Mapping[str, Any]) -> DocumentNode:
    path_raw = row.get("path_ids") or []
    path = tuple(UUID(str(x)) for x in path_raw)
    parent_raw = row.get("parent_id")
    return DocumentNode(
        id=NodeId(UUID(str(row["id"]))),
        document_id=DocumentId(UUID(str(row["document_id"]))),
        parent_id=None if parent_raw is None else NodeId(UUID(str(parent_raw))),
        node_type=NodeType(str(row["node_type"])),
        title=row.get("title"),
        ordinal=int(row["ordinal"]),
        depth=int(row["depth"]),
        uri=str(row["uri"]),
        content=row.get("content"),
        content_hash=str(row["content_hash"]),
        token_count=row.get("token_count"),
        path_ids=path,
        metadata=dict(row.get("metadata") or {}),
        abstract_text=row.get("abstract_text"),
        abstract_status=AbstractStatus(str(row.get("abstract_status") or "none")),
        created_at=row.get("created_at"),
        updated_at=row.get("updated_at"),
    )


def _to_tree_node(node: DocumentNode) -> TreeNode:
    return TreeNode(
        id=node.id,
        document_id=node.document_id,
        parent_id=node.parent_id,
        node_type=node.node_type,
        title=node.title,
        ordinal=node.ordinal,
        depth=node.depth,
        uri=node.uri,
        token_count=node.token_count,
        content_hash=node.content_hash,
        path_ids=node.path_ids,
        metadata=dict(node.metadata),
        abstract_status=node.abstract_status,
        children=[],
        has_content=bool(node.content),
    )


class SqlNodeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def bulk_create(self, nodes: list[DocumentNode]) -> list[DocumentNode]:
        rows: list[DocumentNodeRow] = []
        for node in nodes:
            rows.append(
                DocumentNodeRow(
                    id=node.id,
                    document_id=node.document_id,
                    parent_id=node.parent_id,
                    node_type=node.node_type.value,
                    title=node.title,
                    ordinal=node.ordinal,
                    depth=node.depth,
                    uri=node.uri,
                    content=node.content,
                    content_hash=node.content_hash,
                    token_count=node.token_count,
                    path_ids=list(node.path_ids),
                    abstract_text=node.abstract_text,
                    abstract_status=node.abstract_status.value,
                    metadata_=dict(node.metadata),
                )
            )
        self._session.add_all(rows)
        await self._session.flush()
        return nodes

    async def get(self, node_id: NodeId) -> DocumentNode | None:
        row = await self._session.get(DocumentNodeRow, node_id)
        return _to_node(row) if row else None

    async def get_by_uri(self, uri: str) -> DocumentNode | None:
        stmt = select(DocumentNodeRow).where(DocumentNodeRow.uri == uri).limit(1)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_node(row) if row else None

    async def delete_for_document(self, document_id: DocumentId) -> int:
        stmt = delete(DocumentNodeRow).where(DocumentNodeRow.document_id == document_id)
        result = await self._session.execute(stmt)
        rowcount = result.rowcount  # type: ignore[attr-defined]
        return int(rowcount or 0)

    async def list_children(self, parent_id: NodeId) -> list[DocumentNode]:
        stmt = (
            select(DocumentNodeRow)
            .where(DocumentNodeRow.parent_id == parent_id)
            .order_by(DocumentNodeRow.ordinal.asc())
        )
        result = await self._session.execute(stmt)
        return [_to_node(row) for row in result.scalars().all()]

    async def list_by_document(self, document_id: DocumentId) -> list[DocumentNode]:
        stmt = (
            select(DocumentNodeRow)
            .where(DocumentNodeRow.document_id == document_id)
            .order_by(DocumentNodeRow.depth.asc(), DocumentNodeRow.ordinal.asc())
        )
        result = await self._session.execute(stmt)
        return [_to_node(row) for row in result.scalars().all()]

    async def get_root(self, document_id: DocumentId) -> DocumentNode | None:
        stmt = (
            select(DocumentNodeRow)
            .where(DocumentNodeRow.document_id == document_id)
            .where(DocumentNodeRow.parent_id.is_(None))
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_node(row) if row else None

    async def list_ancestors(self, node_id: NodeId) -> list[DocumentNode]:
        sql = text(
            """
            WITH RECURSIVE ancestors AS (
                SELECT n.id, n.parent_id, 0 AS walk_depth
                FROM document_nodes n
                WHERE n.id = :node_id
                UNION ALL
                SELECT p.id, p.parent_id, a.walk_depth + 1
                FROM document_nodes p
                INNER JOIN ancestors a ON p.id = a.parent_id
            )
            SELECT n.*
            FROM ancestors a
            INNER JOIN document_nodes n ON n.id = a.id
            WHERE a.id <> :node_id
            ORDER BY a.walk_depth DESC
            """
        )
        result = await self._session.execute(sql, {"node_id": node_id})
        rows = result.mappings().all()
        return [_mapping_to_node(dict(row)) for row in rows]

    async def list_descendants(self, node_id: NodeId) -> list[DocumentNode]:
        sql = text(
            """
            WITH RECURSIVE descendants AS (
                SELECT n.id
                FROM document_nodes n
                WHERE n.parent_id = :node_id
                UNION ALL
                SELECT c.id
                FROM document_nodes c
                INNER JOIN descendants d ON c.parent_id = d.id
            )
            SELECT n.*
            FROM descendants d
            INNER JOIN document_nodes n ON n.id = d.id
            ORDER BY n.depth ASC, n.ordinal ASC
            """
        )
        result = await self._session.execute(sql, {"node_id": node_id})
        rows = result.mappings().all()
        return [_mapping_to_node(dict(row)) for row in rows]

    async def build_tree(
        self,
        document_id: DocumentId,
        *,
        include_chunks: bool = False,
    ) -> TreeNode | None:
        """Load all nodes for a document in one query and assemble the tree in memory."""
        nodes = await self.list_by_document(document_id)
        if not include_chunks:
            nodes = [n for n in nodes if n.node_type is not NodeType.CHUNK]
        if not nodes:
            return None

        by_id = {n.id: _to_tree_node(n) for n in nodes}
        root: TreeNode | None = None
        for node in nodes:
            tree_node = by_id[node.id]
            if node.parent_id is None:
                root = tree_node
                continue
            parent = by_id.get(node.parent_id)
            if parent is not None:
                parent.children.append(tree_node)
        if root is not None:
            _sort_tree(root)
        return root


def _sort_tree(node: TreeNode) -> None:
    node.children.sort(key=lambda c: c.ordinal)
    for child in node.children:
        _sort_tree(child)
