"""Structural navigation service - powers future List/Read primitives."""

from __future__ import annotations

from vikingrag.domain.errors import DocumentNotFound, InvalidVikingURI, NodeNotFound
from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.node import DocumentNode, TreeNode
from vikingrag.domain.uri import VikingURI, VikingURIKind, VikingURIParser
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository


class StructuralNavigationService:
    def __init__(
        self,
        *,
        documents: SqlDocumentRepository,
        nodes: SqlNodeRepository,
    ) -> None:
        self._documents = documents
        self._nodes = nodes

    async def get_node(self, node_id: NodeId) -> DocumentNode:
        node = await self._nodes.get(node_id)
        if node is None:
            raise NodeNotFound(f"Node not found: {node_id}")
        return node

    async def get_parent(self, node_id: NodeId) -> DocumentNode | None:
        node = await self.get_node(node_id)
        if node.parent_id is None:
            return None
        parent = await self._nodes.get(node.parent_id)
        if parent is None:
            raise NodeNotFound(f"Parent node missing for: {node_id}")
        return parent

    async def list_children(self, node_id: NodeId) -> list[DocumentNode]:
        await self.get_node(node_id)
        return await self._nodes.list_children(node_id)

    async def list_ancestors(self, node_id: NodeId) -> list[DocumentNode]:
        await self.get_node(node_id)
        return await self._nodes.list_ancestors(node_id)

    async def list_descendants(self, node_id: NodeId) -> list[DocumentNode]:
        await self.get_node(node_id)
        return await self._nodes.list_descendants(node_id)

    async def resolve_uri(self, uri: str) -> DocumentNode | VikingURI:
        parsed = VikingURIParser.parse(uri)
        if parsed.kind is VikingURIKind.DOCUMENT:
            root = await self._nodes.get_root(parsed.document_id)
            if root is None:
                doc = await self._documents.get(parsed.document_id)
                if doc is None:
                    raise DocumentNotFound(f"Document not found: {parsed.document_id}")
                raise NodeNotFound(f"Document root node missing: {parsed.document_id}")
            return root
        assert parsed.node_id is not None
        node = await self._nodes.get_by_uri(uri)
        if node is None:
            # Fall back to id lookup if URI string differs only by normalization
            node = await self._nodes.get(parsed.node_id)
        if node is None:
            raise NodeNotFound(f"No node for URI: {uri}")
        if node.document_id != parsed.document_id:
            raise InvalidVikingURI("URI document_id does not match stored node")
        return node

    async def get_document_tree(
        self,
        document_id: DocumentId,
        *,
        include_chunks: bool = False,
    ) -> TreeNode:
        doc = await self._documents.get(document_id)
        if doc is None:
            raise DocumentNotFound(f"Document not found: {document_id}")
        tree = await self._nodes.build_tree(document_id, include_chunks=include_chunks)
        if tree is None:
            raise NodeNotFound(f"No hierarchy for document: {document_id}")
        return tree
