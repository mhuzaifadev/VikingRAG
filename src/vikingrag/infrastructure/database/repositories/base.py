"""Repository contracts used by application/domain layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from vikingrag.domain.models.document import AbstractStatus, DocumentId, DocumentStatus, NodeId
from vikingrag.domain.models.node import DocumentNode, TreeNode


@dataclass(slots=True)
class DocumentRecord:
    id: DocumentId
    name: str
    mime_type: str
    content_hash: str
    status: DocumentStatus
    external_id: str | None = None
    byte_size: int | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    object_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@runtime_checkable
class DocumentRepository(Protocol):
    async def create(self, record: DocumentRecord) -> DocumentRecord: ...

    async def get(self, document_id: DocumentId) -> DocumentRecord | None: ...

    async def get_by_content_hash(self, content_hash: str) -> DocumentRecord | None: ...

    async def get_by_external_id(self, external_id: str) -> DocumentRecord | None: ...

    async def update(self, record: DocumentRecord) -> DocumentRecord: ...

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[DocumentRecord]: ...

    async def delete(self, document_id: DocumentId) -> bool: ...


@runtime_checkable
class NodeRepository(Protocol):
    async def bulk_create(self, nodes: list[DocumentNode]) -> list[DocumentNode]: ...

    async def get(self, node_id: NodeId) -> DocumentNode | None: ...

    async def get_by_uri(self, uri: str) -> DocumentNode | None: ...

    async def delete_for_document(self, document_id: DocumentId) -> int: ...

    async def list_children(self, parent_id: NodeId) -> list[DocumentNode]: ...

    async def list_by_document(self, document_id: DocumentId) -> list[DocumentNode]: ...

    async def update_abstract(
        self,
        node_id: NodeId,
        *,
        abstract_text: str,
        abstract_status: AbstractStatus,
    ) -> DocumentNode | None: ...

    async def list_ancestors(self, node_id: NodeId) -> list[DocumentNode]: ...

    async def list_descendants(self, node_id: NodeId) -> list[DocumentNode]: ...

    async def get_root(self, document_id: DocumentId) -> DocumentNode | None: ...

    async def build_tree(
        self,
        document_id: DocumentId,
        *,
        include_chunks: bool = False,
    ) -> TreeNode | None: ...
