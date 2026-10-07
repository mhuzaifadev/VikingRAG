"""Hierarchical document node domain models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import UUID

from vikingrag.domain.models.document import AbstractStatus, DocumentId, NodeId, NodeType


@dataclass(slots=True)
class DocumentNode:
    id: NodeId
    document_id: DocumentId
    parent_id: NodeId | None
    node_type: NodeType
    title: str | None
    ordinal: int
    depth: int
    uri: str
    content: str | None
    content_hash: str
    token_count: int | None
    path_ids: tuple[UUID, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    abstract_text: str | None = None
    abstract_status: AbstractStatus = AbstractStatus.NONE
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class TreeNode:
    """Structural tree entry - content omitted by default for API/tree views."""

    id: NodeId
    document_id: DocumentId
    parent_id: NodeId | None
    node_type: NodeType
    title: str | None
    ordinal: int
    depth: int
    uri: str
    token_count: int | None
    content_hash: str
    path_ids: tuple[UUID, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    abstract_status: AbstractStatus = AbstractStatus.NONE
    children: list[TreeNode] = field(default_factory=list)
    has_content: bool = False
