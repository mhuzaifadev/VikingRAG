"""Pydantic API schemas for documents / nodes / trees."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from vikingrag.domain.models.document import IngestionStrategy
from vikingrag.domain.models.node import DocumentNode, TreeNode
from vikingrag.infrastructure.database.repositories.base import DocumentRecord
from vikingrag.ingestion.types import IngestionResult


class IngestDocumentResponse(BaseModel):
    document_id: UUID
    content_hash: str
    status: str
    skipped: bool
    strategy: IngestionStrategy
    structural_node_count: int
    chunk_count: int
    ingestion_id: str
    duration_ms: float
    stage_durations_ms: dict[str, float] = Field(default_factory=dict)

    @classmethod
    def from_result(cls, result: IngestionResult) -> IngestDocumentResponse:
        return cls(
            document_id=result.document_id,
            content_hash=result.content_hash,
            status=result.status,
            skipped=result.skipped,
            strategy=result.strategy,
            structural_node_count=result.structural_node_count,
            chunk_count=result.chunk_count,
            ingestion_id=result.ingestion_id,
            duration_ms=result.duration_ms,
            stage_durations_ms=result.stage_durations_ms,
        )


class DocumentResponse(BaseModel):
    id: UUID
    name: str
    mime_type: str
    content_hash: str
    status: str
    external_id: str | None = None
    byte_size: int | None = None
    parser_name: str | None = None
    parser_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None

    @classmethod
    def from_record(cls, record: DocumentRecord) -> DocumentResponse:
        return cls(
            id=record.id,
            name=record.name,
            mime_type=record.mime_type,
            content_hash=record.content_hash,
            status=record.status.value,
            external_id=record.external_id,
            byte_size=record.byte_size,
            parser_name=record.parser_name,
            parser_version=record.parser_version,
            metadata=record.metadata,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )


class NodeResponse(BaseModel):
    id: UUID
    document_id: UUID
    parent_id: UUID | None
    node_type: str
    title: str | None
    ordinal: int
    depth: int
    uri: str
    token_count: int | None
    content_hash: str
    path_ids: list[UUID]
    metadata: dict[str, Any] = Field(default_factory=dict)
    abstract_status: str
    content: str | None = None

    @classmethod
    def from_node(cls, node: DocumentNode, *, include_content: bool = False) -> NodeResponse:
        return cls(
            id=node.id,
            document_id=node.document_id,
            parent_id=node.parent_id,
            node_type=node.node_type.value,
            title=node.title,
            ordinal=node.ordinal,
            depth=node.depth,
            uri=node.uri,
            token_count=node.token_count,
            content_hash=node.content_hash,
            path_ids=list(node.path_ids),
            metadata=node.metadata,
            abstract_status=node.abstract_status.value,
            content=node.content if include_content else None,
        )


class TreeNodeResponse(BaseModel):
    id: UUID
    document_id: UUID
    parent_id: UUID | None
    node_type: str
    title: str | None
    ordinal: int
    depth: int
    uri: str
    token_count: int | None
    content_hash: str
    has_content: bool
    abstract_status: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    children: list[TreeNodeResponse] = Field(default_factory=list)

    @classmethod
    def from_tree(cls, node: TreeNode) -> TreeNodeResponse:
        return cls(
            id=node.id,
            document_id=node.document_id,
            parent_id=node.parent_id,
            node_type=node.node_type.value,
            title=node.title,
            ordinal=node.ordinal,
            depth=node.depth,
            uri=node.uri,
            token_count=node.token_count,
            content_hash=node.content_hash,
            has_content=node.has_content,
            abstract_status=node.abstract_status.value,
            metadata=node.metadata,
            children=[cls.from_tree(c) for c in node.children],
        )
