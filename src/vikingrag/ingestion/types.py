"""Typed pipeline DTOs - no ORM / FastAPI imports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from vikingrag.domain.models.document import DocumentId, IngestionStrategy, NodeType


@dataclass(slots=True)
class IngestionInput:
    filename: str
    mime_type: str
    content: bytes
    external_id: str | None = None
    strategy: IngestionStrategy = IngestionStrategy.SKIP_IDENTICAL
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LoadedDocument:
    filename: str
    mime_type: str
    content: bytes
    content_hash: str
    byte_size: int
    external_id: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class ContentBlock:
    """Normalized structural block from a parser."""

    text: str
    level: int  # 0 = body, 1 = H1/section, 2 = H2, ...
    title: str | None = None
    page: int | None = None
    offset_start: int | None = None
    offset_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ParsedDocument:
    title: str
    blocks: list[ContentBlock]
    parser_name: str
    parser_version: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HierarchyDraftNode:
    """In-memory hierarchy before persistence / URI finalization."""

    temp_id: UUID
    parent_temp_id: UUID | None
    node_type: NodeType
    title: str | None
    ordinal: int
    depth: int
    text: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HierarchyDraft:
    document_title: str
    nodes: list[HierarchyDraftNode]


@dataclass(slots=True)
class ChunkDraft:
    parent_temp_id: UUID
    ordinal: int
    text: str
    token_count: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IngestionResult:
    document_id: DocumentId
    content_hash: str
    status: str
    skipped: bool
    strategy: IngestionStrategy
    structural_node_count: int
    chunk_count: int
    ingestion_id: str
    duration_ms: float
    stage_durations_ms: dict[str, float] = field(default_factory=dict)
