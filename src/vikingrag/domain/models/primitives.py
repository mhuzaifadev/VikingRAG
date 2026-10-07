"""Typed contracts for Search companions: List, Grep, Read."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vikingrag.domain.models.document import DocumentId, NodeId, NodeType


class OffsetSystem(StrEnum):
    UNICODE_CODE_POINT = "unicode_code_point"


@dataclass(frozen=True, slots=True)
class ListRequest:
    uri: str
    limit: int = 50
    cursor: str | None = None

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("ListRequest.uri must be non-empty")
        if self.limit < 1:
            raise ValueError("ListRequest.limit must be >= 1")


@dataclass(frozen=True, slots=True)
class ListItem:
    node_id: NodeId
    document_id: DocumentId
    uri: str
    parent_uri: str | None
    node_type: NodeType
    title: str | None
    ordinal: int
    has_content: bool
    token_count: int | None = None


@dataclass(frozen=True, slots=True)
class ListResponse:
    uri: str
    parent_node_id: NodeId | None
    document_id: DocumentId
    items: tuple[ListItem, ...]
    next_cursor: str | None
    truncated: bool
    total_children: int


@dataclass(frozen=True, slots=True)
class GrepRequest:
    uri: str
    pattern: str
    case_sensitive: bool = True
    max_matches: int = 20
    max_descendants: int = 500
    cursor: str | None = None

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("GrepRequest.uri must be non-empty")
        if not self.pattern:
            raise ValueError("GrepRequest.pattern must be non-empty")
        if self.max_matches < 1:
            raise ValueError("GrepRequest.max_matches must be >= 1")
        if self.max_descendants < 1:
            raise ValueError("GrepRequest.max_descendants must be >= 1")


@dataclass(frozen=True, slots=True)
class GrepMatch:
    uri: str
    document_id: DocumentId
    node_id: NodeId
    node_type: NodeType
    title: str | None
    excerpt: str
    start_offset: int
    end_offset: int
    offset_system: OffsetSystem = OffsetSystem.UNICODE_CODE_POINT
    content_hash: str = ""


@dataclass(frozen=True, slots=True)
class GrepResponse:
    uri: str
    pattern: str
    case_sensitive: bool
    matches: tuple[GrepMatch, ...]
    next_cursor: str | None
    truncated: bool
    nodes_inspected: int


@dataclass(frozen=True, slots=True)
class ReadRequest:
    uri: str
    start_offset: int = 0
    max_tokens: int = 512
    expected_content_hash: str | None = None

    def __post_init__(self) -> None:
        if not self.uri.strip():
            raise ValueError("ReadRequest.uri must be non-empty")
        if self.start_offset < 0:
            raise ValueError("ReadRequest.start_offset must be >= 0")
        if self.max_tokens < 1:
            raise ValueError("ReadRequest.max_tokens must be >= 1")


@dataclass(frozen=True, slots=True)
class ReadResponse:
    uri: str
    document_id: DocumentId
    node_id: NodeId
    node_type: NodeType
    title: str | None
    content_hash: str
    has_direct_content: bool
    text: str
    start_offset: int
    end_offset: int
    offset_system: OffsetSystem
    token_count: int
    truncated: bool
    next_offset: int | None
    source_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrimitiveTiming:
    total_ms: float = 0.0
    db_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class PrimitiveUsage:
    tool_calls: int = 1
    nodes_inspected: int = 0
    tokens_read: int = 0
    db_operations: int = 0
