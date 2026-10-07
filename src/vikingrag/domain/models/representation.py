"""Semantic representation and embedding identity contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, NewType
from uuid import UUID

from vikingrag.domain.models.document import DocumentId, NodeId, NodeType

RepresentationId = NewType("RepresentationId", UUID)
EmbeddingId = NewType("EmbeddingId", UUID)


class RepresentationType(StrEnum):
    NODE_CONTENT = "node_content"
    NODE_SUMMARY = "node_summary"


class IndexStage(StrEnum):
    STRUCTURED = "structured"
    SUMMARIZING = "summarizing"
    SUMMARIZED = "summarized"
    EMBEDDING = "embedding"
    INDEXED = "indexed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EmbeddingIdentity:
    """Identity of an embedding space - incompatible identities must never be compared."""

    provider: str
    model: str
    dimensions: int
    version: str = "1"

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("EmbeddingIdentity.provider must be non-empty")
        if not self.model.strip():
            raise ValueError("EmbeddingIdentity.model must be non-empty")
        if self.dimensions < 1:
            raise ValueError("EmbeddingIdentity.dimensions must be >= 1")

    def key(self) -> str:
        return f"{self.provider}:{self.model}:{self.dimensions}:v{self.version}"

    def compatible_with(self, other: EmbeddingIdentity) -> bool:
        return (
            self.provider == other.provider
            and self.model == other.model
            and self.dimensions == other.dimensions
            and self.version == other.version
        )


@dataclass(slots=True)
class NodeRepresentation:
    id: RepresentationId
    node_id: NodeId
    document_id: DocumentId
    representation_type: RepresentationType
    content: str
    content_hash: str
    generator: str
    generator_model: str
    version: str
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


@dataclass(slots=True)
class NodeEmbedding:
    id: EmbeddingId
    representation_id: RepresentationId
    node_id: NodeId
    document_id: DocumentId
    identity: EmbeddingIdentity
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime | None = None

    def __post_init__(self) -> None:
        if len(self.embedding) != self.identity.dimensions:
            raise ValueError(
                f"embedding length {len(self.embedding)} != identity.dimensions "
                f"{self.identity.dimensions}"
            )


@dataclass(frozen=True, slots=True)
class SummaryRequest:
    node_id: NodeId
    node_type: NodeType
    title: str | None
    own_content: str | None
    child_summaries: tuple[str, ...]
    max_input_tokens: int = 2_000
    max_output_tokens: int = 256


@dataclass(frozen=True, slots=True)
class SummaryResult:
    text: str
    model: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float | None = None


@dataclass(frozen=True, slots=True)
class NodeSummary:
    node_id: NodeId
    text: str
    content_hash: str
    generator: str
    generator_model: str
    version: str


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query: str
    top_k: int = 8
    document_ids: tuple[DocumentId, ...] = ()
    node_types: tuple[NodeType, ...] = ()
    representation_types: tuple[RepresentationType, ...] = ()
    min_score: float | None = None
    candidate_pool_size: int | None = None

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("SearchRequest.query must be non-empty")
        if self.top_k < 1:
            raise ValueError("SearchRequest.top_k must be >= 1")
        if self.min_score is not None and not (0.0 <= self.min_score <= 1.0):
            raise ValueError("SearchRequest.min_score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class SearchHit:
    uri: str
    title: str | None
    node_id: NodeId
    document_id: DocumentId
    node_type: NodeType
    representation_type: RepresentationType
    score: float
    similarity: float
    preview: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchTiming:
    embedding_ms: float = 0.0
    vector_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class SearchUsage:
    embedding_calls: int = 0
    vector_searches: int = 0
    candidates_inspected: int = 0
    nodes_returned: int = 0
    rerank_calls: int = 0


@dataclass(frozen=True, slots=True)
class SearchResponse:
    query: str
    query_id: UUID
    candidates: tuple[SearchHit, ...]
    timing: SearchTiming
    usage: SearchUsage
    embedding_identity: EmbeddingIdentity
    reranked: bool = False
