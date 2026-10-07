"""Retrieval domain contracts - no vendor or ORM dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

from vikingrag.domain.models.document import ChunkId, DocumentId, NodeId


@dataclass(frozen=True, slots=True)
class RetrievalQuery:
    """Inbound retrieval request (tenant/corpus wiring arrives in later phases)."""

    text: str
    top_k: int = 8
    corpus_id: UUID | None = None
    query_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("RetrievalQuery.text must be non-empty")
        if self.top_k < 1:
            raise ValueError("RetrievalQuery.top_k must be >= 1")


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """Compact discovery hit before authoritative Read."""

    uri: str
    score: float
    preview: str
    object_type: str
    document_id: DocumentId | None = None
    node_id: NodeId | None = None
    chunk_id: ChunkId | None = None
    parent_uri: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalBudget:
    """Hard limits for a single query run."""

    max_rounds: int = 6
    max_tool_calls: int = 12
    max_read_tokens: int = 12_000
    max_wall_time_ms: int = 12_000
    max_embedding_calls: int = 20
    max_vector_searches: int = 20
    max_estimated_cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_rounds",
            "max_tool_calls",
            "max_read_tokens",
            "max_wall_time_ms",
            "max_embedding_calls",
            "max_vector_searches",
        ):
            if getattr(self, name) < 1:
                raise ValueError(f"RetrievalBudget.{name} must be >= 1")


@dataclass(slots=True)
class RetrievalTrace:
    """Append-only style record of retrieval activity for a query."""

    query_id: UUID
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    route: str | None = None

    def record(self, event_type: str, **payload: Any) -> None:
        self.events.append(
            {
                "event_type": event_type,
                "at": datetime.now(UTC).isoformat(),
                **payload,
            }
        )
