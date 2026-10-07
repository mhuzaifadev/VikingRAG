"""Evidence domain contracts for future verification / citation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vikingrag.domain.models.document import ChunkId, DocumentId, NodeId


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """Authoritative content loaded for grounding - retains provenance."""

    uri: str
    text: str
    token_count: int
    document_id: DocumentId | None = None
    node_id: NodeId | None = None
    chunk_id: ChunkId | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("token_count must be >= 0")


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Collection of evidence considered for a candidate answer."""

    items: tuple[RetrievedEvidence, ...] = ()
    total_tokens: int = 0

    def __post_init__(self) -> None:
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")
        computed = sum(item.token_count for item in self.items)
        if self.total_tokens == 0 and self.items:
            object.__setattr__(self, "total_tokens", computed)

    @classmethod
    def from_items(
        cls, items: list[RetrievedEvidence] | tuple[RetrievedEvidence, ...]
    ) -> EvidenceBundle:
        seq = tuple(items)
        return cls(items=seq, total_tokens=sum(i.token_count for i in seq))
