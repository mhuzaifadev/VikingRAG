"""Evidence domain contracts - authoritative excerpts with provenance."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from vikingrag.domain.models.document import ChunkId, DocumentId, NodeId
from vikingrag.domain.models.primitives import OffsetSystem


class ExclusionReason(StrEnum):
    EMPTY = "empty"
    DUPLICATE_RANGE = "duplicate_range"
    OVERLAP_COLLAPSED = "overlap_collapsed"
    BUNDLE_TOKEN_LIMIT = "bundle_token_limit"
    MIXED_REVISION = "mixed_revision"
    SCOPE_DENIED = "scope_denied"
    NO_DIRECT_CONTENT = "no_direct_content"
    READ_FAILED = "read_failed"
    COLLECTION_LIMIT = "collection_limit"


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    """Authoritative content loaded for grounding - retains provenance.

    ``score`` is discovery ranking only - never evidence confidence.
    """

    uri: str
    text: str
    token_count: int
    document_id: DocumentId | None = None
    node_id: NodeId | None = None
    chunk_id: ChunkId | None = None
    evidence_id: str = ""
    content_hash: str = ""
    start_offset: int = 0
    end_offset: int = 0
    offset_system: OffsetSystem = OffsetSystem.UNICODE_CODE_POINT
    discovery_score: float | None = None
    score: float | None = None  # alias for discovery_score (compat)
    provenance: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_count < 0:
            raise ValueError("token_count must be >= 0")
        if self.start_offset < 0 or self.end_offset < self.start_offset:
            raise ValueError("invalid evidence offsets")


@dataclass(frozen=True, slots=True)
class ExcludedEvidence:
    uri: str | None
    reason: ExclusionReason
    detail: str = ""


@dataclass(frozen=True, slots=True)
class EvidenceBundle:
    """Collection of authoritative evidence for a query."""

    items: tuple[RetrievedEvidence, ...] = ()
    total_tokens: int = 0
    excluded: tuple[ExcludedEvidence, ...] = ()
    truncated: bool = False
    complete: bool = True

    def __post_init__(self) -> None:
        if self.total_tokens < 0:
            raise ValueError("total_tokens must be >= 0")
        computed = sum(item.token_count for item in self.items)
        if self.total_tokens != computed:
            raise ValueError(
                f"EvidenceBundle.total_tokens ({self.total_tokens}) must equal "
                f"sum of item token_count ({computed})"
            )

    @classmethod
    def from_items(
        cls,
        items: list[RetrievedEvidence] | tuple[RetrievedEvidence, ...],
        *,
        excluded: list[ExcludedEvidence] | tuple[ExcludedEvidence, ...] = (),
        truncated: bool = False,
        complete: bool = True,
    ) -> EvidenceBundle:
        seq = tuple(items)
        return cls(
            items=seq,
            total_tokens=sum(i.token_count for i in seq),
            excluded=tuple(excluded),
            truncated=truncated,
            complete=complete,
        )

    def by_id(self) -> dict[str, RetrievedEvidence]:
        return {item.evidence_id: item for item in self.items if item.evidence_id}
