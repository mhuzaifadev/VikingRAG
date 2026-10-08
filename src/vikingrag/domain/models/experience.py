"""Experience-edge domain contracts (Algorithms 2-3) - no ORM dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, NewType
from uuid import UUID, uuid4

from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.representation import EmbeddingIdentity

QueryRunId = NewType("QueryRunId", UUID)
RetrievalEventId = NewType("RetrievalEventId", UUID)
ExperiencePayloadId = NewType("ExperiencePayloadId", UUID)
ExperienceEdgeId = NewType("ExperienceEdgeId", UUID)
ExperienceSnapshotId = NewType("ExperienceSnapshotId", UUID)


class LearningPolicy(StrEnum):
    """Controls whether a completed answer may mutate experience edges.

    - off: do not persist a learning job
    - record_only: persist query_run + events; never build edges
    - learn: enqueue PENDING edge-build jobs (default production path)
    - frozen: refuse enqueue; workers must not activate new edges for this run
    """

    OFF = "off"
    RECORD_ONLY = "record_only"
    LEARN = "learn"
    FROZEN = "frozen"


class SnapshotStatus(StrEnum):
    ACTIVE = "active"
    FROZEN = "frozen"
    ARCHIVED = "archived"


class QueryRunStatus(StrEnum):
    """Lifecycle of a query orchestration run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ABSTAINED = "abstained"


class QueryRunRoute(StrEnum):
    ONE_ROUND = "one_round"
    ESCALATED = "escalated"
    FAILED = "failed"
    ABSTAINED = "abstained"


class EdgeBuildStatus(StrEnum):
    """Durable Algorithm-2 job state on a query run (not fire-and-forget)."""

    NONE = "none"
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RetrievalEventType(StrEnum):
    SEARCH = "search"
    LIST = "list"
    GREP = "grep"
    READ = "read"
    EDGE_EXPAND = "edge_expand"
    JUDGE = "judge"
    ANSWER = "answer"


class ExperienceEdgeStatus(StrEnum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    INVALIDATED = "invalidated"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ExperienceExpansionLimits:
    """Hard caps for Algorithm-3 conditioned multi-hop expansion."""

    gamma: float = 0.8
    max_hops: int = 2
    max_nodes: int = 32
    max_edges: int = 64
    max_tokens: int = 4_000
    max_wall_time_ms: int = 5_000

    def __post_init__(self) -> None:
        if not (0.0 <= self.gamma <= 1.0):
            raise ValueError("gamma must be in [0, 1]")
        for name in ("max_hops", "max_nodes", "max_edges", "max_tokens", "max_wall_time_ms"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


@dataclass(slots=True)
class ExperienceSnapshot:
    """Named, versioned set of experience edges for warm-up / held-out isolation."""

    id: ExperienceSnapshotId
    name: str
    status: SnapshotStatus = SnapshotStatus.ACTIVE
    corpus_id: UUID | None = None
    edge_ids: tuple[ExperienceEdgeId, ...] = ()
    payload_ids: tuple[ExperiencePayloadId, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("ExperienceSnapshot.name must be non-empty")


@dataclass(slots=True)
class QueryRun:
    id: QueryRunId
    query_text: str
    status: QueryRunStatus = QueryRunStatus.PENDING
    route: QueryRunRoute | None = None
    answer: str | None = None
    sufficiency_score: float | None = None
    query_embedding: list[float] | None = None
    embedding_identity: EmbeddingIdentity | None = None
    edge_build_status: EdgeBuildStatus = EdgeBuildStatus.NONE
    edge_build_error: str | None = None
    learning_policy: LearningPolicy = LearningPolicy.LEARN
    snapshot_id: ExperienceSnapshotId | None = None
    tenant_id: UUID | None = None
    corpus_id: UUID | None = None
    document_ids: tuple[DocumentId, ...] = ()
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    retrieval_tokens: int = 0
    llm_calls: int = 0
    retrieval_rounds: int = 0
    latency_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.query_text.strip():
            raise ValueError("QueryRun.query_text must be non-empty")
        if (
            self.query_embedding is not None
            and self.embedding_identity is not None
            and len(self.query_embedding) != self.embedding_identity.dimensions
        ):
            raise ValueError(
                f"query_embedding length {len(self.query_embedding)} != "
                f"identity.dimensions {self.embedding_identity.dimensions}"
            )

    @property
    def is_learnable(self) -> bool:
        """Algorithm 2 must never learn from failed or abstained runs."""
        return self.status is QueryRunStatus.SUCCEEDED and self.route not in {
            QueryRunRoute.FAILED,
            QueryRunRoute.ABSTAINED,
        }


@dataclass(slots=True)
class RetrievalEvent:
    id: RetrievalEventId
    query_run_id: QueryRunId
    round_no: int
    event_type: RetrievalEventType
    arguments: dict[str, Any] = field(default_factory=dict)
    result_refs: list[str] = field(default_factory=list)
    latency_ms: int | None = None
    token_cost: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.round_no < 0:
            raise ValueError("round_no must be >= 0")


@dataclass(slots=True)
class ExperiencePayload:
    """Historical query context attached to one or more directed edges."""

    id: ExperiencePayloadId
    query_run_id: QueryRunId
    query_text: str
    query_embedding: list[float]
    embedding_identity: EmbeddingIdentity
    trace_summary: str
    support_uris: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.query_text.strip():
            raise ValueError("ExperiencePayload.query_text must be non-empty")
        if len(self.query_embedding) != self.embedding_identity.dimensions:
            raise ValueError(
                f"query_embedding length {len(self.query_embedding)} != "
                f"identity.dimensions {self.embedding_identity.dimensions}"
            )


@dataclass(slots=True)
class ExperienceEdge:
    """Directed query-conditioned shortcut: source landing → later evidence."""

    id: ExperienceEdgeId
    payload_id: ExperiencePayloadId
    source_uri: str
    target_uri: str
    status: ExperienceEdgeStatus = ExperienceEdgeStatus.ACTIVE
    source_node_id: NodeId | None = None
    target_node_id: NodeId | None = None
    source_revision: str | None = None
    target_revision: str | None = None
    support_score: float = 1.0
    success_count: int = 0
    failure_count: int = 0
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.source_uri.strip() or not self.target_uri.strip():
            raise ValueError("source_uri and target_uri must be non-empty")
        if self.source_uri == self.target_uri:
            raise ValueError("self-edges are forbidden")


@dataclass(frozen=True, slots=True)
class ActivatedEdge:
    """An experience edge that passed gamma-gate and identity checks at query time."""

    edge: ExperienceEdge
    payload: ExperiencePayload
    query_similarity: float
    hop: int


@dataclass(frozen=True, slots=True)
class ExpansionResult:
    seed_uris: tuple[str, ...]
    expanded_uris: tuple[str, ...]
    activated: tuple[ActivatedEdge, ...]
    hops_taken: int
    edges_considered: int
    edges_activated: int
    truncated: bool = False
    truncation_reason: str | None = None

    @property
    def all_uris(self) -> tuple[str, ...]:
        seen: set[str] = set()
        ordered: list[str] = []
        for uri in (*self.seed_uris, *self.expanded_uris):
            if uri not in seen:
                seen.add(uri)
                ordered.append(uri)
        return tuple(ordered)


def new_query_run_id() -> QueryRunId:
    return QueryRunId(uuid4())


def new_retrieval_event_id() -> RetrievalEventId:
    return RetrievalEventId(uuid4())


def new_experience_payload_id() -> ExperiencePayloadId:
    return ExperiencePayloadId(uuid4())


def new_experience_edge_id() -> ExperienceEdgeId:
    return ExperienceEdgeId(uuid4())


def new_experience_snapshot_id() -> ExperienceSnapshotId:
    return ExperienceSnapshotId(uuid4())
