"""Answer / query execution domain contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID, uuid4

from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.evidence import RetrievedEvidence
from vikingrag.domain.models.experience import ExperienceSnapshotId, LearningPolicy


class AnswerStatus(StrEnum):
    ANSWERED = "answered"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    PARTIAL = "partial"
    BUDGET_EXHAUSTED = "budget_exhausted"
    PROVIDER_ERROR = "provider_error"


class ExecutionMode(StrEnum):
    VIKINGRAG = "vikingrag"
    VIKINGRAG_E = "vikingrag_e"
    VIKINGRAG_E_PLUS = "vikingrag_e_plus"


@dataclass(frozen=True, slots=True)
class AnswerCitation:
    evidence_id: str
    uri: str
    quote: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


@dataclass(frozen=True, slots=True)
class AnswerRequest:
    question: str
    document_ids: tuple[DocumentId, ...] = ()
    execution_mode: ExecutionMode = ExecutionMode.VIKINGRAG
    learning_policy: LearningPolicy = LearningPolicy.LEARN
    snapshot_id: ExperienceSnapshotId | None = None
    instructions: str | None = None
    max_rounds: int | None = None
    query_id: UUID = field(default_factory=uuid4)

    def __post_init__(self) -> None:
        if not self.question.strip():
            raise ValueError("AnswerRequest.question must be non-empty")


@dataclass(frozen=True, slots=True)
class AnswerResponse:
    query_id: UUID
    status: AnswerStatus
    answer: str | None
    citations: tuple[AnswerCitation, ...] = ()
    evidence: tuple[RetrievedEvidence, ...] = ()
    execution_mode: ExecutionMode = ExecutionMode.VIKINGRAG
    route: str = "agentic"
    rounds_used: int = 0
    abstain_reason: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    trace_events: tuple[dict[str, Any], ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
