"""Evidence sufficiency assessment contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import UUID


class AssessmentStatus(StrEnum):
    SUFFICIENT = "sufficient"
    INSUFFICIENT = "insufficient"
    UNKNOWN = "unknown"


class AspectSupportStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class QueryAspect:
    aspect_id: str
    text: str


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    evidence_id: str
    quote: str | None = None
    explanation: str | None = None


@dataclass(frozen=True, slots=True)
class AspectSupport:
    aspect_id: str
    status: AspectSupportStatus
    references: tuple[EvidenceReference, ...] = ()
    note: str | None = None


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    status: AssessmentStatus
    required_aspects: tuple[QueryAspect, ...]
    aspect_support: tuple[AspectSupport, ...]
    coverage: float
    missing_aspects: tuple[str, ...]
    unsupported_aspects: tuple[str, ...]
    conflicts: tuple[str, ...]
    reason_codes: tuple[str, ...]
    assessor_model: str
    policy_version: str
    uncalibrated_model_confidence: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    timing_ms: float = 0.0
    query_id: UUID | None = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.coverage <= 1.0):
            raise ValueError("coverage must be in [0, 1]")
        if self.uncalibrated_model_confidence is not None and not (
            0.0 <= self.uncalibrated_model_confidence <= 1.0
        ):
            raise ValueError("uncalibrated_model_confidence must be in [0, 1]")
