"""API schemas for List / Grep / Read / Evidence endpoints."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from vikingrag.application.evidence import EvidenceQueryResult
from vikingrag.domain.models.assessment import EvidenceAssessment
from vikingrag.domain.models.evidence import EvidenceBundle
from vikingrag.domain.models.primitives import (
    GrepResponse,
    ListResponse,
    ReadResponse,
)


class ListRequestBody(BaseModel):
    uri: str = Field(min_length=1)
    limit: int = Field(default=50, ge=1, le=200)
    cursor: str | None = None


class ListItemBody(BaseModel):
    node_id: UUID
    document_id: UUID
    uri: str
    parent_uri: str | None
    node_type: str
    title: str | None
    ordinal: int
    has_content: bool
    token_count: int | None = None


class ListResponseBody(BaseModel):
    uri: str
    parent_node_id: UUID | None
    document_id: UUID
    items: list[ListItemBody]
    next_cursor: str | None
    truncated: bool
    total_children: int

    @classmethod
    def from_domain(cls, response: ListResponse) -> ListResponseBody:
        return cls(
            uri=response.uri,
            parent_node_id=UUID(str(response.parent_node_id)) if response.parent_node_id else None,
            document_id=UUID(str(response.document_id)),
            items=[
                ListItemBody(
                    node_id=UUID(str(i.node_id)),
                    document_id=UUID(str(i.document_id)),
                    uri=i.uri,
                    parent_uri=i.parent_uri,
                    node_type=i.node_type.value,
                    title=i.title,
                    ordinal=i.ordinal,
                    has_content=i.has_content,
                    token_count=i.token_count,
                )
                for i in response.items
            ],
            next_cursor=response.next_cursor,
            truncated=response.truncated,
            total_children=response.total_children,
        )


class GrepRequestBody(BaseModel):
    uri: str = Field(min_length=1)
    pattern: str = Field(min_length=1)
    case_sensitive: bool = True
    max_matches: int = Field(default=20, ge=1, le=100)
    max_descendants: int = Field(default=500, ge=1, le=2000)
    cursor: str | None = None


class GrepMatchBody(BaseModel):
    uri: str
    document_id: UUID
    node_id: UUID
    node_type: str
    title: str | None
    excerpt: str
    start_offset: int
    end_offset: int
    offset_system: str
    content_hash: str


class GrepResponseBody(BaseModel):
    uri: str
    pattern: str
    case_sensitive: bool
    matches: list[GrepMatchBody]
    next_cursor: str | None
    truncated: bool
    nodes_inspected: int

    @classmethod
    def from_domain(cls, response: GrepResponse) -> GrepResponseBody:
        return cls(
            uri=response.uri,
            pattern=response.pattern,
            case_sensitive=response.case_sensitive,
            matches=[
                GrepMatchBody(
                    uri=m.uri,
                    document_id=UUID(str(m.document_id)),
                    node_id=UUID(str(m.node_id)),
                    node_type=m.node_type.value,
                    title=m.title,
                    excerpt=m.excerpt,
                    start_offset=m.start_offset,
                    end_offset=m.end_offset,
                    offset_system=m.offset_system.value,
                    content_hash=m.content_hash,
                )
                for m in response.matches
            ],
            next_cursor=response.next_cursor,
            truncated=response.truncated,
            nodes_inspected=response.nodes_inspected,
        )


class ReadRequestBody(BaseModel):
    uri: str = Field(min_length=1)
    start_offset: int = Field(default=0, ge=0)
    max_tokens: int = Field(default=512, ge=1, le=4000)
    expected_content_hash: str | None = None


class ReadResponseBody(BaseModel):
    uri: str
    document_id: UUID
    node_id: UUID
    node_type: str
    title: str | None
    content_hash: str
    has_direct_content: bool
    text: str
    start_offset: int
    end_offset: int
    offset_system: str
    token_count: int
    truncated: bool
    next_offset: int | None
    source_metadata: dict[str, Any]

    @classmethod
    def from_domain(cls, response: ReadResponse) -> ReadResponseBody:
        return cls(
            uri=response.uri,
            document_id=UUID(str(response.document_id)),
            node_id=UUID(str(response.node_id)),
            node_type=response.node_type.value,
            title=response.title,
            content_hash=response.content_hash,
            has_direct_content=response.has_direct_content,
            text=response.text,
            start_offset=response.start_offset,
            end_offset=response.end_offset,
            offset_system=response.offset_system.value,
            token_count=response.token_count,
            truncated=response.truncated,
            next_offset=response.next_offset,
            source_metadata=dict(response.source_metadata),
        )


class EvidenceRequestBody(BaseModel):
    query: str = Field(min_length=1)
    document_ids: list[UUID] = Field(default_factory=list)
    search_top_k: int = Field(default=8, ge=1, le=50)
    max_candidates: int = Field(default=8, ge=1, le=30)
    max_list_children: int = Field(default=5, ge=1, le=50)
    max_descent_depth: int = Field(default=1, ge=0, le=3)
    read_max_tokens: int = Field(default=400, ge=1, le=2000)
    bundle_max_tokens: int = Field(default=4000, ge=1, le=20_000)
    include_grep: bool = False
    grep_pattern: str | None = None


class EvidenceItemBody(BaseModel):
    evidence_id: str
    uri: str
    document_id: UUID | None
    node_id: UUID | None
    content_hash: str
    text: str
    start_offset: int
    end_offset: int
    offset_system: str
    token_count: int
    provenance: list[str]
    discovery_score: float | None = None


class AssessmentBody(BaseModel):
    status: str
    coverage: float
    required_aspects: list[dict[str, str]]
    aspect_support: list[dict[str, Any]]
    missing_aspects: list[str]
    unsupported_aspects: list[str]
    conflicts: list[str]
    reason_codes: list[str]
    assessor_model: str
    policy_version: str
    uncalibrated_model_confidence: float | None = None

    @classmethod
    def from_domain(cls, assessment: EvidenceAssessment) -> AssessmentBody:
        return cls(
            status=assessment.status.value,
            coverage=assessment.coverage,
            required_aspects=[
                {"aspect_id": a.aspect_id, "text": a.text} for a in assessment.required_aspects
            ],
            aspect_support=[
                {
                    "aspect_id": a.aspect_id,
                    "status": a.status.value,
                    "references": [
                        {
                            "evidence_id": r.evidence_id,
                            "quote": r.quote,
                            "explanation": r.explanation,
                        }
                        for r in a.references
                    ],
                }
                for a in assessment.aspect_support
            ],
            missing_aspects=list(assessment.missing_aspects),
            unsupported_aspects=list(assessment.unsupported_aspects),
            conflicts=list(assessment.conflicts),
            reason_codes=list(assessment.reason_codes),
            assessor_model=assessment.assessor_model,
            policy_version=assessment.policy_version,
            uncalibrated_model_confidence=assessment.uncalibrated_model_confidence,
        )


class EvidenceResponseBody(BaseModel):
    query: str
    query_id: UUID
    trace_id: str
    evidence: list[EvidenceItemBody]
    total_tokens: int
    truncated: bool
    complete: bool
    excluded_count: int
    search_uris: list[str]
    truncation_reasons: list[str]
    assessment: AssessmentBody
    usage: dict[str, Any]
    timings_ms: dict[str, float]

    @classmethod
    def from_result(cls, result: EvidenceQueryResult) -> EvidenceResponseBody:
        bundle: EvidenceBundle = result.collection.bundle
        return cls(
            query=result.query,
            query_id=result.query_id,
            trace_id=result.trace_id,
            evidence=[
                EvidenceItemBody(
                    evidence_id=i.evidence_id,
                    uri=i.uri,
                    document_id=UUID(str(i.document_id)) if i.document_id else None,
                    node_id=UUID(str(i.node_id)) if i.node_id else None,
                    content_hash=i.content_hash,
                    text=i.text,
                    start_offset=i.start_offset,
                    end_offset=i.end_offset,
                    offset_system=i.offset_system.value,
                    token_count=i.token_count,
                    provenance=list(i.provenance),
                    discovery_score=i.discovery_score,
                )
                for i in bundle.items
            ],
            total_tokens=bundle.total_tokens,
            truncated=bundle.truncated or bool(result.collection.truncation_reasons),
            complete=result.collection.complete,
            excluded_count=len(bundle.excluded),
            search_uris=list(result.collection.search_uris),
            truncation_reasons=list(result.collection.truncation_reasons),
            assessment=AssessmentBody.from_domain(result.assessment),
            usage=dict(result.usage),
            timings_ms=dict(result.timings_ms),
        )
