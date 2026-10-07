"""HTTP schemas for answers / query endpoints."""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, Field

from vikingrag.domain.models.answer import AnswerResponse, AnswerStatus, ExecutionMode


class AnswerRequestBody(BaseModel):
    question: str = Field(min_length=1)
    document_ids: list[UUID] = Field(default_factory=list)
    instructions: str | None = None
    max_rounds: int | None = Field(default=None, ge=1, le=50)
    execution_mode: Literal["vikingrag", "vikingrag_e", "vikingrag_e_plus"] = "vikingrag"


class QueryRequestBody(AnswerRequestBody):
    """Alias body for POST /v1/query (execution_mode required for E/E+ routing)."""

    execution_mode: Literal["vikingrag", "vikingrag_e", "vikingrag_e_plus"] = "vikingrag"


class CitationBody(BaseModel):
    evidence_id: str
    uri: str
    quote: str | None = None
    start_offset: int | None = None
    end_offset: int | None = None


class EvidenceItemBody(BaseModel):
    evidence_id: str
    uri: str
    text: str
    token_count: int
    start_offset: int = 0
    end_offset: int = 0
    content_hash: str = ""


class AnswerResponseBody(BaseModel):
    query_id: UUID
    status: AnswerStatus
    answer: str | None
    citations: list[CitationBody]
    evidence: list[EvidenceItemBody]
    execution_mode: ExecutionMode
    route: str
    rounds_used: int
    abstain_reason: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_domain(cls, response: AnswerResponse) -> AnswerResponseBody:
        return cls(
            query_id=response.query_id,
            status=response.status,
            answer=response.answer,
            citations=[
                CitationBody(
                    evidence_id=c.evidence_id,
                    uri=c.uri,
                    quote=c.quote,
                    start_offset=c.start_offset,
                    end_offset=c.end_offset,
                )
                for c in response.citations
            ],
            evidence=[
                EvidenceItemBody(
                    evidence_id=e.evidence_id,
                    uri=e.uri,
                    text=e.text,
                    token_count=e.token_count,
                    start_offset=e.start_offset,
                    end_offset=e.end_offset,
                    content_hash=e.content_hash,
                )
                for e in response.evidence
            ],
            execution_mode=response.execution_mode,
            route=response.route,
            rounds_used=response.rounds_used,
            abstain_reason=response.abstain_reason,
            usage=response.usage,
            metadata=response.metadata,
        )
