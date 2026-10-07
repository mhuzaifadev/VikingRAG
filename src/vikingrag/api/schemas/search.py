"""API schemas for semantic search and document indexing."""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from vikingrag.application.indexing import IndexResult, IndexStatus
from vikingrag.domain.models.document import NodeType
from vikingrag.domain.models.representation import RepresentationType, SearchResponse


class IndexDocumentRequest(BaseModel):
    force_summaries: bool = False
    force_embeddings: bool = False


class IndexMetricsResponse(BaseModel):
    summaries_generated: int
    summaries_reused: int
    embeddings_generated: int
    embeddings_reused: int
    summary_input_tokens: int
    summary_output_tokens: int
    embedding_tokens: int
    summary_latency_ms: float
    embedding_latency_ms: float
    total_latency_ms: float


class IndexDocumentResponse(BaseModel):
    document_id: UUID
    status: str
    stage: str
    force_summaries: bool
    force_embeddings: bool
    metrics: IndexMetricsResponse

    @classmethod
    def from_result(cls, result: IndexResult) -> IndexDocumentResponse:
        m = result.metrics
        return cls(
            document_id=UUID(str(result.document_id)),
            status=result.status.value,
            stage=result.stage.value,
            force_summaries=result.force_summaries,
            force_embeddings=result.force_embeddings,
            metrics=IndexMetricsResponse(
                summaries_generated=m.summaries_generated,
                summaries_reused=m.summaries_reused,
                embeddings_generated=m.embeddings_generated,
                embeddings_reused=m.embeddings_reused,
                summary_input_tokens=m.summary_input_tokens,
                summary_output_tokens=m.summary_output_tokens,
                embedding_tokens=m.embedding_tokens,
                summary_latency_ms=m.summary_latency_ms,
                embedding_latency_ms=m.embedding_latency_ms,
                total_latency_ms=m.total_latency_ms,
            ),
        )


class IndexStatusResponse(BaseModel):
    document_id: UUID
    status: str
    stage: str
    representation_count: int
    embedding_count: int
    metrics: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_status(cls, status: IndexStatus) -> IndexStatusResponse:
        return cls(
            document_id=UUID(str(status.document_id)),
            status=status.status.value,
            stage=status.stage.value,
            representation_count=status.representation_count,
            embedding_count=status.embedding_count,
            metrics=dict(status.metrics),
        )


class SearchRequestBody(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=100)
    document_ids: list[UUID] = Field(default_factory=list)
    node_types: list[NodeType] = Field(default_factory=list)
    representation_types: list[RepresentationType] = Field(default_factory=list)
    min_score: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_pool_size: int | None = Field(default=None, ge=1, le=200)


class SearchHitResponse(BaseModel):
    uri: str
    title: str | None
    node_id: UUID
    document_id: UUID
    node_type: str
    representation_type: str
    score: float
    similarity: float
    preview: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchTimingResponse(BaseModel):
    embedding_ms: float
    vector_ms: float
    rerank_ms: float
    total_ms: float


class SearchUsageResponse(BaseModel):
    embedding_calls: int
    vector_searches: int
    candidates_inspected: int
    nodes_returned: int
    rerank_calls: int


class SearchResponseBody(BaseModel):
    query: str
    query_id: UUID
    candidates: list[SearchHitResponse]
    timing: SearchTimingResponse
    usage: SearchUsageResponse
    embedding_provider: str
    embedding_model: str
    embedding_dimensions: int
    reranked: bool

    @classmethod
    def from_domain(cls, response: SearchResponse) -> SearchResponseBody:
        return cls(
            query=response.query,
            query_id=response.query_id,
            candidates=[
                SearchHitResponse(
                    uri=c.uri,
                    title=c.title,
                    node_id=UUID(str(c.node_id)),
                    document_id=UUID(str(c.document_id)),
                    node_type=c.node_type.value,
                    representation_type=c.representation_type.value,
                    score=c.score,
                    similarity=c.similarity,
                    preview=c.preview,
                    metadata=dict(c.metadata),
                )
                for c in response.candidates
            ],
            timing=SearchTimingResponse(
                embedding_ms=response.timing.embedding_ms,
                vector_ms=response.timing.vector_ms,
                rerank_ms=response.timing.rerank_ms,
                total_ms=response.timing.total_ms,
            ),
            usage=SearchUsageResponse(
                embedding_calls=response.usage.embedding_calls,
                vector_searches=response.usage.vector_searches,
                candidates_inspected=response.usage.candidates_inspected,
                nodes_returned=response.usage.nodes_returned,
                rerank_calls=response.usage.rerank_calls,
            ),
            embedding_provider=response.embedding_identity.provider,
            embedding_model=response.embedding_identity.model,
            embedding_dimensions=response.embedding_identity.dimensions,
            reranked=response.reranked,
        )
