"""Semantic search and document indexing HTTP endpoints."""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Request

from vikingrag.api.schemas.search import (
    IndexDocumentRequest,
    IndexDocumentResponse,
    IndexStatusResponse,
    SearchRequestBody,
    SearchResponseBody,
)
from vikingrag.application.indexing import DocumentIndexingService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.representation import SearchRequest
from vikingrag.providers.factory import (
    build_embedding_provider,
    build_reranker,
    build_summary_generator,
)

router = APIRouter(prefix="/v1", tags=["retrieval"])


def _indexing_service(request: Request) -> DocumentIndexingService:
    settings = request.app.state.settings
    return DocumentIndexingService(
        database=request.app.state.database,
        summary_generator=build_summary_generator(settings),
        embedding_provider=build_embedding_provider(settings),
        indexing_settings=settings.indexing,
        embedding_settings=settings.embedding,
    )


def _search_service(request: Request) -> SemanticSearchService:
    settings = request.app.state.settings
    reranker = build_reranker(settings) if settings.retrieval.rerank_enabled else None
    return SemanticSearchService(
        database=request.app.state.database,
        embedding_provider=build_embedding_provider(settings),
        retrieval_settings=settings.retrieval,
        embedding_settings=settings.embedding,
        reranker=reranker,
    )


@router.post(
    "/documents/{document_id}/index",
    response_model=IndexDocumentResponse,
)
async def index_document(
    request: Request,
    document_id: UUID,
    body: IndexDocumentRequest | None = None,
) -> IndexDocumentResponse:
    payload = body or IndexDocumentRequest()
    service = _indexing_service(request)
    result = await service.index_document(
        DocumentId(document_id),
        force_summaries=payload.force_summaries,
        force_embeddings=payload.force_embeddings,
    )
    return IndexDocumentResponse.from_result(result)


@router.get(
    "/documents/{document_id}/index-status",
    response_model=IndexStatusResponse,
)
async def index_status(request: Request, document_id: UUID) -> IndexStatusResponse:
    service = _indexing_service(request)
    status = await service.get_status(DocumentId(document_id))
    return IndexStatusResponse.from_status(status)


@router.post("/search", response_model=SearchResponseBody)
async def search(request: Request, body: SearchRequestBody) -> SearchResponseBody:
    service = _search_service(request)
    domain_request = SearchRequest(
        query=body.query,
        top_k=body.top_k,
        document_ids=tuple(DocumentId(x) for x in body.document_ids),
        node_types=tuple(body.node_types),
        representation_types=tuple(body.representation_types),
        min_score=body.min_score,
        candidate_pool_size=body.candidate_pool_size,
    )
    response = await service.search(domain_request)
    return SearchResponseBody.from_domain(response)
