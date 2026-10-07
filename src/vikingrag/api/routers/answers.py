"""HTTP endpoints for answers and query execution modes."""

from __future__ import annotations

from importlib import import_module
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Request

from vikingrag.api.auth import AuthContext, require_auth
from vikingrag.api.schemas.answers import (
    AnswerRequestBody,
    AnswerResponseBody,
    QueryRequestBody,
)
from vikingrag.application.answer.generate import AnswerGenerator
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.models.answer import AnswerRequest, ExecutionMode
from vikingrag.domain.models.document import DocumentId
from vikingrag.providers.factory import (
    build_embedding_provider,
    build_llm_provider,
    build_reranker,
)

router = APIRouter(
    prefix="/v1",
    tags=["answers"],
    dependencies=[Depends(require_auth)],
)


def _embedding_provider(request: Request) -> Any:
    owned = getattr(request.app.state, "embedding_provider", None)
    if owned is not None:
        return owned
    return build_embedding_provider(request.app.state.settings)


def _llm_provider(request: Request) -> Any:
    owned = getattr(request.app.state, "llm_provider", None)
    if owned is not None:
        return owned
    return build_llm_provider(request.app.state.settings)


def _search_service(request: Request) -> SemanticSearchService:
    settings = request.app.state.settings
    reranker = build_reranker(settings) if settings.retrieval.rerank_enabled else None
    return SemanticSearchService(
        database=request.app.state.database,
        embedding_provider=_embedding_provider(request),
        retrieval_settings=settings.retrieval,
        embedding_settings=settings.embedding,
        reranker=reranker,
    )


def _maybe_search_plus(request: Request) -> Any | None:
    """Prefer Search+ hook when application/search_plus exists."""
    try:
        mod = import_module("vikingrag.application.search_plus")
    except ModuleNotFoundError:
        return None
    factory = getattr(mod, "build_search_plus", None) or getattr(mod, "SearchPlusService", None)
    if factory is None:
        return None
    try:
        if callable(factory) and getattr(factory, "__name__", "") == "build_search_plus":
            return factory(request.app.state.settings, request.app.state.database)
        return factory(
            database=request.app.state.database,
            settings=request.app.state.settings,
        )
    except (TypeError, Exception):
        return None


def _answer_generator(request: Request) -> AnswerGenerator:
    settings = request.app.state.settings
    return AnswerGenerator(
        llm=_llm_provider(request),
        search=_search_service(request),
        list_service=ListService(database=request.app.state.database),
        grep_service=GrepService(database=request.app.state.database),
        read_service=ReadService(database=request.app.state.database),
        settings=settings,
        search_plus=_maybe_search_plus(request),
        database=request.app.state.database,
    )


def _to_domain(body: AnswerRequestBody) -> AnswerRequest:
    return AnswerRequest(
        question=body.question,
        document_ids=tuple(DocumentId(x) for x in body.document_ids),
        execution_mode=ExecutionMode(body.execution_mode),
        instructions=body.instructions,
        max_rounds=body.max_rounds,
        query_id=uuid4(),
    )


@router.post("/answers", response_model=AnswerResponseBody)
async def create_answer(
    request: Request,
    body: AnswerRequestBody,
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> AnswerResponseBody:
    result = await _answer_generator(request).generate(
        _to_domain(body),
        permitted_document_ids=auth.permitted_document_ids,
    )
    return AnswerResponseBody.from_domain(result)


@router.post("/query", response_model=AnswerResponseBody)
async def query(
    request: Request,
    body: QueryRequestBody,
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> AnswerResponseBody:
    """Execute vikingrag | vikingrag_e | vikingrag_e_plus."""
    result = await _answer_generator(request).generate(
        _to_domain(body),
        permitted_document_ids=auth.permitted_document_ids,
    )
    return AnswerResponseBody.from_domain(result)
