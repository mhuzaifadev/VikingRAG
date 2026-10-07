"""HTTP endpoints for List / Grep / Read / Evidence."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request

from vikingrag.api.auth import AuthContext, require_auth
from vikingrag.api.schemas.retrieval import (
    EvidenceRequestBody,
    EvidenceResponseBody,
    GrepRequestBody,
    GrepResponseBody,
    ListRequestBody,
    ListResponseBody,
    ReadRequestBody,
    ReadResponseBody,
)
from vikingrag.application.assessment import (
    EmptyBundleAssessor,
    LLMEvidenceAssessor,
    ScriptedEvidenceAssessor,
)
from vikingrag.application.budget import RetrievalContext
from vikingrag.application.evidence import EvidenceRetrievalService
from vikingrag.application.evidence_collector import EvidenceCollectionSettings, EvidenceCollector
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.primitives import GrepRequest, ListRequest, ReadRequest
from vikingrag.providers.factory import (
    build_embedding_provider,
    build_llm_provider,
    build_reranker,
)
from vikingrag.providers.llm.fake import FakeLLMProvider

router = APIRouter(
    prefix="/v1/retrieval",
    tags=["retrieval"],
    dependencies=[Depends(require_auth)],
)


def _list_service(request: Request) -> ListService:
    return ListService(database=request.app.state.database)


def _grep_service(request: Request) -> GrepService:
    return GrepService(database=request.app.state.database)


def _read_service(request: Request) -> ReadService:
    return ReadService(database=request.app.state.database)


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


def _assessor(request: Request):  # type: ignore[no-untyped-def]
    settings = request.app.state.settings
    mode = settings.retrieval.assessor_provider.lower().strip()
    if mode in {"empty", "none"}:
        return EmptyBundleAssessor()
    if mode in {"scripted", "fake", "test"}:
        return ScriptedEvidenceAssessor()
    if mode in {"unimplemented", ""}:
        raise NotImplementedCapabilityError("evidence_assessor")
    llm = _llm_provider(request)
    if isinstance(llm, FakeLLMProvider):
        # Fake LLM is not a semantic assessor - use scripted for explicit offline mode
        return ScriptedEvidenceAssessor(model="scripted-from-fake-llm")
    return LLMEvidenceAssessor(
        llm,
        model=settings.llm.model,
        temperature=0.0,
        min_coverage=settings.retrieval.min_assessment_coverage,
    )


def _evidence_service(request: Request) -> EvidenceRetrievalService:
    collector = EvidenceCollector(
        search=_search_service(request),
        list_service=_list_service(request),
        read_service=_read_service(request),
        grep_service=_grep_service(request),
    )
    return EvidenceRetrievalService(collector=collector, assessor=_assessor(request))


def _ctx(auth: AuthContext) -> RetrievalContext:
    return RetrievalContext.create(permitted_document_ids=auth.permitted_document_ids)


@router.post("/list", response_model=ListResponseBody)
async def list_nodes(
    request: Request,
    body: ListRequestBody,
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> ListResponseBody:
    settings = request.app.state.settings
    limit = min(body.limit, settings.retrieval.max_list_limit)
    response = await _list_service(request).list(
        ListRequest(uri=body.uri, limit=limit, cursor=body.cursor),
        ctx=_ctx(auth),
    )
    return ListResponseBody.from_domain(response)


@router.post("/grep", response_model=GrepResponseBody)
async def grep_nodes(
    request: Request,
    body: GrepRequestBody,
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> GrepResponseBody:
    settings = request.app.state.settings
    response = await _grep_service(request).grep(
        GrepRequest(
            uri=body.uri,
            pattern=body.pattern,
            case_sensitive=body.case_sensitive,
            max_matches=min(body.max_matches, settings.retrieval.max_grep_matches),
            max_descendants=min(body.max_descendants, settings.retrieval.max_grep_descendants),
            cursor=body.cursor,
        ),
        ctx=_ctx(auth),
    )
    return GrepResponseBody.from_domain(response)


@router.post("/read", response_model=ReadResponseBody)
async def read_node(
    request: Request,
    body: ReadRequestBody,
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> ReadResponseBody:
    settings = request.app.state.settings
    response = await _read_service(request).read(
        ReadRequest(
            uri=body.uri,
            start_offset=body.start_offset,
            max_tokens=min(body.max_tokens, settings.retrieval.max_read_tokens_per_call),
            expected_content_hash=body.expected_content_hash,
        ),
        ctx=_ctx(auth),
    )
    return ReadResponseBody.from_domain(response)


@router.post("/evidence", response_model=EvidenceResponseBody)
async def retrieve_evidence(
    request: Request,
    body: EvidenceRequestBody,
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> EvidenceResponseBody:
    settings = request.app.state.settings
    cfg = EvidenceCollectionSettings(
        search_top_k=min(body.search_top_k, settings.retrieval.initial_top_k),
        search_pool=settings.retrieval.candidate_pool_size,
        max_candidates=min(body.max_candidates, settings.retrieval.max_evidence_candidates),
        max_list_children=min(body.max_list_children, settings.retrieval.max_list_limit),
        max_descent_depth=min(body.max_descent_depth, settings.retrieval.max_descent_depth),
        read_max_tokens=min(body.read_max_tokens, settings.retrieval.max_read_tokens_per_call),
        bundle_max_tokens=min(body.bundle_max_tokens, settings.retrieval.max_bundle_tokens),
        include_grep=body.include_grep,
        grep_pattern=body.grep_pattern,
    )
    result = await _evidence_service(request).retrieve_evidence(
        body.query,
        document_ids=tuple(DocumentId(x) for x in body.document_ids),
        settings=cfg,
        permitted_document_ids=auth.permitted_document_ids,
    )
    return EvidenceResponseBody.from_result(result)
