"""Answer generation use-case: agentic retrieval + citation validation."""

from __future__ import annotations

from typing import Any

from vikingrag.application.agent.loop import AgenticRetrievalExecutor, AgentRunResult
from vikingrag.application.agent.tool_executor import RetrievalToolExecutor
from vikingrag.application.assessment import EvidenceAssessor, build_assessor_from_settings
from vikingrag.application.budget import BudgetLimits, RetrievalContext
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.domain.models.answer import (
    AnswerCitation,
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
    ExecutionMode,
)
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.evidence import RetrievedEvidence
from vikingrag.domain.models.experience import QueryRunRoute
from vikingrag.observability.logging import get_logger
from vikingrag.providers.llm.base import LLMProvider
from vikingrag.settings.config import Settings

logger = get_logger(__name__)


def intersect_document_scope(
    permitted: frozenset[DocumentId] | None,
    requested: tuple[DocumentId, ...],
) -> frozenset[DocumentId] | None:
    """Auth allowlist ∩ request.document_ids.

    - No request filter → keep auth scope (None = unrestricted).
    - Request filter + unrestricted auth → frozenset(requested).
    - Both set → intersection (may be empty = deny-all).
    """
    if not requested:
        return permitted
    req = frozenset(requested)
    if permitted is None:
        return req
    return frozenset(d for d in req if d in permitted)


class AnswerGenerator:
    """Runs Algorithm 1 (and optional E/E+ via Search+) to produce AnswerResponse."""

    def __init__(
        self,
        *,
        llm: LLMProvider,
        search: SemanticSearchService,
        list_service: ListService,
        grep_service: GrepService,
        read_service: ReadService,
        settings: Settings,
        search_plus: Any | None = None,
        database: Any | None = None,
        assessor: EvidenceAssessor | None = None,
    ) -> None:
        self._llm = llm
        self._search = search
        self._list = list_service
        self._grep = grep_service
        self._read = read_service
        self._settings = settings
        self._search_plus = search_plus
        self._database = database
        self._assessor = (
            assessor if assessor is not None else build_assessor_from_settings(settings, llm)
        )

    def _tool_executor(
        self, *, use_search_plus: bool, document_ids: tuple[DocumentId, ...] = ()
    ) -> RetrievalToolExecutor:
        return RetrievalToolExecutor(
            search=self._search,
            list_service=self._list,
            grep=self._grep,
            read=self._read,
            search_plus=self._search_plus,
            use_search_plus=use_search_plus,
            document_ids=document_ids,
        )

    def _agent(
        self,
        *,
        max_rounds: int | None,
        use_search_plus: bool,
        document_ids: tuple[DocumentId, ...] = (),
    ) -> AgenticRetrievalExecutor:
        r = self._settings.retrieval
        return AgenticRetrievalExecutor(
            llm=self._llm,
            tool_executor=self._tool_executor(
                use_search_plus=use_search_plus, document_ids=document_ids
            ),
            max_rounds=max_rounds or r.max_rounds,
            finalization_llm_reserve=r.finalization_llm_reserve,
            default_top_k=r.initial_top_k,
            default_read_tokens=r.max_read_tokens_per_call,
            model=self._settings.llm.model or None,
        )

    def _budget_limits(self) -> BudgetLimits:
        r = self._settings.retrieval
        if self._settings.paper.enabled:
            return BudgetLimits.paper_profile()
        return BudgetLimits(
            max_tool_calls=r.max_tool_calls,
            max_read_tokens=r.max_read_tokens,
            max_wall_time_ms=r.max_wall_time_ms,
            max_llm_calls=max(4, r.max_rounds + r.finalization_llm_reserve + 2),
        )

    async def generate(
        self,
        request: AnswerRequest,
        *,
        permitted_document_ids: frozenset[DocumentId] | None = None,
        ctx: RetrievalContext | None = None,
    ) -> AnswerResponse:
        mode = request.execution_mode
        use_plus = mode in {ExecutionMode.VIKINGRAG_E, ExecutionMode.VIKINGRAG_E_PLUS}
        route = "agentic"
        escalation_reason: str | None = None

        if use_plus and self._search_plus is None:
            raise NotImplementedCapabilityError(
                f"search_plus (required for execution_mode={mode.value}; "
                "refusing silent fallback to ordinary Search)"
            )

        scoped = intersect_document_scope(permitted_document_ids, request.document_ids)
        context = ctx or RetrievalContext.create(
            limits=self._budget_limits(),
            permitted_document_ids=scoped,
            query_id=request.query_id,
        )
        if ctx is not None and request.document_ids:
            # Narrow an injected context the same way.
            context.permitted_document_ids = intersect_document_scope(
                context.permitted_document_ids, request.document_ids
            )
        context.require_scope_not_empty()

        # E+: one-round Search+ → strict sufficiency → single Alg1 fallback
        if mode is ExecutionMode.VIKINGRAG_E_PLUS and self._search_plus is not None:
            from vikingrag.application.agent.loop import AgenticRetrievalLoop
            from vikingrag.application.orchestration.query import QueryOrchestrator, QueryRequest

            agent_exec = self._agent(
                max_rounds=request.max_rounds,
                use_search_plus=True,
                document_ids=request.document_ids,
            )
            orch = QueryOrchestrator(
                search=self._search,
                search_plus=self._search_plus,
                assessor=self._assessor,
                agent=AgenticRetrievalLoop(agent_exec),
                read_service=self._read,
                llm=self._llm,
                llm_model=self._settings.llm.model or None,
            )
            orch_resp = await orch.run(
                QueryRequest(
                    query=request.question,
                    mode=ExecutionMode.VIKINGRAG_E_PLUS,
                    document_ids=request.document_ids,
                    top_k=self._settings.retrieval.initial_top_k,
                    instructions=request.instructions,
                ),
                ctx=context,
            )
            from vikingrag.domain.models.experience import QueryRunRoute as QR

            if orch_resp.route is QR.ONE_ROUND and not orch_resp.gaps:
                return await self._finalize(
                    request=request,
                    context=context,
                    status=AnswerStatus.ANSWERED,
                    answer=orch_resp.answer,
                    citations=list(orch_resp.citations),
                    evidence=list(orch_resp.evidence.items),
                    route="one_round_search_plus",
                    rounds_used=0,
                    usage=dict(context.usage.snapshot()),
                    events=list(orch_resp.trace_events),
                    metadata={
                        "cold_path": orch_resp.cold_path,
                        "e_plus": True,
                        **orch_resp.metadata,
                    },
                    experience_route=QueryRunRoute.ONE_ROUND,
                )

            # Escalation already ran Alg1 inside the orchestrator — do not re-run.
            route = "e_plus_escalated"
            escalation_reason = (
                "gaps:" + ",".join(orch_resp.gaps)
                if orch_resp.gaps
                else "strict_sufficiency_failed"
            )
            status = orch_resp.agent_status or AnswerStatus.ANSWERED
            if orch_resp.answer is None and status is AnswerStatus.ANSWERED:
                status = AnswerStatus.INSUFFICIENT_EVIDENCE
            return await self._finalize(
                request=request,
                context=context,
                status=status,
                answer=orch_resp.answer,
                citations=list(orch_resp.citations),
                evidence=list(orch_resp.evidence.items),
                route=route,
                rounds_used=int(orch_resp.metadata.get("rounds") or 0),
                usage=dict(orch_resp.usage or context.usage.snapshot()),
                events=list(orch_resp.trace_events),
                metadata={
                    "cold_path": orch_resp.cold_path,
                    "e_plus": True,
                    "escalation_reason": escalation_reason,
                    **orch_resp.metadata,
                },
                abstain_reason=escalation_reason,
                experience_route=QueryRunRoute.ESCALATED,
            )

        agent = self._agent(
            max_rounds=request.max_rounds,
            use_search_plus=use_plus,
            document_ids=request.document_ids,
        )
        result = await agent.run(
            request.question,
            ctx=context,
            instructions=request.instructions,
        )

        if mode is ExecutionMode.VIKINGRAG_E and use_plus:
            route = "agentic_search_plus"
        elif mode is ExecutionMode.VIKINGRAG:
            route = "agentic"

        return await self._finalize_from_agent(
            request=request,
            context=context,
            result=result,
            route=route,
            escalation_reason=escalation_reason,
            experience_route=QueryRunRoute.ESCALATED
            if "agentic" in route or "escalated" in route
            else QueryRunRoute.ONE_ROUND,
        )

    async def _finalize_from_agent(
        self,
        *,
        request: AnswerRequest,
        context: RetrievalContext,
        result: AgentRunResult,
        route: str,
        escalation_reason: str | None,
        experience_route: QueryRunRoute,
    ) -> AnswerResponse:
        logger.info(
            "answer_generated",
            query_id=str(result.query_id or request.query_id),
            status=result.status.value,
            mode=request.execution_mode.value,
            rounds=result.rounds_used,
        )
        return await self._finalize(
            request=request,
            context=context,
            status=result.status,
            answer=result.answer,
            citations=list(result.citations),
            evidence=list(result.evidence),
            route=route,
            rounds_used=result.rounds_used,
            usage=dict(result.usage),
            events=list(result.events),
            metadata={
                **result.metadata,
                "trace_id": result.trace_id,
                "escalation_reason": escalation_reason,
            },
            abstain_reason=escalation_reason or result.metadata.get("abstain_reason"),
            experience_route=experience_route,
            query_id_override=result.query_id,
        )

    async def _finalize(
        self,
        *,
        request: AnswerRequest,
        context: RetrievalContext,
        status: AnswerStatus,
        answer: str | None,
        citations: list[AnswerCitation],
        evidence: list[RetrievedEvidence],
        route: str,
        rounds_used: int,
        usage: dict[str, Any],
        events: list[dict[str, Any]],
        metadata: dict[str, Any],
        experience_route: QueryRunRoute,
        abstain_reason: str | None = None,
        query_id_override: Any = None,
    ) -> AnswerResponse:
        meta = dict(metadata)
        if self._database is not None and status is AnswerStatus.ANSWERED:
            try:
                from vikingrag.application.experience.enqueue import enqueue_experience_learning
                from vikingrag.domain.models.experience import QueryRunStatus

                identity = self._search.embedding_identity()
                cache_key = (request.question, identity.key())
                cached = context.get_cached_query_embedding(cache_key)
                run = await enqueue_experience_learning(
                    self._database,
                    query_text=request.question,
                    answer=answer,
                    status=QueryRunStatus.SUCCEEDED,
                    route=experience_route,
                    query_embedding=list(cached) if cached else None,
                    embedding_identity=identity if cached else None,
                    trace_events=list(events),
                    citation_uris=[c.uri for c in citations],
                    metadata={
                        "query_id": str(query_id_override or request.query_id),
                        "execution_mode": request.execution_mode.value,
                        "route": route,
                        "escalation_reason": meta.get("escalation_reason"),
                        "sufficiency": meta.get("assessment") or meta.get("sufficiency"),
                        "citation_uris": [c.uri for c in citations],
                        "rounds_used": rounds_used,
                        "usage": dict(usage),
                    },
                    learning_policy=request.learning_policy,
                    snapshot_id=request.snapshot_id,
                    document_ids=request.document_ids,
                )
                if run is not None:
                    meta["experience_query_run_id"] = str(run.id)
                    meta["edge_build_status"] = run.edge_build_status.value
                    meta["learning_policy"] = request.learning_policy.value
                    meta["replay"] = "recorded"
                    if request.snapshot_id is not None:
                        meta["snapshot_id"] = str(request.snapshot_id)
            except Exception as exc:
                logger.warning("experience_enqueue_failed", error=str(exc))

        return AnswerResponse(
            query_id=query_id_override or request.query_id,
            status=status,
            answer=answer,
            citations=tuple(citations),
            evidence=tuple(evidence),
            execution_mode=request.execution_mode,
            route=route,
            rounds_used=rounds_used,
            abstain_reason=abstain_reason,
            usage=usage,
            trace_events=tuple(events),
            metadata=meta,
        )
