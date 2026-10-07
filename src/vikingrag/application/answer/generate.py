"""Answer generation use-case: agentic retrieval + citation validation."""

from __future__ import annotations

from typing import Any

from vikingrag.application.agent.loop import AgenticRetrievalExecutor
from vikingrag.application.agent.tool_executor import RetrievalToolExecutor
from vikingrag.application.assessment import EvidenceAssessor, build_assessor_from_settings
from vikingrag.application.budget import BudgetLimits, RetrievalContext
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.models.answer import (
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
    ExecutionMode,
)
from vikingrag.domain.models.document import DocumentId
from vikingrag.observability.logging import get_logger
from vikingrag.providers.llm.base import LLMProvider
from vikingrag.settings.config import Settings

logger = get_logger(__name__)


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

    def _tool_executor(self, *, use_search_plus: bool) -> RetrievalToolExecutor:
        return RetrievalToolExecutor(
            search=self._search,
            list_service=self._list,
            grep=self._grep,
            read=self._read,
            search_plus=self._search_plus,
            use_search_plus=use_search_plus,
        )

    def _agent(self, *, max_rounds: int | None, use_search_plus: bool) -> AgenticRetrievalExecutor:
        r = self._settings.retrieval
        return AgenticRetrievalExecutor(
            llm=self._llm,
            tool_executor=self._tool_executor(use_search_plus=use_search_plus),
            max_rounds=max_rounds or r.max_rounds,
            finalization_llm_reserve=r.finalization_llm_reserve,
            default_top_k=r.initial_top_k,
            default_read_tokens=r.max_read_tokens_per_call,
            model=self._settings.llm.model,
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

        context = ctx or RetrievalContext.create(
            limits=self._budget_limits(),
            permitted_document_ids=permitted_document_ids,
            query_id=request.query_id,
        )

        # E+: one-round Search+ → strict sufficiency → Alg1 fallback with gaps
        if mode is ExecutionMode.VIKINGRAG_E_PLUS and self._search_plus is not None:
            from vikingrag.application.agent.loop import AgenticRetrievalLoop
            from vikingrag.application.orchestration.query import QueryOrchestrator, QueryRequest
            from vikingrag.domain.models.experience import QueryRunRoute

            agent_exec = self._agent(max_rounds=request.max_rounds, use_search_plus=True)
            orch = QueryOrchestrator(
                search=self._search,
                search_plus=self._search_plus,
                assessor=self._assessor,
                agent=AgenticRetrievalLoop(agent_exec),
                read_service=self._read,
                llm=self._llm,
                llm_model=self._settings.llm.model,
            )
            orch_resp = await orch.run(
                QueryRequest(
                    query=request.question,
                    mode=ExecutionMode.VIKINGRAG_E_PLUS,
                    document_ids=request.document_ids,
                    top_k=self._settings.retrieval.initial_top_k,
                ),
                ctx=context,
            )
            if orch_resp.route is QueryRunRoute.ONE_ROUND and not orch_resp.gaps:
                return AnswerResponse(
                    query_id=request.query_id,
                    status=AnswerStatus.ANSWERED,
                    answer=orch_resp.answer,
                    citations=(),
                    evidence=tuple(orch_resp.evidence.items),
                    execution_mode=mode,
                    route="one_round_search_plus",
                    rounds_used=0,
                    usage={},
                    metadata={
                        "cold_path": orch_resp.cold_path,
                        "e_plus": True,
                        **orch_resp.metadata,
                    },
                )
            route = "e_plus_escalated"
            escalation_reason = (
                "gaps:" + ",".join(orch_resp.gaps)
                if orch_resp.gaps
                else "strict_sufficiency_failed"
            )

        agent = self._agent(max_rounds=request.max_rounds, use_search_plus=use_plus)
        result = await agent.run(
            request.question,
            ctx=context,
            instructions=request.instructions,
        )

        status = result.status
        if mode is ExecutionMode.VIKINGRAG_E and use_plus:
            route = "agentic_search_plus"
        elif mode is ExecutionMode.VIKINGRAG:
            route = "agentic"

        logger.info(
            "answer_generated",
            query_id=str(result.query_id or request.query_id),
            status=status.value,
            mode=mode.value,
            rounds=result.rounds_used,
        )

        meta: dict[str, Any] = {
            **result.metadata,
            "trace_id": result.trace_id,
            "escalation_reason": escalation_reason,
        }
        if self._database is not None and status is AnswerStatus.ANSWERED:
            try:
                from vikingrag.application.experience.enqueue import enqueue_experience_learning
                from vikingrag.domain.models.experience import QueryRunRoute, QueryRunStatus

                cached = None
                identity = self._search.embedding_identity()
                cache_key = (request.question, identity.key())
                cached = context.get_cached_query_embedding(cache_key)
                run = await enqueue_experience_learning(
                    self._database,
                    query_text=request.question,
                    answer=result.answer,
                    status=QueryRunStatus.SUCCEEDED,
                    route=QueryRunRoute.ESCALATED
                    if "agentic" in route or "escalated" in route
                    else QueryRunRoute.ONE_ROUND,
                    query_embedding=list(cached) if cached else None,
                    embedding_identity=identity if cached else None,
                    trace_events=list(result.events),
                    citation_uris=[c.uri for c in result.citations],
                    metadata={"query_id": str(result.query_id or request.query_id)},
                )
                if run is not None:
                    meta["experience_query_run_id"] = str(run.id)
                    meta["edge_build_status"] = run.edge_build_status.value
            except Exception as exc:
                logger.warning("experience_enqueue_failed", error=str(exc))

        return AnswerResponse(
            query_id=result.query_id or request.query_id,
            status=status,
            answer=result.answer,
            citations=tuple(result.citations),
            evidence=tuple(result.evidence),
            execution_mode=mode,
            route=route,
            rounds_used=result.rounds_used,
            abstain_reason=escalation_reason or result.metadata.get("abstain_reason"),
            usage=dict(result.usage),
            trace_events=tuple(result.events),
            metadata=meta,
        )
