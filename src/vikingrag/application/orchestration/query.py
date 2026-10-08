"""Query orchestrator with vikingrag / vikingrag_e / vikingrag_e_plus modes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vikingrag.application.agent.loop import (
    AgenticRetrievalLoop,
    AgentRetrieveResult,
    StubAgenticRetrievalLoop,
)
from vikingrag.application.assessment import EvidenceAssessor
from vikingrag.application.budget import RetrievalContext
from vikingrag.application.evidence_bundle import evidence_from_read
from vikingrag.application.evidence_collector import EvidenceCollector
from vikingrag.application.orchestration.candidate import (
    citation_uris_subset_of_evidence,
    draft_candidate_answer,
)
from vikingrag.application.search import SemanticSearchService
from vikingrag.application.search_plus import ExperienceAugmentedSearch, SearchPlusResponse
from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.domain.models.answer import ExecutionMode
from vikingrag.domain.models.assessment import AssessmentStatus, EvidenceAssessment
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.evidence import EvidenceBundle, RetrievedEvidence
from vikingrag.domain.models.experience import (
    EdgeBuildStatus,
    QueryRun,
    QueryRunRoute,
    QueryRunStatus,
    new_query_run_id,
)
from vikingrag.domain.models.primitives import ReadRequest
from vikingrag.domain.models.representation import SearchRequest
from vikingrag.domain.models.retrieval import RetrievalTrace
from vikingrag.observability.logging import get_logger
from vikingrag.providers.llm.base import LLMProvider

logger = get_logger(__name__)


def learning_events_from_eplus(
    plus: SearchPlusResponse,
    evidence: EvidenceBundle,
) -> tuple[dict[str, Any], ...]:
    """Build enqueue-compatible Search / EDGE_EXPAND / Read events for Alg-2.

    Observability ``RetrievalTrace`` records lack URIs and wrong names; do not
    use them for experience learning.
    """
    events: list[dict[str, Any]] = []
    seed_uris = tuple(h.uri for h in plus.base.candidates)
    events.append(
        {
            "name": "Search",
            "arguments": {"query": plus.base.query},
            "result_uris": list(seed_uris),
            "ok": True,
        }
    )
    expand_uris = tuple(e.target_uri for e in plus.expansions)
    if expand_uris:
        events.append(
            {
                "name": "EDGE_EXPAND",
                "arguments": {},
                "result_uris": list(expand_uris),
                "ok": True,
            }
        )
    for item in evidence.items:
        uri = item.uri
        events.append(
            {
                "name": "Read",
                "arguments": {"uri": uri},
                "result_uris": [uri],
                "ok": True,
            }
        )
    return tuple(events)


@dataclass(frozen=True, slots=True)
class QueryRequest:
    query: str
    mode: ExecutionMode = ExecutionMode.VIKINGRAG
    document_ids: tuple[DocumentId, ...] = ()
    top_k: int = 8
    max_reads: int = 8
    instructions: str | None = None


@dataclass(slots=True)
class QueryResponse:
    query: str
    mode: ExecutionMode
    route: QueryRunRoute
    status: QueryRunStatus
    evidence: EvidenceBundle
    assessment: EvidenceAssessment | None = None
    answer: str | None = None
    citations: tuple[Any, ...] = ()
    gaps: tuple[str, ...] = ()
    query_run_id: object | None = None
    cold_path: bool = False
    usage: dict[str, Any] = field(default_factory=dict)
    trace_events: tuple[dict[str, Any], ...] = ()
    agent_status: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class QueryOrchestrator:
    """Routes queries through paper execution modes.

    - vikingrag: Alg1 + ordinary Search
    - vikingrag_e: Alg1 + Search+
    - vikingrag_e_plus: one non-agentic Search+ → bounded Reads → strict
      sufficiency → return iff fully supported else Alg1 fallback with gaps
    """

    def __init__(
        self,
        *,
        search: SemanticSearchService,
        search_plus: ExperienceAugmentedSearch | None = None,
        evidence_collector: EvidenceCollector | None = None,
        assessor: EvidenceAssessor | None = None,
        agent: AgenticRetrievalLoop | StubAgenticRetrievalLoop | None = None,
        read_service: Any | None = None,
        llm: LLMProvider | None = None,
        llm_model: str | None = None,
    ) -> None:
        self._search = search
        self._search_plus = search_plus
        self._collector = evidence_collector
        self._assessor = assessor
        self._agent: AgenticRetrievalLoop | StubAgenticRetrievalLoop = (
            agent or StubAgenticRetrievalLoop()
        )
        self._read = read_service
        self._llm = llm
        self._llm_model = llm_model

    async def run(
        self,
        request: QueryRequest,
        *,
        ctx: RetrievalContext | None = None,
    ) -> QueryResponse:
        if request.mode is ExecutionMode.VIKINGRAG_E_PLUS:
            return await self._run_e_plus(request, ctx=ctx)
        if request.mode is ExecutionMode.VIKINGRAG_E:
            return await self._run_agentic(request, use_search_plus=True, ctx=ctx)
        return await self._run_agentic(request, use_search_plus=False, ctx=ctx)

    async def _run_agentic(
        self,
        request: QueryRequest,
        *,
        use_search_plus: bool,
        ctx: RetrievalContext | None,
    ) -> QueryResponse:
        result: AgentRetrieveResult = await self._agent.retrieve(
            request.query,
            mode=request.mode,
            use_search_plus=use_search_plus,
            ctx=ctx,
            instructions=request.instructions,
        )
        return QueryResponse(
            query=request.query,
            mode=request.mode,
            route=QueryRunRoute.ESCALATED,
            status=QueryRunStatus.SUCCEEDED,
            evidence=result.evidence,
            answer=result.answer,
            citations=tuple(result.citations),
            gaps=result.gaps,
            usage=dict(result.usage),
            trace_events=tuple(result.events),
            agent_status=result.status,
            metadata={"rounds": result.rounds, "tool_calls": result.tool_calls},
        )

    async def _run_e_plus(
        self,
        request: QueryRequest,
        *,
        ctx: RetrievalContext | None,
    ) -> QueryResponse:
        """Section 5: one-round Search+ path with strict sufficiency gate."""
        if self._search_plus is None:
            raise NotImplementedCapabilityError("search_plus")

        context = ctx or RetrievalContext.create()
        trace = RetrievalTrace(query_id=context.query_id)
        run_id = new_query_run_id()

        plus = await self._search_plus.search(
            SearchRequest(
                query=request.query,
                top_k=request.top_k,
                document_ids=request.document_ids,
            ),
            ctx=context,
            trace=trace,
        )

        evidence = await self._bounded_reads(
            plus.expansion.all_uris[: request.max_reads],
            ctx=context,
        )
        learning_events = learning_events_from_eplus(plus, evidence)

        assessment: EvidenceAssessment | None = None
        candidate_answer: str | None = None
        citations: list[Any] = []
        candidate_meta: dict[str, Any] = {}

        def _usage() -> dict[str, Any]:
            return dict(context.usage.snapshot())

        # Section 5: draft candidate A, then claim-aware strict sufficiency.
        if self._assessor is None:
            gaps: tuple[str, ...] = ("assessor_unavailable",)
        elif not evidence.items:
            gaps = ("empty_evidence",)
            assessment = await self._assessor.assess(request.query, evidence, ctx=context)
        elif self._llm is None:
            # Assessor alone (scripted) — still require SUFFICIENT; no answer text.
            assessment = await self._assessor.assess(request.query, evidence, ctx=context)
            if assessment.status is AssessmentStatus.SUFFICIENT:
                return QueryResponse(
                    query=request.query,
                    mode=request.mode,
                    route=QueryRunRoute.ONE_ROUND,
                    status=QueryRunStatus.SUCCEEDED,
                    evidence=evidence,
                    assessment=assessment,
                    answer=None,
                    citations=(),
                    gaps=(),
                    query_run_id=run_id,
                    cold_path=plus.cold_path,
                    usage=_usage(),
                    trace_events=learning_events,
                    metadata={
                        "search_plus": True,
                        "trace_event_count": len(learning_events),
                        "note": "sufficient_without_candidate_llm",
                    },
                )
            gaps = tuple(assessment.missing_aspects) or tuple(assessment.unsupported_aspects)
            if not gaps:
                gaps = ("strict_sufficiency_failed",)
        else:
            candidate_answer, citations, candidate_meta = await draft_candidate_answer(
                question=request.query,
                evidence=evidence,
                llm=self._llm,
                ctx=context,
                model=self._llm_model,
            )
            assessment = await self._assessor.assess(
                request.query,
                evidence,
                ctx=context,
                candidate_answer=candidate_answer,
            )
            citations_ok = bool(citations) and citation_uris_subset_of_evidence(citations, evidence)
            if (
                candidate_answer
                and citations_ok
                and assessment.status is AssessmentStatus.SUFFICIENT
            ):
                return QueryResponse(
                    query=request.query,
                    mode=request.mode,
                    route=QueryRunRoute.ONE_ROUND,
                    status=QueryRunStatus.SUCCEEDED,
                    evidence=evidence,
                    assessment=assessment,
                    answer=candidate_answer,
                    citations=tuple(citations),
                    gaps=(),
                    query_run_id=run_id,
                    cold_path=plus.cold_path,
                    usage=_usage(),
                    trace_events=learning_events,
                    metadata={
                        "search_plus": True,
                        "trace_event_count": len(learning_events),
                        "candidate": candidate_meta,
                        "citation_count": len(citations),
                    },
                )
            gaps = tuple(assessment.missing_aspects) or tuple(assessment.unsupported_aspects)
            if not candidate_answer:
                gaps = (*gaps, "candidate_answer_failed")
            elif not citations_ok:
                gaps = (*gaps, "citations_invalid")
            if not gaps:
                gaps = ("strict_sufficiency_failed",)

        try:
            agent_result = await self._agent.retrieve(
                request.query,
                mode=ExecutionMode.VIKINGRAG_E_PLUS,
                initial_evidence=evidence,
                gaps=gaps,
                use_search_plus=True,
                ctx=context,
                instructions=request.instructions,
            )
            return QueryResponse(
                query=request.query,
                mode=request.mode,
                route=QueryRunRoute.ESCALATED,
                status=QueryRunStatus.SUCCEEDED,
                evidence=agent_result.evidence,
                assessment=assessment,
                answer=agent_result.answer,
                citations=tuple(agent_result.citations),
                gaps=agent_result.gaps or gaps,
                query_run_id=run_id,
                cold_path=plus.cold_path,
                usage=dict(agent_result.usage) or _usage(),
                trace_events=tuple(agent_result.events),
                agent_status=agent_result.status,
                metadata={
                    "fallback": "algorithm_1",
                    "gaps": list(gaps),
                    "candidate": candidate_meta,
                    "rounds": agent_result.rounds,
                    "tool_calls": agent_result.tool_calls,
                },
            )
        except NotImplementedCapabilityError:
            return QueryResponse(
                query=request.query,
                mode=request.mode,
                route=QueryRunRoute.ONE_ROUND,
                status=QueryRunStatus.ABSTAINED,
                evidence=evidence,
                assessment=assessment,
                citations=tuple(citations),
                gaps=gaps,
                query_run_id=run_id,
                cold_path=plus.cold_path,
                usage=_usage(),
                trace_events=learning_events,
                metadata={
                    "fallback": "algorithm_1_stub",
                    "gaps": list(gaps),
                    "note": "agentic loop not wired; returning one-round gaps",
                },
            )

    async def _bounded_reads(
        self,
        uris: tuple[str, ...],
        *,
        ctx: RetrievalContext,
    ) -> EvidenceBundle:
        items: list[RetrievedEvidence] = []
        if self._read is None:
            return EvidenceBundle.from_items([])

        for i, uri in enumerate(uris):
            try:
                resp = await self._read.read(ReadRequest(uri=uri), ctx=ctx)
                ev = evidence_from_read(
                    resp,
                    evidence_id=f"e{i + 1}",
                    provenance=("search_plus",),
                )
                if ev is not None:
                    items.append(ev)
            except Exception as exc:
                logger.warning("e_plus_read_failed", uri=uri, error=str(exc))
        return EvidenceBundle.from_items(items)


def mark_run_for_edge_build(run: QueryRun) -> QueryRun:
    """Enqueue durable Alg-2 job via status (worker polls PENDING)."""
    if run.is_learnable:
        run.edge_build_status = EdgeBuildStatus.PENDING
    else:
        run.edge_build_status = EdgeBuildStatus.SKIPPED
        run.edge_build_error = "not_learnable"
    return run
