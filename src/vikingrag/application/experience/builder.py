"""Algorithm 2: online experience construction from a successful query run."""

from __future__ import annotations

from dataclasses import dataclass

from vikingrag.application.experience.support_select import (
    DeterministicSupportSelector,
    SupportSelector,
)
from vikingrag.application.experience.trace_sets import extract_trace_uri_sets, summarize_trace
from vikingrag.domain.errors import DomainError
from vikingrag.domain.models.experience import (
    EdgeBuildStatus,
    ExperienceEdge,
    ExperienceEdgeStatus,
    ExperiencePayload,
    QueryRun,
    QueryRunStatus,
    new_experience_edge_id,
    new_experience_payload_id,
)
from vikingrag.infrastructure.database.repositories.experience import (
    ExperienceEdgeRepository,
    QueryRunRepository,
    RetrievalEventRepository,
)
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


class ExperienceLearnError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="experience_learn_error")


@dataclass(frozen=True, slots=True)
class ExperienceBuildResult:
    run_id: object
    payload: ExperiencePayload | None
    edges_created: tuple[ExperienceEdge, ...]
    edges_suppressed: int
    skipped: bool
    skip_reason: str | None = None


class ExperienceEdgeBuilder:
    """Construct directed experience edges from a completed successful trace.

    Invariants:
    - Never learn from FAILED / ABSTAINED runs.
    - U_tgt = selected_support - U_edge.
    - No self-edges.
    - Idempotent per query_run (existing payload → skip).
    - Suppress duplicate active (source, target) under compatible embedding identity.
    """

    def __init__(
        self,
        *,
        runs: QueryRunRepository,
        events: RetrievalEventRepository,
        edges: ExperienceEdgeRepository,
        max_support: int = 16,
        support_selector: SupportSelector | None = None,
    ) -> None:
        self._runs = runs
        self._events = events
        self._edges = edges
        self._max_support = max_support
        self._support: SupportSelector = support_selector or DeterministicSupportSelector(
            max_support=max_support
        )

    async def build_for_run(self, run: QueryRun) -> ExperienceBuildResult:
        if not run.is_learnable:
            reason = f"run status={run.status.value} route={run.route}"
            logger.info("experience_build_skipped", run_id=str(run.id), reason=reason)
            run.edge_build_status = EdgeBuildStatus.SKIPPED
            run.edge_build_error = reason
            await self._runs.update(run)
            return ExperienceBuildResult(
                run_id=run.id,
                payload=None,
                edges_created=(),
                edges_suppressed=0,
                skipped=True,
                skip_reason=reason,
            )

        if run.query_embedding is None or run.embedding_identity is None:
            raise ExperienceLearnError(
                f"QueryRun {run.id} missing query_embedding / embedding_identity"
            )

        # Idempotency: one payload per successful run
        existing = await self._edges.get_payload_for_run(run.id)
        if existing is not None:
            run.edge_build_status = EdgeBuildStatus.COMPLETED
            await self._runs.update(run)
            return ExperienceBuildResult(
                run_id=run.id,
                payload=existing,
                edges_created=(),
                edges_suppressed=0,
                skipped=True,
                skip_reason="payload_already_exists",
            )

        event_rows = await self._events.list_for_run(run.id)
        sets = extract_trace_uri_sets(event_rows)
        citations = list(run.metadata.get("citation_uris") or [])
        summary = summarize_trace(event_rows)
        selected = await self._support.select(
            sets.u_cand,
            question=run.query_text,
            answer=run.answer,
            citations=citations,
            trace_summary=summary,
        )
        # U_tgt = selected - U_edge (do not re-learn edge-only discoveries)
        u_tgt = selected - sets.u_edge

        if not sets.u_src or not u_tgt:
            reason = "empty_u_src_or_u_tgt"
            run.edge_build_status = EdgeBuildStatus.SKIPPED
            run.edge_build_error = reason
            await self._runs.update(run)
            return ExperienceBuildResult(
                run_id=run.id,
                payload=None,
                edges_created=(),
                edges_suppressed=0,
                skipped=True,
                skip_reason=reason,
            )

        payload = ExperiencePayload(
            id=new_experience_payload_id(),
            query_run_id=run.id,
            query_text=run.query_text,
            query_embedding=list(run.query_embedding),
            embedding_identity=run.embedding_identity,
            trace_summary=summarize_trace(event_rows),
            support_uris=tuple(sorted(u_tgt)),
            metadata={"u_src_count": len(sets.u_src), "u_cand_count": len(sets.u_cand)},
        )
        payload = await self._edges.create_payload(payload)

        created: list[ExperienceEdge] = []
        suppressed = 0
        identity = run.embedding_identity
        for src in sorted(sets.u_src):
            for tgt in sorted(u_tgt):
                if src == tgt:
                    continue
                dup = await self._edges.find_active_pair(src, tgt, identity=identity)
                if dup is not None:
                    suppressed += 1
                    continue
                created.append(
                    ExperienceEdge(
                        id=new_experience_edge_id(),
                        payload_id=payload.id,
                        source_uri=src,
                        target_uri=tgt,
                        status=ExperienceEdgeStatus.ACTIVE,
                        support_score=1.0,
                        metadata={"from_run": str(run.id)},
                    )
                )

        if created:
            created = await self._edges.create_edges(created)

        run.status = QueryRunStatus.SUCCEEDED
        run.edge_build_status = EdgeBuildStatus.COMPLETED
        run.edge_build_error = None
        await self._runs.update(run)

        logger.info(
            "experience_edges_built",
            run_id=str(run.id),
            edges_created=len(created),
            edges_suppressed=suppressed,
            u_src=len(sets.u_src),
            u_tgt=len(u_tgt),
        )
        return ExperienceBuildResult(
            run_id=run.id,
            payload=payload,
            edges_created=tuple(created),
            edges_suppressed=suppressed,
            skipped=False,
        )
