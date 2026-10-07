"""Evidence retrieval application service: collect + assess."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from vikingrag.application.assessment import EvidenceAssessor
from vikingrag.application.budget import BudgetLimits, RetrievalContext
from vikingrag.application.evidence_collector import (
    EvidenceCollectionResult,
    EvidenceCollectionSettings,
    EvidenceCollector,
)
from vikingrag.domain.models.assessment import EvidenceAssessment
from vikingrag.domain.models.document import DocumentId
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class EvidenceQueryResult:
    query: str
    query_id: UUID
    trace_id: str
    collection: EvidenceCollectionResult
    assessment: EvidenceAssessment
    usage: dict[str, Any]
    timings_ms: dict[str, float]


class EvidenceRetrievalService:
    def __init__(
        self,
        *,
        collector: EvidenceCollector,
        assessor: EvidenceAssessor,
        default_settings: EvidenceCollectionSettings | None = None,
    ) -> None:
        self._collector = collector
        self._assessor = assessor
        self._defaults = default_settings or EvidenceCollectionSettings()

    async def retrieve_evidence(
        self,
        query: str,
        *,
        document_ids: tuple[DocumentId, ...] = (),
        settings: EvidenceCollectionSettings | None = None,
        ctx: RetrievalContext | None = None,
        permitted_document_ids: frozenset[DocumentId] | None = None,
    ) -> EvidenceQueryResult:
        if not query.strip():
            raise ValueError("query must be non-empty")
        cfg = settings or self._defaults
        context = ctx or RetrievalContext.create(
            permitted_document_ids=permitted_document_ids,
            limits=BudgetLimits(
                max_tool_calls=32,
                max_read_tokens=max(cfg.bundle_max_tokens * 3, 12_000),
                max_wall_time_ms=45_000,
                max_embedding_calls=6,
                max_vector_searches=6,
                max_llm_calls=4,
                max_nodes_inspected=300,
            ),
        )
        if permitted_document_ids is not None:
            context.permitted_document_ids = permitted_document_ids

        collection = await self._collector.collect(
            query,
            document_ids=document_ids,
            settings=cfg,
            ctx=context,
        )
        assessment = await self._assessor.assess(query, collection.bundle, ctx=context)

        logger.info(
            "evidence_query_completed",
            query_id=str(context.query_id),
            trace_id=context.trace_id,
            assessment_status=assessment.status.value,
            coverage=assessment.coverage,
            evidence_items=len(collection.bundle.items),
            missing_aspects=len(assessment.missing_aspects),
        )
        timings = {
            **collection.timings_ms,
            "assessment_ms": assessment.timing_ms,
        }
        return EvidenceQueryResult(
            query=query,
            query_id=context.query_id,
            trace_id=context.trace_id,
            collection=collection,
            assessment=assessment,
            usage=context.usage.snapshot(),
            timings_ms=timings,
        )
