"""Bounded initial evidence collection - Search → List/Read → bundle (no agent loop)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from vikingrag.application.budget import BudgetLimits, RetrievalContext
from vikingrag.application.evidence_bundle import assemble_evidence_bundle, evidence_from_read
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.errors import BudgetExhaustedError, ScopeDeniedError
from vikingrag.domain.models.document import DocumentId, NodeType
from vikingrag.domain.models.evidence import (
    EvidenceBundle,
    ExcludedEvidence,
    ExclusionReason,
    RetrievedEvidence,
)
from vikingrag.domain.models.primitives import GrepRequest, ListRequest, ReadRequest
from vikingrag.domain.models.representation import SearchRequest
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class EvidenceCollectionSettings:
    search_top_k: int = 8
    search_pool: int = 30
    max_candidates: int = 8
    max_list_children: int = 5
    max_descent_depth: int = 1
    read_max_tokens: int = 400
    bundle_max_tokens: int = 4_000
    include_grep: bool = False
    grep_pattern: str | None = None


@dataclass(slots=True)
class EvidenceCollectionResult:
    query: str
    query_id: UUID
    trace_id: str
    bundle: EvidenceBundle
    search_uris: tuple[str, ...]
    usage: dict[str, Any] = field(default_factory=dict)
    timings_ms: dict[str, float] = field(default_factory=dict)
    truncation_reasons: tuple[str, ...] = ()
    complete: bool = True


class EvidenceCollector:
    def __init__(
        self,
        *,
        search: SemanticSearchService,
        list_service: ListService,
        read_service: ReadService,
        grep_service: GrepService | None = None,
    ) -> None:
        self._search = search
        self._list = list_service
        self._read = read_service
        self._grep = grep_service

    async def collect(
        self,
        query: str,
        *,
        document_ids: tuple[DocumentId, ...] = (),
        settings: EvidenceCollectionSettings | None = None,
        ctx: RetrievalContext | None = None,
    ) -> EvidenceCollectionResult:
        cfg = settings or EvidenceCollectionSettings()
        context = ctx or RetrievalContext.create(
            limits=BudgetLimits(
                max_tool_calls=24,
                max_read_tokens=max(cfg.bundle_max_tokens * 2, 8_000),
                max_wall_time_ms=30_000,
                max_embedding_calls=4,
                max_vector_searches=4,
                max_llm_calls=4,
            )
        )
        started = time.perf_counter()
        truncation: list[str] = []
        collected: list[RetrievedEvidence] = []
        excluded: list[ExcludedEvidence] = []
        search_uris: list[str] = []

        # 1) Search once
        search_started = time.perf_counter()
        try:
            search_response = await self._search.search(
                SearchRequest(
                    query=query,
                    top_k=cfg.search_top_k,
                    document_ids=document_ids,
                    candidate_pool_size=cfg.search_pool,
                ),
                ctx=context,
            )
        except BudgetExhaustedError as exc:
            truncation.append(f"search_budget:{exc.resource}")
            return self._result(
                query,
                context,
                EvidenceBundle.from_items([], excluded=excluded, truncated=True, complete=False),
                search_uris,
                truncation,
                started,
                {"search_ms": (time.perf_counter() - search_started) * 1000.0},
            )
        search_ms = (time.perf_counter() - search_started) * 1000.0

        candidates = list(search_response.candidates[: cfg.max_candidates])
        search_uris = [c.uri for c in candidates]

        # Optional Grep (explicit only)
        if cfg.include_grep and cfg.grep_pattern and self._grep is not None and document_ids:
            for doc_id in document_ids[:1]:
                try:
                    from vikingrag.domain.uri import VikingURIParser

                    grep_resp = await self._grep.grep(
                        GrepRequest(
                            uri=VikingURIParser.build_document_uri(doc_id),
                            pattern=cfg.grep_pattern,
                            max_matches=5,
                        ),
                        ctx=context,
                    )
                    for match in grep_resp.matches:
                        # Grep is discovery - must Read before evidence
                        try:
                            read = await self._read.read(
                                ReadRequest(
                                    uri=match.uri,
                                    start_offset=match.start_offset,
                                    max_tokens=cfg.read_max_tokens,
                                    expected_content_hash=match.content_hash or None,
                                ),
                                ctx=context,
                            )
                            ev = evidence_from_read(
                                read,
                                evidence_id="",
                                provenance=("grep", "read"),
                            )
                            if ev is not None:
                                collected.append(ev)
                        except Exception as exc:
                            excluded.append(
                                ExcludedEvidence(
                                    uri=match.uri,
                                    reason=ExclusionReason.READ_FAILED,
                                    detail=str(exc)[:200],
                                )
                            )
                except BudgetExhaustedError as exc:
                    truncation.append(f"grep_budget:{exc.resource}")
                except ScopeDeniedError:
                    truncation.append("grep_scope_denied")

        # 2-5) Resolve candidates with bounded List/Read descent
        for hit in candidates:
            context.check_deadline()
            try:
                await self._collect_from_uri(
                    hit.uri,
                    discovery_score=hit.score,
                    depth=0,
                    cfg=cfg,
                    context=context,
                    collected=collected,
                    excluded=excluded,
                    truncation=truncation,
                    provenance=("search",),
                )
            except BudgetExhaustedError as exc:
                truncation.append(f"collect_budget:{exc.resource}")
                break

        bundle = assemble_evidence_bundle(collected, max_tokens=cfg.bundle_max_tokens)
        # Merge exclusions
        all_excluded = list(bundle.excluded) + excluded
        bundle = EvidenceBundle.from_items(
            list(bundle.items),
            excluded=all_excluded,
            truncated=bundle.truncated or bool(truncation),
            complete=bundle.complete and not truncation,
        )
        if bundle.total_tokens > 0:
            await context.add_usage(evidence_tokens_retained=bundle.total_tokens)

        return self._result(
            query,
            context,
            bundle,
            search_uris,
            truncation,
            started,
            {"search_ms": search_ms},
        )

    async def _collect_from_uri(
        self,
        uri: str,
        *,
        discovery_score: float | None,
        depth: int,
        cfg: EvidenceCollectionSettings,
        context: RetrievalContext,
        collected: list[RetrievedEvidence],
        excluded: list[ExcludedEvidence],
        truncation: list[str],
        provenance: tuple[str, ...],
    ) -> None:
        read = await self._read.read(
            ReadRequest(uri=uri, max_tokens=cfg.read_max_tokens),
            ctx=context,
        )
        if read.has_direct_content:
            ev = evidence_from_read(
                read,
                evidence_id="",
                provenance=(*provenance, "read"),
                discovery_score=discovery_score,
            )
            if ev is None:
                excluded.append(
                    ExcludedEvidence(uri=uri, reason=ExclusionReason.EMPTY, detail="empty read")
                )
            else:
                collected.append(ev)
            return

        excluded.append(
            ExcludedEvidence(
                uri=uri,
                reason=ExclusionReason.NO_DIRECT_CONTENT,
                detail=f"node_type={read.node_type.value}",
            )
        )
        if depth >= cfg.max_descent_depth:
            truncation.append(f"max_descent_depth:{uri}")
            return
        if read.node_type is NodeType.CHUNK:
            return

        listing = await self._list.list(
            ListRequest(uri=uri, limit=cfg.max_list_children),
            ctx=context,
        )
        for child in listing.items:
            if not child.has_content and child.node_type in {
                NodeType.DOCUMENT,
                NodeType.SECTION,
                NodeType.SUBSECTION,
            }:
                # Descend one more level into structural children
                await self._collect_from_uri(
                    child.uri,
                    discovery_score=discovery_score,
                    depth=depth + 1,
                    cfg=cfg,
                    context=context,
                    collected=collected,
                    excluded=excluded,
                    truncation=truncation,
                    provenance=(*provenance, "list"),
                )
            elif child.has_content:
                await self._collect_from_uri(
                    child.uri,
                    discovery_score=discovery_score,
                    depth=depth + 1,
                    cfg=cfg,
                    context=context,
                    collected=collected,
                    excluded=excluded,
                    truncation=truncation,
                    provenance=(*provenance, "list"),
                )
            if listing.truncated:
                truncation.append(f"list_truncated:{uri}")

    def _result(
        self,
        query: str,
        context: RetrievalContext,
        bundle: EvidenceBundle,
        search_uris: list[str],
        truncation: list[str],
        started: float,
        timings: dict[str, float],
    ) -> EvidenceCollectionResult:
        timings = {**timings, "total_ms": (time.perf_counter() - started) * 1000.0}
        logger.info(
            "evidence_collection_completed",
            query_id=str(context.query_id),
            trace_id=context.trace_id,
            evidence_items=len(bundle.items),
            total_tokens=bundle.total_tokens,
            truncated=bundle.truncated or bool(truncation),
            complete=bundle.complete and not truncation,
        )
        return EvidenceCollectionResult(
            query=query,
            query_id=context.query_id,
            trace_id=context.trace_id,
            bundle=bundle,
            search_uris=tuple(search_uris),
            usage=context.usage.snapshot(),
            timings_ms=timings,
            truncation_reasons=tuple(truncation),
            complete=bundle.complete and not truncation,
        )
