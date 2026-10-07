"""Persistence for query runs, retrieval events, and experience edges."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.experience import (
    EdgeBuildStatus,
    ExperienceEdge,
    ExperienceEdgeId,
    ExperienceEdgeStatus,
    ExperiencePayload,
    ExperiencePayloadId,
    QueryRun,
    QueryRunId,
    QueryRunRoute,
    QueryRunStatus,
    RetrievalEvent,
    RetrievalEventId,
    RetrievalEventType,
)
from vikingrag.domain.models.representation import EmbeddingIdentity
from vikingrag.infrastructure.database.models import (
    ExperienceEdgeRow,
    ExperiencePayloadRow,
    QueryRunRow,
    RetrievalEventRow,
)


def _identity_from_run_row(row: QueryRunRow) -> EmbeddingIdentity | None:
    if not row.embedding_provider or not row.embedding_model or row.embedding_dimensions is None:
        return None
    return EmbeddingIdentity(
        provider=row.embedding_provider,
        model=row.embedding_model,
        dimensions=row.embedding_dimensions,
        version=row.embedding_identity_version or "1",
    )


def _to_query_run(row: QueryRunRow) -> QueryRun:
    emb = None
    if row.query_embedding is not None:
        emb = list(row.query_embedding)
    return QueryRun(
        id=QueryRunId(row.id if isinstance(row.id, UUID) else UUID(str(row.id))),
        query_text=row.query_text,
        status=QueryRunStatus(row.status),
        route=QueryRunRoute(row.route) if row.route else None,
        answer=row.answer,
        sufficiency_score=row.sufficiency_score,
        query_embedding=emb,
        embedding_identity=_identity_from_run_row(row),
        edge_build_status=EdgeBuildStatus(row.edge_build_status),
        edge_build_error=row.edge_build_error,
        tenant_id=row.tenant_id,
        corpus_id=row.corpus_id,
        document_ids=tuple(
            DocumentId(d if isinstance(d, UUID) else UUID(str(d))) for d in (row.document_ids or [])
        ),
        total_input_tokens=row.total_input_tokens,
        total_output_tokens=row.total_output_tokens,
        retrieval_tokens=row.retrieval_tokens,
        llm_calls=row.llm_calls,
        retrieval_rounds=row.retrieval_rounds,
        latency_ms=row.latency_ms,
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _to_event(row: RetrievalEventRow) -> RetrievalEvent:
    refs = row.result_refs or []
    return RetrievalEvent(
        id=RetrievalEventId(row.id if isinstance(row.id, UUID) else UUID(str(row.id))),
        query_run_id=QueryRunId(
            row.query_run_id if isinstance(row.query_run_id, UUID) else UUID(str(row.query_run_id))
        ),
        round_no=row.round_no,
        event_type=RetrievalEventType(row.event_type),
        arguments=dict(row.arguments or {}),
        result_refs=[str(r) for r in refs],
        latency_ms=row.latency_ms,
        token_cost=row.token_cost,
        created_at=row.created_at,
    )


def _to_payload(row: ExperiencePayloadRow) -> ExperiencePayload:
    return ExperiencePayload(
        id=ExperiencePayloadId(row.id if isinstance(row.id, UUID) else UUID(str(row.id))),
        query_run_id=QueryRunId(
            row.query_run_id if isinstance(row.query_run_id, UUID) else UUID(str(row.query_run_id))
        ),
        query_text=row.query_text,
        query_embedding=list(row.query_embedding),
        embedding_identity=EmbeddingIdentity(
            provider=row.embedding_provider,
            model=row.embedding_model,
            dimensions=row.embedding_dimensions,
            version=row.embedding_identity_version,
        ),
        trace_summary=row.trace_summary,
        support_uris=tuple(str(u) for u in (row.support_uris or [])),
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
    )


def _to_edge(row: ExperienceEdgeRow) -> ExperienceEdge:
    return ExperienceEdge(
        id=ExperienceEdgeId(row.id if isinstance(row.id, UUID) else UUID(str(row.id))),
        payload_id=ExperiencePayloadId(
            row.payload_id if isinstance(row.payload_id, UUID) else UUID(str(row.payload_id))
        ),
        source_uri=row.source_uri,
        target_uri=row.target_uri,
        status=ExperienceEdgeStatus(row.status),
        source_node_id=NodeId(row.source_node_id) if row.source_node_id else None,
        target_node_id=NodeId(row.target_node_id) if row.target_node_id else None,
        source_revision=row.source_revision,
        target_revision=row.target_revision,
        support_score=row.support_score,
        success_count=row.success_count,
        failure_count=row.failure_count,
        last_used_at=row.last_used_at,
        expires_at=row.expires_at,
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
    )


@runtime_checkable
class QueryRunRepository(Protocol):
    async def create(self, run: QueryRun) -> QueryRun: ...

    async def get(self, run_id: QueryRunId) -> QueryRun | None: ...

    async def update(self, run: QueryRun) -> QueryRun: ...

    async def claim_pending_edge_jobs(self, *, limit: int = 10) -> list[QueryRun]: ...


@runtime_checkable
class RetrievalEventRepository(Protocol):
    async def append(self, event: RetrievalEvent) -> RetrievalEvent: ...

    async def list_for_run(self, run_id: QueryRunId) -> list[RetrievalEvent]: ...


@runtime_checkable
class ExperienceEdgeRepository(Protocol):
    async def get_payload_for_run(self, run_id: QueryRunId) -> ExperiencePayload | None: ...

    async def create_payload(self, payload: ExperiencePayload) -> ExperiencePayload: ...

    async def create_edges(self, edges: list[ExperienceEdge]) -> list[ExperienceEdge]: ...

    async def find_active_by_source(
        self,
        source_uri: str,
        *,
        identity: EmbeddingIdentity,
    ) -> list[tuple[ExperienceEdge, ExperiencePayload]]: ...

    async def find_active_pair(
        self,
        source_uri: str,
        target_uri: str,
        *,
        identity: EmbeddingIdentity,
    ) -> ExperienceEdge | None: ...

    async def supersede(self, edge_id: ExperienceEdgeId) -> None: ...

    async def invalidate_by_uris(self, uris: set[str]) -> int: ...

    async def count_active(self) -> int: ...


class SqlQueryRunRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, run: QueryRun) -> QueryRun:
        identity = run.embedding_identity
        row = QueryRunRow(
            id=run.id,
            tenant_id=run.tenant_id,
            corpus_id=run.corpus_id,
            query_text=run.query_text,
            query_embedding=run.query_embedding,
            embedding_provider=identity.provider if identity else None,
            embedding_model=identity.model if identity else None,
            embedding_dimensions=identity.dimensions if identity else None,
            embedding_identity_version=identity.version if identity else None,
            route=run.route.value if run.route else None,
            answer=run.answer,
            sufficiency_score=run.sufficiency_score,
            status=run.status.value,
            edge_build_status=run.edge_build_status.value,
            edge_build_error=run.edge_build_error,
            total_input_tokens=run.total_input_tokens,
            total_output_tokens=run.total_output_tokens,
            retrieval_tokens=run.retrieval_tokens,
            llm_calls=run.llm_calls,
            retrieval_rounds=run.retrieval_rounds,
            latency_ms=run.latency_ms,
            document_ids=list(run.document_ids),
            metadata_=dict(run.metadata),
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return _to_query_run(row)

    async def get(self, run_id: QueryRunId) -> QueryRun | None:
        row = await self._session.get(QueryRunRow, run_id)
        return _to_query_run(row) if row else None

    async def update(self, run: QueryRun) -> QueryRun:
        row = await self._session.get(QueryRunRow, run.id)
        if row is None:
            raise LookupError(f"QueryRun not found: {run.id}")
        identity = run.embedding_identity
        row.query_text = run.query_text
        row.query_embedding = run.query_embedding
        row.embedding_provider = identity.provider if identity else None
        row.embedding_model = identity.model if identity else None
        row.embedding_dimensions = identity.dimensions if identity else None
        row.embedding_identity_version = identity.version if identity else None
        row.route = run.route.value if run.route else None
        row.answer = run.answer
        row.sufficiency_score = run.sufficiency_score
        row.status = run.status.value
        row.edge_build_status = run.edge_build_status.value
        row.edge_build_error = run.edge_build_error
        row.total_input_tokens = run.total_input_tokens
        row.total_output_tokens = run.total_output_tokens
        row.retrieval_tokens = run.retrieval_tokens
        row.llm_calls = run.llm_calls
        row.retrieval_rounds = run.retrieval_rounds
        row.latency_ms = run.latency_ms
        row.document_ids = list(run.document_ids)
        row.metadata_ = dict(run.metadata)
        row.updated_at = datetime.now(UTC)
        await self._session.flush()
        await self._session.refresh(row)
        return _to_query_run(row)

    async def claim_pending_edge_jobs(self, *, limit: int = 10) -> list[QueryRun]:
        """Atomically claim PENDING edge-build jobs (durable worker poll)."""
        stmt = (
            select(QueryRunRow)
            .where(QueryRunRow.edge_build_status == EdgeBuildStatus.PENDING.value)
            .where(QueryRunRow.status == QueryRunStatus.SUCCEEDED.value)
            .order_by(QueryRunRow.created_at.asc())
            .limit(limit)
            .with_for_update(skip_locked=True)
        )
        result = await self._session.execute(stmt)
        rows = list(result.scalars().all())
        claimed: list[QueryRun] = []
        now = datetime.now(UTC)
        for row in rows:
            row.edge_build_status = EdgeBuildStatus.PROCESSING.value
            row.updated_at = now
            claimed.append(_to_query_run(row))
        if rows:
            await self._session.flush()
        return claimed


class SqlRetrievalEventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def append(self, event: RetrievalEvent) -> RetrievalEvent:
        row = RetrievalEventRow(
            id=event.id,
            query_run_id=event.query_run_id,
            round_no=event.round_no,
            event_type=event.event_type.value,
            arguments=dict(event.arguments),
            result_refs=list(event.result_refs),
            latency_ms=event.latency_ms,
            token_cost=event.token_cost,
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return _to_event(row)

    async def list_for_run(self, run_id: QueryRunId) -> list[RetrievalEvent]:
        stmt = (
            select(RetrievalEventRow)
            .where(RetrievalEventRow.query_run_id == run_id)
            .order_by(RetrievalEventRow.round_no.asc(), RetrievalEventRow.created_at.asc())
        )
        result = await self._session.execute(stmt)
        return [_to_event(row) for row in result.scalars().all()]


class SqlExperienceEdgeRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_payload_for_run(self, run_id: QueryRunId) -> ExperiencePayload | None:
        stmt = select(ExperiencePayloadRow).where(ExperiencePayloadRow.query_run_id == run_id)
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_payload(row) if row else None

    async def create_payload(self, payload: ExperiencePayload) -> ExperiencePayload:
        identity = payload.embedding_identity
        row = ExperiencePayloadRow(
            id=payload.id,
            query_run_id=payload.query_run_id,
            query_text=payload.query_text,
            query_embedding=payload.query_embedding,
            embedding_provider=identity.provider,
            embedding_model=identity.model,
            embedding_dimensions=identity.dimensions,
            embedding_identity_version=identity.version,
            trace_summary=payload.trace_summary,
            support_uris=list(payload.support_uris),
            metadata_=dict(payload.metadata),
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return _to_payload(row)

    async def create_edges(self, edges: list[ExperienceEdge]) -> list[ExperienceEdge]:
        created: list[ExperienceEdge] = []
        for edge in edges:
            row = ExperienceEdgeRow(
                id=edge.id,
                payload_id=edge.payload_id,
                source_uri=edge.source_uri,
                target_uri=edge.target_uri,
                status=edge.status.value,
                source_node_id=edge.source_node_id,
                target_node_id=edge.target_node_id,
                source_revision=edge.source_revision,
                target_revision=edge.target_revision,
                support_score=edge.support_score,
                success_count=edge.success_count,
                failure_count=edge.failure_count,
                last_used_at=edge.last_used_at,
                expires_at=edge.expires_at,
                metadata_=dict(edge.metadata),
            )
            self._session.add(row)
            created.append(edge)
        await self._session.flush()
        return created

    async def find_active_by_source(
        self,
        source_uri: str,
        *,
        identity: EmbeddingIdentity,
    ) -> list[tuple[ExperienceEdge, ExperiencePayload]]:
        stmt = (
            select(ExperienceEdgeRow, ExperiencePayloadRow)
            .join(ExperiencePayloadRow, ExperienceEdgeRow.payload_id == ExperiencePayloadRow.id)
            .where(ExperienceEdgeRow.source_uri == source_uri)
            .where(ExperienceEdgeRow.status == ExperienceEdgeStatus.ACTIVE.value)
            .where(ExperiencePayloadRow.embedding_provider == identity.provider)
            .where(ExperiencePayloadRow.embedding_model == identity.model)
            .where(ExperiencePayloadRow.embedding_dimensions == identity.dimensions)
            .where(ExperiencePayloadRow.embedding_identity_version == identity.version)
        )
        result = await self._session.execute(stmt)
        return [(_to_edge(edge), _to_payload(payload)) for edge, payload in result.all()]

    async def find_active_pair(
        self,
        source_uri: str,
        target_uri: str,
        *,
        identity: EmbeddingIdentity,
    ) -> ExperienceEdge | None:
        stmt = (
            select(ExperienceEdgeRow)
            .join(ExperiencePayloadRow, ExperienceEdgeRow.payload_id == ExperiencePayloadRow.id)
            .where(ExperienceEdgeRow.source_uri == source_uri)
            .where(ExperienceEdgeRow.target_uri == target_uri)
            .where(ExperienceEdgeRow.status == ExperienceEdgeStatus.ACTIVE.value)
            .where(ExperiencePayloadRow.embedding_provider == identity.provider)
            .where(ExperiencePayloadRow.embedding_model == identity.model)
            .where(ExperiencePayloadRow.embedding_dimensions == identity.dimensions)
            .where(ExperiencePayloadRow.embedding_identity_version == identity.version)
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_edge(row) if row else None

    async def supersede(self, edge_id: ExperienceEdgeId) -> None:
        stmt = (
            update(ExperienceEdgeRow)
            .where(ExperienceEdgeRow.id == edge_id)
            .values(status=ExperienceEdgeStatus.SUPERSEDED.value)
        )
        await self._session.execute(stmt)

    async def invalidate_by_uris(self, uris: set[str]) -> int:
        if not uris:
            return 0
        stmt = (
            update(ExperienceEdgeRow)
            .where(ExperienceEdgeRow.status == ExperienceEdgeStatus.ACTIVE.value)
            .where(
                (ExperienceEdgeRow.source_uri.in_(uris)) | (ExperienceEdgeRow.target_uri.in_(uris))
            )
            .values(status=ExperienceEdgeStatus.INVALIDATED.value)
        )
        result = await self._session.execute(stmt)
        return int(getattr(result, "rowcount", 0) or 0)

    async def count_active(self) -> int:
        from sqlalchemy import func

        stmt = (
            select(func.count())
            .select_from(ExperienceEdgeRow)
            .where(ExperienceEdgeRow.status == ExperienceEdgeStatus.ACTIVE.value)
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())
