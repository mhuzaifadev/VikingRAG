"""Hierarchical summarization + multi-granular embedding indexing."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

from vikingrag.domain.errors import DocumentNotFound, DocumentNotReadyError, IndexingError
from vikingrag.domain.models.document import (
    AbstractStatus,
    DocumentId,
    DocumentStatus,
    NodeId,
    NodeType,
)
from vikingrag.domain.models.node import DocumentNode
from vikingrag.domain.models.representation import (
    EmbeddingIdentity,
    IndexStage,
    NodeEmbedding,
    NodeRepresentation,
    RepresentationType,
    SummaryRequest,
)
from vikingrag.domain.summary import SummaryGenerator
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.embedding import (
    SqlEmbeddingRepository,
    new_embedding_id,
)
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.infrastructure.database.repositories.representation import (
    SqlRepresentationRepository,
    new_representation_id,
)
from vikingrag.ingestion.hashing import hash_text
from vikingrag.observability.logging import get_logger
from vikingrag.providers.embeddings.base import EmbeddingProvider
from vikingrag.settings.config import EmbeddingSettings, IndexingSettings

logger = get_logger(__name__)


@dataclass(slots=True)
class IndexingMetrics:
    summaries_generated: int = 0
    summaries_reused: int = 0
    embeddings_generated: int = 0
    embeddings_reused: int = 0
    summary_input_tokens: int = 0
    summary_output_tokens: int = 0
    embedding_tokens: int = 0
    summary_latency_ms: float = 0.0
    embedding_latency_ms: float = 0.0
    total_latency_ms: float = 0.0


@dataclass(slots=True)
class IndexStatus:
    document_id: DocumentId
    status: DocumentStatus
    stage: IndexStage
    representation_count: int
    embedding_count: int
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IndexResult:
    document_id: DocumentId
    status: DocumentStatus
    stage: IndexStage
    metrics: IndexingMetrics
    force_summaries: bool
    force_embeddings: bool


class DocumentIndexingService:
    """Separate from structural ingest - summaries then embeddings."""

    def __init__(
        self,
        *,
        database: Database,
        summary_generator: SummaryGenerator,
        embedding_provider: EmbeddingProvider,
        indexing_settings: IndexingSettings,
        embedding_settings: EmbeddingSettings,
    ) -> None:
        self._database = database
        self._summaries = summary_generator
        self._embeddings = embedding_provider
        self._indexing = indexing_settings
        self._embedding_settings = embedding_settings

    def embedding_identity(self) -> EmbeddingIdentity:
        return EmbeddingIdentity(
            provider=self._embeddings.provider_name,
            model=self._embeddings.default_model,
            dimensions=self._embeddings.dimensions,
            version=self._embedding_settings.identity_version,
        )

    async def get_status(self, document_id: DocumentId) -> IndexStatus:
        async with self._database.session() as session:
            docs = SqlDocumentRepository(session)
            reps = SqlRepresentationRepository(session)
            embs = SqlEmbeddingRepository(session)
            record = await docs.get(document_id)
            if record is None:
                raise DocumentNotFound(f"Document not found: {document_id}")
            identity = self.embedding_identity()
            rep_count = await reps.count_for_document(document_id)
            emb_count = await embs.count_for_document(document_id, identity=identity)
            stage = _status_to_stage(record.status, rep_count=rep_count, emb_count=emb_count)
            return IndexStatus(
                document_id=document_id,
                status=record.status,
                stage=stage,
                representation_count=rep_count,
                embedding_count=emb_count,
                metrics=dict(record.metadata.get("indexing") or {}),
            )

    async def index_document(
        self,
        document_id: DocumentId,
        *,
        force_summaries: bool = False,
        force_embeddings: bool = False,
    ) -> IndexResult:
        started = time.perf_counter()
        metrics = IndexingMetrics()
        trace_id = str(uuid4())

        async with self._database.session() as session:
            docs = SqlDocumentRepository(session)
            record = await docs.get(document_id)
            if record is None:
                raise DocumentNotFound(f"Document not found: {document_id}")
            if record.status in {DocumentStatus.PENDING, DocumentStatus.PROCESSING}:
                raise DocumentNotReadyError(
                    f"Document {document_id} is not structurally ready (status={record.status})"
                )
            if record.status is DocumentStatus.FAILED:
                raise DocumentNotReadyError(f"Document {document_id} is in failed state")
            if record.status is DocumentStatus.DELETED:
                raise DocumentNotFound(f"Document not found: {document_id}")

        logger.info(
            "indexing_started",
            document_id=str(document_id),
            trace_id=trace_id,
            force_summaries=force_summaries,
            force_embeddings=force_embeddings,
            stage=IndexStage.SUMMARIZING.value,
        )

        try:
            await self._generate_summaries(
                document_id,
                force=force_summaries,
                metrics=metrics,
                trace_id=trace_id,
            )
            await self._set_status(document_id, DocumentStatus.SUMMARIZED, metrics)

            await self._generate_embeddings(
                document_id,
                force=force_embeddings,
                metrics=metrics,
                trace_id=trace_id,
            )
            await self._set_status(document_id, DocumentStatus.INDEXED, metrics)
        except Exception as exc:
            logger.exception(
                "indexing_failed",
                document_id=str(document_id),
                trace_id=trace_id,
                error=str(exc),
            )
            # Do not revert READY/SUMMARIZED structural success into FAILED blindly;
            # mark failed only if we never had a durable hierarchy (already READY+).
            raise IndexingError(f"Indexing failed for {document_id}: {exc}") from exc

        metrics.total_latency_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "indexing_completed",
            document_id=str(document_id),
            trace_id=trace_id,
            stage=IndexStage.INDEXED.value,
            summaries_generated=metrics.summaries_generated,
            embeddings_generated=metrics.embeddings_generated,
            total_latency_ms=metrics.total_latency_ms,
        )
        return IndexResult(
            document_id=document_id,
            status=DocumentStatus.INDEXED,
            stage=IndexStage.INDEXED,
            metrics=metrics,
            force_summaries=force_summaries,
            force_embeddings=force_embeddings,
        )

    async def _set_status(
        self,
        document_id: DocumentId,
        status: DocumentStatus,
        metrics: IndexingMetrics,
    ) -> None:
        async with self._database.session() as session:
            docs = SqlDocumentRepository(session)
            record = await docs.get(document_id)
            if record is None:
                raise DocumentNotFound(f"Document not found: {document_id}")
            record.status = status
            meta = dict(record.metadata)
            meta["indexing"] = {
                "summaries_generated": metrics.summaries_generated,
                "summaries_reused": metrics.summaries_reused,
                "embeddings_generated": metrics.embeddings_generated,
                "embeddings_reused": metrics.embeddings_reused,
                "summary_input_tokens": metrics.summary_input_tokens,
                "summary_output_tokens": metrics.summary_output_tokens,
                "embedding_tokens": metrics.embedding_tokens,
            }
            record.metadata = meta
            await docs.update(record)

    async def _generate_summaries(
        self,
        document_id: DocumentId,
        *,
        force: bool,
        metrics: IndexingMetrics,
        trace_id: str,
    ) -> None:
        async with self._database.session() as session:
            nodes_repo = SqlNodeRepository(session)
            nodes = await nodes_repo.list_by_document(document_id)

        by_parent: dict[NodeId | None, list[DocumentNode]] = defaultdict(list)
        for node in nodes:
            by_parent[node.parent_id].append(node)
        for children in by_parent.values():
            children.sort(key=lambda n: n.ordinal)

        # Deepest first
        ordered = sorted(nodes, key=lambda n: (-n.depth, n.ordinal))
        summary_text: dict[NodeId, str] = {}
        sem = asyncio.Semaphore(self._indexing.summary_concurrency)

        async def _process(node: DocumentNode) -> None:
            # Chunks: content representation only (no LLM abstract required)
            if node.node_type is NodeType.CHUNK:
                content = (node.content or "").strip()
                if not content:
                    return
                await self._upsert_representation(
                    node,
                    representation_type=RepresentationType.NODE_CONTENT,
                    content=content,
                    generator="ingest",
                    generator_model="raw",
                    version="1",
                    force=force,
                    metrics=metrics,
                    is_summary=False,
                )
                return

            child_nodes = by_parent.get(node.id, [])
            child_summaries: list[str] = []
            for child in child_nodes:
                if child.node_type is NodeType.CHUNK:
                    # Full owned chunk content (bounded later by summary max_input_tokens)
                    snippet = (child.content or child.title or "").strip()
                    if snippet:
                        child_summaries.append(snippet)
                elif child.id in summary_text:
                    child_summaries.append(summary_text[child.id])

            # Idempotency: reuse when source fingerprint unchanged
            request = SummaryRequest(
                node_id=node.id,
                node_type=node.node_type,
                title=node.title,
                own_content=node.content,
                child_summaries=tuple(child_summaries),
                max_input_tokens=self._indexing.summary_max_input_tokens,
                max_output_tokens=self._indexing.summary_max_output_tokens,
            )
            source_fingerprint = hash_text(_summary_fingerprint(request))

            async with self._database.session() as session:
                reps_repo = SqlRepresentationRepository(session)
                existing = await reps_repo.get(node.id, RepresentationType.NODE_SUMMARY)
                if (
                    existing is not None
                    and not force
                    and existing.metadata.get("source_fingerprint") == source_fingerprint
                    and existing.generator == self._summaries.name
                    and existing.generator_model == self._summaries.model
                    and existing.version == self._summaries.version
                ):
                    summary_text[node.id] = existing.content
                    metrics.summaries_reused += 1
                    return

            async with sem:
                started = time.perf_counter()
                result = await self._summaries.summarize(request)
                metrics.summary_latency_ms += (time.perf_counter() - started) * 1000.0
                if result.input_tokens:
                    metrics.summary_input_tokens += result.input_tokens
                if result.output_tokens:
                    metrics.summary_output_tokens += result.output_tokens

            await self._upsert_representation(
                node,
                representation_type=RepresentationType.NODE_SUMMARY,
                content=result.text,
                generator=self._summaries.name,
                generator_model=result.model,
                version=self._summaries.version,
                force=True,
                metrics=metrics,
                is_summary=True,
                source_fingerprint=source_fingerprint,
            )
            summary_text[node.id] = result.text
            metrics.summaries_generated += 1
            logger.info(
                "summary_generated",
                document_id=str(document_id),
                node_id=str(node.id),
                node_type=node.node_type.value,
                trace_id=trace_id,
                provider=self._summaries.name,
                model=result.model,
            )

        # Process depth groups sequentially so parents see child summaries
        depths = sorted({n.depth for n in ordered}, reverse=True)
        for depth in depths:
            group = [n for n in ordered if n.depth == depth]
            await asyncio.gather(*[_process(n) for n in group])

    async def _upsert_representation(
        self,
        node: DocumentNode,
        *,
        representation_type: RepresentationType,
        content: str,
        generator: str,
        generator_model: str,
        version: str,
        force: bool,
        metrics: IndexingMetrics,
        is_summary: bool,
        content_hash_override: str | None = None,
        source_fingerprint: str | None = None,
    ) -> NodeRepresentation:
        # content_hash always reflects output text; source_fingerprint is separate
        content_hash = content_hash_override or hash_text(content)
        meta: dict[str, object] = {}
        if source_fingerprint is not None:
            meta["source_fingerprint"] = source_fingerprint
        async with self._database.session() as session:
            reps = SqlRepresentationRepository(session)
            nodes = SqlNodeRepository(session)
            existing = await reps.get(node.id, representation_type)
            if (
                existing is not None
                and not force
                and existing.content_hash == content_hash
                and (
                    source_fingerprint is None
                    or existing.metadata.get("source_fingerprint") == source_fingerprint
                )
                and existing.generator == generator
                and existing.generator_model == generator_model
                and existing.version == version
            ):
                if is_summary:
                    metrics.summaries_reused += 1
                return existing

            representation = NodeRepresentation(
                id=existing.id if existing is not None else new_representation_id(),
                node_id=node.id,
                document_id=node.document_id,
                representation_type=representation_type,
                content=content,
                content_hash=content_hash,
                generator=generator,
                generator_model=generator_model,
                version=version,
                metadata=meta,
            )
            saved = await reps.upsert(representation)
            if is_summary:
                await nodes.update_abstract(
                    node.id,
                    abstract_text=content,
                    abstract_status=AbstractStatus.READY,
                )
            return saved

    async def _generate_embeddings(
        self,
        document_id: DocumentId,
        *,
        force: bool,
        metrics: IndexingMetrics,
        trace_id: str,
    ) -> None:
        identity = self.embedding_identity()
        async with self._database.session() as session:
            nodes_repo = SqlNodeRepository(session)
            reps_repo = SqlRepresentationRepository(session)
            embs_repo = SqlEmbeddingRepository(session)
            nodes = {n.id: n for n in await nodes_repo.list_by_document(document_id)}
            representations = await reps_repo.list_for_document(document_id)

            # Choose which representation to embed per node type
            to_embed: list[NodeRepresentation] = []
            for rep in representations:
                node = nodes.get(rep.node_id)
                if node is None:
                    continue
                if node.node_type is NodeType.CHUNK:
                    if rep.representation_type is RepresentationType.NODE_CONTENT:
                        to_embed.append(rep)
                else:
                    if rep.representation_type is RepresentationType.NODE_SUMMARY:
                        to_embed.append(rep)

            pending: list[NodeRepresentation] = []
            for rep in to_embed:
                existing = await embs_repo.get_for_representation(rep.id, identity)
                if (
                    existing is not None
                    and not force
                    and existing.metadata.get("content_hash") == rep.content_hash
                ):
                    metrics.embeddings_reused += 1
                    continue
                pending.append(rep)

        batch_size = self._embedding_settings.batch_size
        for offset in range(0, len(pending), batch_size):
            batch = pending[offset : offset + batch_size]
            texts = [r.content for r in batch]
            started = time.perf_counter()
            result = await self._embeddings.embed_batch(texts)
            metrics.embedding_latency_ms += (time.perf_counter() - started) * 1000.0
            if result.usage and result.usage.total_tokens:
                metrics.embedding_tokens += result.usage.total_tokens
            if len(result.vectors) != len(batch):
                raise IndexingError("Embedding provider returned unexpected vector count")

            async with self._database.session() as session:
                embs_repo = SqlEmbeddingRepository(session)
                for rep, vector in zip(batch, result.vectors, strict=True):
                    embedding = NodeEmbedding(
                        id=new_embedding_id(),
                        representation_id=rep.id,
                        node_id=rep.node_id,
                        document_id=rep.document_id,
                        identity=identity,
                        embedding=vector,
                        metadata={"content_hash": rep.content_hash},
                    )
                    # Preserve id on conflict by loading existing first
                    existing = await embs_repo.get_for_representation(rep.id, identity)
                    if existing is not None:
                        embedding = NodeEmbedding(
                            id=existing.id,
                            representation_id=rep.id,
                            node_id=rep.node_id,
                            document_id=rep.document_id,
                            identity=identity,
                            embedding=vector,
                            metadata={"content_hash": rep.content_hash},
                        )
                    await embs_repo.upsert(embedding)
                    metrics.embeddings_generated += 1
                    logger.info(
                        "embedding_persisted",
                        document_id=str(document_id),
                        node_id=str(rep.node_id),
                        representation_type=rep.representation_type.value,
                        trace_id=trace_id,
                        provider=identity.provider,
                        model=identity.model,
                    )


def _summary_fingerprint(request: SummaryRequest) -> str:
    parts = [
        request.node_type.value,
        request.title or "",
        request.own_content or "",
        "\n".join(request.child_summaries),
        str(request.max_input_tokens),
        str(request.max_output_tokens),
    ]
    return "\n".join(parts)


def _status_to_stage(
    status: DocumentStatus,
    *,
    rep_count: int,
    emb_count: int,
) -> IndexStage:
    if status is DocumentStatus.INDEXED and emb_count > 0:
        return IndexStage.INDEXED
    if status is DocumentStatus.SUMMARIZED or (rep_count > 0 and emb_count == 0):
        return IndexStage.SUMMARIZED
    if status is DocumentStatus.FAILED:
        return IndexStage.FAILED
    return IndexStage.STRUCTURED
