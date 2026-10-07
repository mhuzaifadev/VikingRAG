"""Ingestion pipeline: load → parse → hierarchy → chunk → persist."""

from __future__ import annotations

import time
from uuid import UUID, uuid4

from vikingrag.domain.models.document import (
    AbstractStatus,
    DocumentId,
    DocumentStatus,
    IngestionStrategy,
    NodeId,
    NodeType,
)
from vikingrag.domain.models.node import DocumentNode
from vikingrag.domain.uri import VikingURIParser
from vikingrag.infrastructure.database.repositories.base import DocumentRecord
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.infrastructure.object_store.base import ObjectStore
from vikingrag.ingestion.chunking import (
    ChunkingConfig,
    ChunkingPolicy,
    StructureAwareChunkingPolicy,
)
from vikingrag.ingestion.hashing import hash_bytes, hash_text
from vikingrag.ingestion.hierarchy import build_hierarchy
from vikingrag.ingestion.parsers.registry import ParserRegistry, build_default_parser_registry
from vikingrag.ingestion.tokenization import Tokenizer, create_tokenizer
from vikingrag.ingestion.types import (
    HierarchyDraftNode,
    IngestionInput,
    IngestionResult,
    LoadedDocument,
)
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    def __init__(
        self,
        *,
        documents: SqlDocumentRepository,
        nodes: SqlNodeRepository,
        object_store: ObjectStore,
        parsers: ParserRegistry | None = None,
        chunking: ChunkingPolicy | None = None,
        tokenizer: Tokenizer | None = None,
        chunking_config: ChunkingConfig | None = None,
    ) -> None:
        self._documents = documents
        self._nodes = nodes
        self._object_store = object_store
        self._parsers = parsers or build_default_parser_registry()
        self._tokenizer = tokenizer or create_tokenizer()
        self._chunking = chunking or StructureAwareChunkingPolicy(
            config=chunking_config,
            tokenizer=self._tokenizer,
        )

    async def run(self, inbound: IngestionInput) -> IngestionResult:
        ingestion_id = str(uuid4())
        started = time.perf_counter()
        stages: dict[str, float] = {}

        def mark(stage: str, t0: float) -> None:
            stages[stage] = round((time.perf_counter() - t0) * 1000, 3)

        t0 = time.perf_counter()
        loaded = self._load(inbound)
        mark("load", t0)

        logger.info(
            "ingestion_started",
            ingestion_id=ingestion_id,
            filename=loaded.filename,
            mime_type=loaded.mime_type,
            byte_size=loaded.byte_size,
            strategy=inbound.strategy.value,
            content_hash=loaded.content_hash,
        )

        # Idempotency gate
        existing = await self._resolve_existing(inbound.strategy, loaded)
        if existing is not None and inbound.strategy is IngestionStrategy.SKIP_IDENTICAL:
            duration_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "ingestion_skipped_identical",
                ingestion_id=ingestion_id,
                document_id=str(existing.id),
                content_hash=loaded.content_hash,
                duration_ms=round(duration_ms, 3),
            )
            return IngestionResult(
                document_id=existing.id,
                content_hash=loaded.content_hash,
                status=existing.status.value,
                skipped=True,
                strategy=inbound.strategy,
                structural_node_count=0,
                chunk_count=0,
                ingestion_id=ingestion_id,
                duration_ms=round(duration_ms, 3),
                stage_durations_ms=stages,
            )

        document_id = (
            existing.id
            if existing and inbound.strategy is IngestionStrategy.REPLACE
            else DocumentId(uuid4())
        )

        t0 = time.perf_counter()
        parser = self._parsers.resolve(filename=loaded.filename, mime_type=loaded.mime_type)
        parsed = parser.parse(loaded.content, filename=loaded.filename, mime_type=loaded.mime_type)
        mark("parse", t0)

        t0 = time.perf_counter()
        hierarchy = build_hierarchy(parsed)
        mark("hierarchy", t0)

        t0 = time.perf_counter()
        hierarchy, _chunks = self._chunking.chunk(hierarchy)
        mark("chunk", t0)

        t0 = time.perf_counter()
        object_key = f"documents/{document_id}/source/{loaded.filename}"
        await self._object_store.put(object_key, loaded.content, content_type=loaded.mime_type)
        mark("store", t0)

        t0 = time.perf_counter()
        if existing and inbound.strategy is IngestionStrategy.REPLACE:
            await self._nodes.delete_for_document(existing.id)
            record = DocumentRecord(
                id=existing.id,
                name=loaded.filename,
                mime_type=loaded.mime_type,
                content_hash=loaded.content_hash,
                status=DocumentStatus.PROCESSING,
                external_id=loaded.external_id or existing.external_id,
                byte_size=loaded.byte_size,
                parser_name=parsed.parser_name,
                parser_version=parsed.parser_version,
                object_key=object_key,
                metadata={**loaded.metadata, **parsed.metadata, "ingestion_id": ingestion_id},
            )
            await self._documents.update(record)
            document_id = existing.id
        else:
            record = DocumentRecord(
                id=document_id,
                name=loaded.filename,
                mime_type=loaded.mime_type,
                content_hash=loaded.content_hash,
                status=DocumentStatus.PROCESSING,
                external_id=loaded.external_id,
                byte_size=loaded.byte_size,
                parser_name=parsed.parser_name,
                parser_version=parsed.parser_version,
                object_key=object_key,
                metadata={**loaded.metadata, **parsed.metadata, "ingestion_id": ingestion_id},
            )
            await self._documents.create(record)

        persisted = self._materialize_nodes(document_id, hierarchy.nodes)
        persisted.sort(key=lambda n: (n.depth, n.ordinal))
        await self._nodes.bulk_create(persisted)

        record.status = DocumentStatus.READY
        await self._documents.update(record)
        mark("persist", t0)

        structural = sum(1 for n in persisted if n.node_type is not NodeType.CHUNK)
        chunk_count = sum(1 for n in persisted if n.node_type is NodeType.CHUNK)
        duration_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "ingestion_completed",
            ingestion_id=ingestion_id,
            document_id=str(document_id),
            mime_type=loaded.mime_type,
            byte_size=loaded.byte_size,
            structural_nodes=structural,
            chunks=chunk_count,
            duration_ms=round(duration_ms, 3),
            stage_durations_ms=stages,
            skipped=False,
        )

        return IngestionResult(
            document_id=document_id,
            content_hash=loaded.content_hash,
            status=DocumentStatus.READY.value,
            skipped=False,
            strategy=inbound.strategy,
            structural_node_count=structural,
            chunk_count=chunk_count,
            ingestion_id=ingestion_id,
            duration_ms=round(duration_ms, 3),
            stage_durations_ms=stages,
        )

    def _load(self, inbound: IngestionInput) -> LoadedDocument:
        return LoadedDocument(
            filename=inbound.filename,
            mime_type=inbound.mime_type or "application/octet-stream",
            content=inbound.content,
            content_hash=hash_bytes(inbound.content),
            byte_size=len(inbound.content),
            external_id=inbound.external_id,
            metadata=dict(inbound.metadata),
        )

    async def _resolve_existing(
        self,
        strategy: IngestionStrategy,
        loaded: LoadedDocument,
    ) -> DocumentRecord | None:
        if strategy is IngestionStrategy.CREATE_VERSION:
            return None
        if strategy is IngestionStrategy.SKIP_IDENTICAL:
            return await self._documents.get_by_content_hash(loaded.content_hash)
        if strategy is IngestionStrategy.REPLACE:
            if loaded.external_id:
                return await self._documents.get_by_external_id(loaded.external_id)
            return await self._documents.get_by_content_hash(loaded.content_hash)
        return None

    def _materialize_nodes(
        self,
        document_id: DocumentId,
        draft_nodes: list[HierarchyDraftNode],
    ) -> list[DocumentNode]:
        # Assign stable UUIDs (= temp_ids already UUIDs) and URIs / path_ids
        id_map: dict[UUID, NodeId] = {n.temp_id: NodeId(n.temp_id) for n in draft_nodes}
        by_temp: dict[UUID, HierarchyDraftNode] = {n.temp_id: n for n in draft_nodes}
        materialized: list[DocumentNode] = []

        for draft in draft_nodes:
            node_id = id_map[draft.temp_id]
            parent_id = id_map[draft.parent_temp_id] if draft.parent_temp_id else None
            path_ids = self._compute_path(draft.temp_id, by_temp, id_map)
            content = draft.text
            token_count = self._tokenizer.count(content) if content else None
            content_hash = hash_text(content or f"{draft.node_type}:{draft.title}:{draft.ordinal}")
            uri = (
                VikingURIParser.build_document_uri(document_id)
                if draft.node_type is NodeType.DOCUMENT
                else VikingURIParser.build_node_uri(document_id, node_id)
            )
            materialized.append(
                DocumentNode(
                    id=node_id,
                    document_id=document_id,
                    parent_id=parent_id,
                    node_type=draft.node_type,
                    title=draft.title,
                    ordinal=draft.ordinal,
                    depth=draft.depth,
                    uri=uri,
                    content=content,
                    content_hash=content_hash,
                    token_count=token_count,
                    path_ids=path_ids,
                    metadata=dict(draft.metadata),
                    abstract_status=AbstractStatus.NONE,
                )
            )
        return materialized

    def _compute_path(
        self,
        temp_id: UUID,
        by_temp: dict[UUID, HierarchyDraftNode],
        id_map: dict[UUID, NodeId],
    ) -> tuple[UUID, ...]:
        chain: list[UUID] = []
        current = temp_id
        seen: set[UUID] = set()
        while current is not None and current not in seen:
            seen.add(current)
            chain.append(UUID(str(id_map[current])))
            parent = by_temp[current].parent_temp_id
            if parent is None:
                break
            current = parent
        chain.reverse()
        return tuple(chain)
