"""pgvector persistence and cosine similarity search for node embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import Select, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from vikingrag.domain.errors import EmbeddingIdentityMismatch
from vikingrag.domain.models.document import DocumentId, NodeId, NodeType
from vikingrag.domain.models.representation import (
    EmbeddingId,
    EmbeddingIdentity,
    NodeEmbedding,
    RepresentationId,
    RepresentationType,
)
from vikingrag.infrastructure.database.models import (
    DEFAULT_VECTOR_DIMENSIONS,
    DocumentNodeRow,
    NodeEmbeddingRow,
    NodeRepresentationRow,
)


@dataclass(frozen=True, slots=True)
class VectorSearchHit:
    embedding_id: EmbeddingId
    representation_id: RepresentationId
    node_id: NodeId
    document_id: DocumentId
    node_type: NodeType
    representation_type: RepresentationType
    uri: str
    title: str | None
    preview: str
    similarity: float
    provider: str
    model: str
    dimensions: int


def _to_domain(row: NodeEmbeddingRow) -> NodeEmbedding:
    identity = EmbeddingIdentity(
        provider=row.provider,
        model=row.model,
        dimensions=row.dimensions,
        version=row.identity_version,
    )
    vector = list(row.embedding) if row.embedding is not None else []
    return NodeEmbedding(
        id=EmbeddingId(row.id if isinstance(row.id, UUID) else UUID(str(row.id))),
        representation_id=RepresentationId(
            row.representation_id
            if isinstance(row.representation_id, UUID)
            else UUID(str(row.representation_id))
        ),
        node_id=NodeId(row.node_id if isinstance(row.node_id, UUID) else UUID(str(row.node_id))),
        document_id=DocumentId(
            row.document_id if isinstance(row.document_id, UUID) else UUID(str(row.document_id))
        ),
        identity=identity,
        embedding=vector,
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
    )


class SqlEmbeddingRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_for_representation(
        self,
        representation_id: RepresentationId,
        identity: EmbeddingIdentity,
    ) -> NodeEmbedding | None:
        stmt = (
            select(NodeEmbeddingRow)
            .where(NodeEmbeddingRow.representation_id == representation_id)
            .where(NodeEmbeddingRow.provider == identity.provider)
            .where(NodeEmbeddingRow.model == identity.model)
            .where(NodeEmbeddingRow.dimensions == identity.dimensions)
            .where(NodeEmbeddingRow.identity_version == identity.version)
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_domain(row) if row else None

    async def upsert(self, embedding: NodeEmbedding) -> NodeEmbedding:
        if embedding.identity.dimensions != DEFAULT_VECTOR_DIMENSIONS:
            raise EmbeddingIdentityMismatch(
                f"Persisted vector column expects dimensions={DEFAULT_VECTOR_DIMENSIONS}, "
                f"got {embedding.identity.dimensions}"
            )
        stmt = (
            insert(NodeEmbeddingRow)
            .values(
                id=embedding.id,
                representation_id=embedding.representation_id,
                node_id=embedding.node_id,
                document_id=embedding.document_id,
                provider=embedding.identity.provider,
                model=embedding.identity.model,
                dimensions=embedding.identity.dimensions,
                identity_version=embedding.identity.version,
                embedding=embedding.embedding,
                metadata_=dict(embedding.metadata),
            )
            .on_conflict_do_update(
                constraint="uq_node_embeddings_identity",
                set_={
                    "embedding": embedding.embedding,
                    "metadata": dict(embedding.metadata),
                },
            )
            .returning(NodeEmbeddingRow)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one()
        await self._session.flush()
        return _to_domain(row)

    async def count_for_document(
        self,
        document_id: DocumentId,
        *,
        identity: EmbeddingIdentity | None = None,
    ) -> int:
        stmt = (
            select(func.count())
            .select_from(NodeEmbeddingRow)
            .where(NodeEmbeddingRow.document_id == document_id)
        )
        if identity is not None:
            stmt = (
                stmt.where(NodeEmbeddingRow.provider == identity.provider)
                .where(NodeEmbeddingRow.model == identity.model)
                .where(NodeEmbeddingRow.dimensions == identity.dimensions)
                .where(NodeEmbeddingRow.identity_version == identity.version)
            )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())

    async def search_cosine(
        self,
        query_embedding: list[float],
        *,
        identity: EmbeddingIdentity,
        top_k: int,
        document_ids: tuple[DocumentId, ...] = (),
        node_types: tuple[NodeType, ...] = (),
        representation_types: tuple[RepresentationType, ...] = (),
        min_score: float | None = None,
    ) -> list[VectorSearchHit]:
        if len(query_embedding) != identity.dimensions:
            raise EmbeddingIdentityMismatch(
                f"query embedding length {len(query_embedding)} != "
                f"identity.dimensions {identity.dimensions}"
            )
        if identity.dimensions != DEFAULT_VECTOR_DIMENSIONS:
            raise EmbeddingIdentityMismatch(
                f"Search expects dimensions={DEFAULT_VECTOR_DIMENSIONS}, got {identity.dimensions}"
            )

        # cosine distance <=> ; similarity = 1 - distance
        distance = NodeEmbeddingRow.embedding.cosine_distance(query_embedding)
        similarity_expr = (1 - distance).label("similarity")

        stmt: Select[Any] = (
            select(
                NodeEmbeddingRow.id,
                NodeEmbeddingRow.representation_id,
                NodeEmbeddingRow.node_id,
                NodeEmbeddingRow.document_id,
                NodeEmbeddingRow.provider,
                NodeEmbeddingRow.model,
                NodeEmbeddingRow.dimensions,
                DocumentNodeRow.node_type,
                DocumentNodeRow.uri,
                DocumentNodeRow.title,
                NodeRepresentationRow.representation_type,
                NodeRepresentationRow.content,
                similarity_expr,
            )
            .join(
                DocumentNodeRow,
                DocumentNodeRow.id == NodeEmbeddingRow.node_id,
            )
            .join(
                NodeRepresentationRow,
                NodeRepresentationRow.id == NodeEmbeddingRow.representation_id,
            )
            .where(NodeEmbeddingRow.provider == identity.provider)
            .where(NodeEmbeddingRow.model == identity.model)
            .where(NodeEmbeddingRow.dimensions == identity.dimensions)
            .where(NodeEmbeddingRow.identity_version == identity.version)
            .order_by(distance.asc())
            .limit(top_k)
        )
        if document_ids:
            stmt = stmt.where(NodeEmbeddingRow.document_id.in_(list(document_ids)))
        if node_types:
            stmt = stmt.where(DocumentNodeRow.node_type.in_([t.value for t in node_types]))
        if representation_types:
            stmt = stmt.where(
                NodeRepresentationRow.representation_type.in_(
                    [t.value for t in representation_types]
                )
            )
        if min_score is not None:
            stmt = stmt.where(similarity_expr >= min_score)

        result = await self._session.execute(stmt)
        hits: list[VectorSearchHit] = []
        for row in result.all():
            content = str(row.content or "")
            preview = content if len(content) <= 240 else content[:237] + "..."
            hits.append(
                VectorSearchHit(
                    embedding_id=EmbeddingId(UUID(str(row.id))),
                    representation_id=RepresentationId(UUID(str(row.representation_id))),
                    node_id=NodeId(UUID(str(row.node_id))),
                    document_id=DocumentId(UUID(str(row.document_id))),
                    node_type=NodeType(str(row.node_type)),
                    representation_type=RepresentationType(str(row.representation_type)),
                    uri=str(row.uri),
                    title=row.title,
                    preview=preview,
                    similarity=float(row.similarity),
                    provider=str(row.provider),
                    model=str(row.model),
                    dimensions=int(row.dimensions),
                )
            )
        return hits


def new_embedding_id() -> EmbeddingId:
    return EmbeddingId(uuid4())


async def ensure_vector_extension(session: AsyncSession) -> None:
    await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
