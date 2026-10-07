"""Persistence for node semantic representations."""

from __future__ import annotations

from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.representation import (
    NodeRepresentation,
    RepresentationId,
    RepresentationType,
)
from vikingrag.infrastructure.database.models import NodeRepresentationRow


def _to_domain(row: NodeRepresentationRow) -> NodeRepresentation:
    return NodeRepresentation(
        id=RepresentationId(row.id if isinstance(row.id, UUID) else UUID(str(row.id))),
        node_id=NodeId(row.node_id if isinstance(row.node_id, UUID) else UUID(str(row.node_id))),
        document_id=DocumentId(
            row.document_id if isinstance(row.document_id, UUID) else UUID(str(row.document_id))
        ),
        representation_type=RepresentationType(row.representation_type),
        content=row.content,
        content_hash=row.content_hash,
        generator=row.generator,
        generator_model=row.generator_model,
        version=row.version,
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SqlRepresentationRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(
        self,
        node_id: NodeId,
        representation_type: RepresentationType,
    ) -> NodeRepresentation | None:
        stmt = (
            select(NodeRepresentationRow)
            .where(NodeRepresentationRow.node_id == node_id)
            .where(NodeRepresentationRow.representation_type == representation_type.value)
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_domain(row) if row else None

    async def get_by_id(self, representation_id: RepresentationId) -> NodeRepresentation | None:
        row = await self._session.get(NodeRepresentationRow, representation_id)
        return _to_domain(row) if row else None

    async def list_for_document(self, document_id: DocumentId) -> list[NodeRepresentation]:
        stmt = select(NodeRepresentationRow).where(NodeRepresentationRow.document_id == document_id)
        result = await self._session.execute(stmt)
        return [_to_domain(row) for row in result.scalars().all()]

    async def upsert(self, representation: NodeRepresentation) -> NodeRepresentation:
        stmt = (
            insert(NodeRepresentationRow)
            .values(
                id=representation.id,
                node_id=representation.node_id,
                document_id=representation.document_id,
                representation_type=representation.representation_type.value,
                content=representation.content,
                content_hash=representation.content_hash,
                generator=representation.generator,
                generator_model=representation.generator_model,
                version=representation.version,
                metadata_=dict(representation.metadata),
            )
            .on_conflict_do_update(
                constraint="uq_node_representations_node_type",
                set_={
                    "content": representation.content,
                    "content_hash": representation.content_hash,
                    "generator": representation.generator,
                    "generator_model": representation.generator_model,
                    "version": representation.version,
                    "metadata": dict(representation.metadata),
                },
            )
            .returning(NodeRepresentationRow)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one()
        await self._session.flush()
        return _to_domain(row)

    async def count_for_document(self, document_id: DocumentId) -> int:
        from sqlalchemy import func

        stmt = (
            select(func.count())
            .select_from(NodeRepresentationRow)
            .where(NodeRepresentationRow.document_id == document_id)
        )
        result = await self._session.execute(stmt)
        return int(result.scalar_one())


def new_representation_id() -> RepresentationId:
    return RepresentationId(uuid4())
