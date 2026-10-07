"""SQLAlchemy async document repository."""

from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from vikingrag.domain.models.document import DocumentId, DocumentStatus
from vikingrag.infrastructure.database.models import DocumentRow
from vikingrag.infrastructure.database.repositories.base import DocumentRecord


def _to_record(row: DocumentRow) -> DocumentRecord:
    return DocumentRecord(
        id=DocumentId(row.id if isinstance(row.id, UUID) else UUID(str(row.id))),
        external_id=row.external_id,
        name=row.name,
        mime_type=row.mime_type,
        content_hash=row.content_hash,
        status=DocumentStatus(row.status),
        byte_size=row.byte_size,
        parser_name=row.parser_name,
        parser_version=row.parser_version,
        object_key=row.object_key,
        metadata=dict(row.metadata_ or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


class SqlDocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(self, record: DocumentRecord) -> DocumentRecord:
        row = DocumentRow(
            id=record.id,
            external_id=record.external_id,
            name=record.name,
            mime_type=record.mime_type,
            content_hash=record.content_hash,
            status=record.status.value,
            byte_size=record.byte_size,
            parser_name=record.parser_name,
            parser_version=record.parser_version,
            object_key=record.object_key,
            metadata_=dict(record.metadata),
        )
        self._session.add(row)
        await self._session.flush()
        await self._session.refresh(row)
        return _to_record(row)

    async def get(self, document_id: DocumentId) -> DocumentRecord | None:
        row = await self._session.get(DocumentRow, document_id)
        return _to_record(row) if row else None

    async def get_by_content_hash(self, content_hash: str) -> DocumentRecord | None:
        stmt = (
            select(DocumentRow)
            .where(DocumentRow.content_hash == content_hash)
            .where(DocumentRow.status != DocumentStatus.DELETED.value)
            .order_by(DocumentRow.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_record(row) if row else None

    async def get_by_external_id(self, external_id: str) -> DocumentRecord | None:
        stmt = (
            select(DocumentRow)
            .where(DocumentRow.external_id == external_id)
            .where(DocumentRow.status != DocumentStatus.DELETED.value)
            .order_by(DocumentRow.created_at.desc())
            .limit(1)
        )
        result = await self._session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_record(row) if row else None

    async def update(self, record: DocumentRecord) -> DocumentRecord:
        row = await self._session.get(DocumentRow, record.id)
        if row is None:
            raise ValueError(f"Document not found: {record.id}")
        row.name = record.name
        row.mime_type = record.mime_type
        row.content_hash = record.content_hash
        row.status = record.status.value
        row.external_id = record.external_id
        row.byte_size = record.byte_size
        row.parser_name = record.parser_name
        row.parser_version = record.parser_version
        row.object_key = record.object_key
        row.metadata_ = dict(record.metadata)
        await self._session.flush()
        await self._session.refresh(row)
        return _to_record(row)

    async def list(self, *, limit: int = 50, offset: int = 0) -> list[DocumentRecord]:
        stmt = (
            select(DocumentRow).order_by(DocumentRow.created_at.desc()).limit(limit).offset(offset)
        )
        result = await self._session.execute(stmt)
        return [_to_record(row) for row in result.scalars().all()]

    async def delete(self, document_id: DocumentId) -> bool:
        row = await self._session.get(DocumentRow, document_id)
        if row is None:
            return False
        await self._session.delete(row)
        await self._session.flush()
        return True
