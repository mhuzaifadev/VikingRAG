from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.domain.models.document import DocumentId, DocumentStatus
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.base import DocumentRecord
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_document_repository_crud(database: Database) -> None:
    async with database.session_factory() as session:
        repo = SqlDocumentRepository(session)
        doc_id = DocumentId(uuid4())
        created = await repo.create(
            DocumentRecord(
                id=doc_id,
                name="readme.md",
                mime_type="text/markdown",
                content_hash="abc123",
                status=DocumentStatus.PENDING,
                external_id="ext-1",
                metadata={"source": "test"},
            )
        )
        await session.commit()

        assert created.id == doc_id
        assert created.name == "readme.md"
        assert created.status == DocumentStatus.PENDING

    async with database.session_factory() as session:
        repo = SqlDocumentRepository(session)
        fetched = await repo.get(doc_id)
        assert fetched is not None
        assert fetched.content_hash == "abc123"
        assert fetched.metadata["source"] == "test"

        listed = await repo.list(limit=10)
        assert any(item.id == doc_id for item in listed)

        deleted = await repo.delete(doc_id)
        await session.commit()
        assert deleted is True

        assert await repo.get(doc_id) is None


@pytest.mark.asyncio
async def test_pgvector_extension_enabled(database: Database) -> None:
    from sqlalchemy import text

    async with database.engine.connect() as conn:
        result = await conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        )
        row = result.first()
        assert row is not None
        assert row[0] == "vector"
