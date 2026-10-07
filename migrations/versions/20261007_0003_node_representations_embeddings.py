"""Add node_representations and node_embeddings with pgvector HNSW.

Revision ID: 20261007_0003
Revises: 20261007_0002
Create Date: 2026-10-07

Default vector dimension is 1536 (text-embedding-3-small / common OpenAI-compatible).
Changing dimensions requires a new migration and re-index; rows store dimensions
and provider/model so incompatible spaces are never silently mixed.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "20261007_0003"
down_revision: str | None = "20261007_0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Keep in sync with VIKINGRAG_EMBEDDING_DIMENSIONS default / EmbeddingSettings.dimensions
VECTOR_DIMENSIONS = 1536


def upgrade() -> None:
    op.create_table(
        "node_representations",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "node_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_nodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("representation_type", sa.String(length=64), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("generator", sa.String(length=128), nullable=False),
        sa.Column("generator_model", sa.String(length=256), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False, server_default="1"),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "node_id",
            "representation_type",
            name="uq_node_representations_node_type",
        ),
    )
    op.create_index("ix_node_representations_document_id", "node_representations", ["document_id"])
    op.create_index("ix_node_representations_node_id", "node_representations", ["node_id"])
    op.create_index(
        "ix_node_representations_content_hash",
        "node_representations",
        ["content_hash"],
    )

    op.create_table(
        "node_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "representation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("node_representations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "node_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_nodes.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("provider", sa.String(length=128), nullable=False),
        sa.Column("model", sa.String(length=256), nullable=False),
        sa.Column("dimensions", sa.Integer(), nullable=False),
        sa.Column("identity_version", sa.String(length=64), nullable=False, server_default="1"),
        sa.Column("embedding", Vector(VECTOR_DIMENSIONS), nullable=False),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.UniqueConstraint(
            "representation_id",
            "provider",
            "model",
            "dimensions",
            "identity_version",
            name="uq_node_embeddings_identity",
        ),
    )
    op.create_index("ix_node_embeddings_document_id", "node_embeddings", ["document_id"])
    op.create_index("ix_node_embeddings_node_id", "node_embeddings", ["node_id"])
    op.create_index(
        "ix_node_embeddings_identity",
        "node_embeddings",
        ["provider", "model", "dimensions", "identity_version"],
    )
    op.execute(
        """
        CREATE INDEX ix_node_embeddings_hnsw_cosine
        ON node_embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_node_embeddings_hnsw_cosine")
    op.drop_table("node_embeddings")
    op.drop_table("node_representations")
