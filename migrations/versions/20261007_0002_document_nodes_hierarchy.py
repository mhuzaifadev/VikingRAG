"""Add document_nodes hierarchy and document ingestion columns.

Revision ID: 20261007_0002
Revises: 20260326_0001
Create Date: 2026-10-07

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261007_0002"
down_revision: str | None = "20260326_0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("byte_size", sa.Integer(), nullable=True))
    op.add_column("documents", sa.Column("parser_name", sa.String(length=64), nullable=True))
    op.add_column("documents", sa.Column("parser_version", sa.String(length=32), nullable=True))
    op.add_column("documents", sa.Column("object_key", sa.String(length=1024), nullable=True))

    op.create_table(
        "document_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "parent_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("document_nodes.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("node_type", sa.String(length=64), nullable=False),
        sa.Column("title", sa.String(length=1024), nullable=True),
        sa.Column("ordinal", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("depth", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("uri", sa.String(length=512), nullable=False),
        sa.Column("content", sa.Text(), nullable=True),
        sa.Column("content_hash", sa.String(length=128), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column(
            "path_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
        ),
        sa.Column("abstract_text", sa.Text(), nullable=True),
        sa.Column(
            "abstract_status",
            sa.String(length=32),
            nullable=False,
            server_default="none",
        ),
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
        sa.UniqueConstraint("uri", name="uq_document_nodes_uri"),
        sa.UniqueConstraint(
            "document_id",
            "parent_id",
            "ordinal",
            name="uq_document_nodes_sibling_ordinal",
        ),
    )
    op.create_index("ix_document_nodes_document_id", "document_nodes", ["document_id"])
    op.create_index("ix_document_nodes_parent_id", "document_nodes", ["parent_id"])
    op.create_index("ix_document_nodes_uri", "document_nodes", ["uri"])
    op.create_index(
        "ix_document_nodes_document_parent",
        "document_nodes",
        ["document_id", "parent_id"],
    )
    op.create_index(
        "ix_document_nodes_path_ids",
        "document_nodes",
        ["path_ids"],
        postgresql_using="gin",
    )


def downgrade() -> None:
    op.drop_index("ix_document_nodes_path_ids", table_name="document_nodes")
    op.drop_index("ix_document_nodes_document_parent", table_name="document_nodes")
    op.drop_index("ix_document_nodes_uri", table_name="document_nodes")
    op.drop_index("ix_document_nodes_parent_id", table_name="document_nodes")
    op.drop_index("ix_document_nodes_document_id", table_name="document_nodes")
    op.drop_table("document_nodes")
    op.drop_column("documents", "object_key")
    op.drop_column("documents", "parser_version")
    op.drop_column("documents", "parser_name")
    op.drop_column("documents", "byte_size")
