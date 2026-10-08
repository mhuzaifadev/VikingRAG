"""Add learning_policy, snapshot_id, experience_snapshots.

Revision ID: 20261008_0005
Revises: 20261007_0004
Create Date: 2026-10-08
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "20261008_0005"
down_revision: str | None = "20261007_0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "query_runs",
        sa.Column(
            "learning_policy",
            sa.String(length=64),
            nullable=False,
            server_default="learn",
        ),
    )
    op.add_column(
        "query_runs",
        sa.Column("snapshot_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.create_index("ix_query_runs_learning_policy", "query_runs", ["learning_policy"])
    op.create_index("ix_query_runs_snapshot_id", "query_runs", ["snapshot_id"])

    op.create_table(
        "experience_snapshots",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column(
            "status",
            sa.String(length=64),
            nullable=False,
            server_default="active",
        ),
        sa.Column("corpus_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column(
            "edge_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "payload_ids",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
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
        sa.UniqueConstraint("name", name="uq_experience_snapshots_name"),
    )
    op.create_index("ix_experience_snapshots_status", "experience_snapshots", ["status"])


def downgrade() -> None:
    op.drop_index("ix_experience_snapshots_status", table_name="experience_snapshots")
    op.drop_table("experience_snapshots")
    op.drop_index("ix_query_runs_snapshot_id", table_name="query_runs")
    op.drop_index("ix_query_runs_learning_policy", table_name="query_runs")
    op.drop_column("query_runs", "snapshot_id")
    op.drop_column("query_runs", "learning_policy")
