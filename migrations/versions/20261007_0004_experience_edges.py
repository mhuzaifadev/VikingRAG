"""Add query_runs, retrieval_events, experience_payloads, experience_edges.

Revision ID: 20261007_0004
Revises: 20261007_0003
Create Date: 2026-10-07

Experience edges are query-conditioned retrieval shortcuts (Alg 2/3), not
generic section adjacency. Incoming/outgoing URI indexes support invalidation
when endpoints are deleted or replaced. Query embedding identity is stored so
incompatible spaces are never compared at activation time.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "20261007_0004"
down_revision: str | None = "20261007_0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Keep in sync with DEFAULT_VECTOR_DIMENSIONS / EmbeddingSettings.dimensions
VECTOR_DIMENSIONS = 1536


def upgrade() -> None:
    op.create_table(
        "query_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column("tenant_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("corpus_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_embedding", Vector(VECTOR_DIMENSIONS), nullable=True),
        sa.Column("embedding_provider", sa.String(length=128), nullable=True),
        sa.Column("embedding_model", sa.String(length=256), nullable=True),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=True),
        sa.Column(
            "embedding_identity_version",
            sa.String(length=64),
            nullable=True,
            server_default="1",
        ),
        sa.Column("route", sa.String(length=64), nullable=True),
        sa.Column("answer", sa.Text(), nullable=True),
        sa.Column("sufficiency_score", sa.Float(), nullable=True),
        sa.Column(
            "status",
            sa.String(length=64),
            nullable=False,
            server_default="pending",
        ),
        sa.Column(
            "edge_build_status",
            sa.String(length=64),
            nullable=False,
            server_default="none",
        ),
        sa.Column("edge_build_error", sa.Text(), nullable=True),
        sa.Column("total_input_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_output_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retrieval_tokens", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("llm_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retrieval_rounds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "document_ids",
            postgresql.ARRAY(postgresql.UUID(as_uuid=True)),
            nullable=False,
            server_default=sa.text("'{}'::uuid[]"),
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
    )
    op.create_index("ix_query_runs_status", "query_runs", ["status"])
    op.create_index("ix_query_runs_edge_build_status", "query_runs", ["edge_build_status"])
    op.create_index("ix_query_runs_corpus_id", "query_runs", ["corpus_id"])
    op.create_index("ix_query_runs_created_at", "query_runs", ["created_at"])

    op.create_table(
        "retrieval_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "query_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("round_no", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("event_type", sa.String(length=64), nullable=False),
        sa.Column(
            "arguments",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "result_refs",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column("token_cost", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_retrieval_events_query_run_id", "retrieval_events", ["query_run_id"])
    op.create_index(
        "ix_retrieval_events_run_type",
        "retrieval_events",
        ["query_run_id", "event_type"],
    )

    op.create_table(
        "experience_payloads",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "query_run_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("query_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("query_embedding", Vector(VECTOR_DIMENSIONS), nullable=False),
        sa.Column("embedding_provider", sa.String(length=128), nullable=False),
        sa.Column("embedding_model", sa.String(length=256), nullable=False),
        sa.Column("embedding_dimensions", sa.Integer(), nullable=False),
        sa.Column(
            "embedding_identity_version",
            sa.String(length=64),
            nullable=False,
            server_default="1",
        ),
        sa.Column("trace_summary", sa.Text(), nullable=False),
        sa.Column(
            "support_uris",
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
        sa.UniqueConstraint("query_run_id", name="uq_experience_payloads_query_run_id"),
    )
    op.create_index(
        "ix_experience_payloads_embedding_identity",
        "experience_payloads",
        [
            "embedding_provider",
            "embedding_model",
            "embedding_dimensions",
            "embedding_identity_version",
        ],
    )

    op.create_table(
        "experience_edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True, nullable=False),
        sa.Column(
            "payload_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("experience_payloads.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_uri", sa.String(length=512), nullable=False),
        sa.Column("target_uri", sa.String(length=512), nullable=False),
        sa.Column(
            "status",
            sa.String(length=64),
            nullable=False,
            server_default="active",
        ),
        sa.Column("source_node_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("target_node_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_revision", sa.String(length=128), nullable=True),
        sa.Column("target_revision", sa.String(length=128), nullable=True),
        sa.Column("support_score", sa.Float(), nullable=False, server_default="1.0"),
        sa.Column("success_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_used_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
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
        sa.CheckConstraint("source_uri <> target_uri", name="ck_experience_edges_no_self"),
    )
    # Outgoing: edges leaving a source URI (activation / Search+)
    op.create_index("ix_experience_edges_source_uri", "experience_edges", ["source_uri"])
    # Incoming: edges pointing at a target URI (invalidation on delete/replace)
    op.create_index("ix_experience_edges_target_uri", "experience_edges", ["target_uri"])
    op.create_index("ix_experience_edges_status", "experience_edges", ["status"])
    op.create_index("ix_experience_edges_payload_id", "experience_edges", ["payload_id"])
    op.create_index(
        "ix_experience_edges_active_pair",
        "experience_edges",
        ["source_uri", "target_uri", "status"],
    )


def downgrade() -> None:
    op.drop_index("ix_experience_edges_active_pair", table_name="experience_edges")
    op.drop_index("ix_experience_edges_payload_id", table_name="experience_edges")
    op.drop_index("ix_experience_edges_status", table_name="experience_edges")
    op.drop_index("ix_experience_edges_target_uri", table_name="experience_edges")
    op.drop_index("ix_experience_edges_source_uri", table_name="experience_edges")
    op.drop_table("experience_edges")

    op.drop_index(
        "ix_experience_payloads_embedding_identity",
        table_name="experience_payloads",
    )
    op.drop_table("experience_payloads")

    op.drop_index("ix_retrieval_events_run_type", table_name="retrieval_events")
    op.drop_index("ix_retrieval_events_query_run_id", table_name="retrieval_events")
    op.drop_table("retrieval_events")

    op.drop_index("ix_query_runs_created_at", table_name="query_runs")
    op.drop_index("ix_query_runs_corpus_id", table_name="query_runs")
    op.drop_index("ix_query_runs_edge_build_status", table_name="query_runs")
    op.drop_index("ix_query_runs_status", table_name="query_runs")
    op.drop_table("query_runs")
