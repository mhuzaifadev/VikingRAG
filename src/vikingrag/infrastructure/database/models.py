"""SQLAlchemy ORM models - keep domain models free of ORM types."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import uuid4

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Default pgvector column width - must match migrations 20261007_0003 / 0004
DEFAULT_VECTOR_DIMENSIONS = 1536


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    external_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    name: Mapped[str] = mapped_column(String(512), nullable=False)
    mime_type: Mapped[str] = mapped_column(String(255), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="pending")
    byte_size: Mapped[int | None] = mapped_column(Integer, nullable=True)
    parser_name: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    object_key: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class DocumentNodeRow(Base):
    __tablename__ = "document_nodes"

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    document_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_id: Mapped[Any | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    node_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    uri: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    content: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    path_ids: Mapped[list[Any]] = mapped_column(
        ARRAY(UUID(as_uuid=True)),
        nullable=False,
        default=list,
    )
    # Mirror of NODE_SUMMARY representation for navigation convenience
    abstract_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    abstract_status: Mapped[str] = mapped_column(String(32), nullable=False, default="none")
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class NodeRepresentationRow(Base):
    __tablename__ = "node_representations"
    __table_args__ = (
        UniqueConstraint(
            "node_id", "representation_type", name="uq_node_representations_node_type"
        ),
    )

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    node_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    representation_type: Mapped[str] = mapped_column(String(64), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    generator: Mapped[str] = mapped_column(String(128), nullable=False)
    generator_model: Mapped[str] = mapped_column(String(256), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class NodeEmbeddingRow(Base):
    __tablename__ = "node_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "representation_id",
            "provider",
            "model",
            "dimensions",
            "identity_version",
            name="uq_node_embeddings_identity",
        ),
    )

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    representation_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("node_representations.id", ondelete="CASCADE"),
        nullable=False,
    )
    node_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("document_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    provider: Mapped[str] = mapped_column(String(128), nullable=False)
    model: Mapped[str] = mapped_column(String(256), nullable=False)
    dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    identity_version: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    embedding: Mapped[Any] = mapped_column(Vector(DEFAULT_VECTOR_DIMENSIONS), nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class QueryRunRow(Base):
    __tablename__ = "query_runs"

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    tenant_id: Mapped[Any | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    corpus_id: Mapped[Any | None] = mapped_column(UUID(as_uuid=True), nullable=True, index=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_embedding: Mapped[Any | None] = mapped_column(
        Vector(DEFAULT_VECTOR_DIMENSIONS), nullable=True
    )
    embedding_provider: Mapped[str | None] = mapped_column(String(128), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(256), nullable=True)
    embedding_dimensions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_identity_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    route: Mapped[str | None] = mapped_column(String(64), nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    sufficiency_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="pending", index=True)
    edge_build_status: Mapped[str] = mapped_column(
        String(64), nullable=False, default="none", index=True
    )
    edge_build_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    total_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retrieval_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retrieval_rounds: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    document_ids: Mapped[list[Any]] = mapped_column(
        ARRAY(UUID(as_uuid=True)), nullable=False, default=list
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class RetrievalEventRow(Base):
    __tablename__ = "retrieval_events"

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_run_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    round_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    arguments: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    result_refs: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_cost: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ExperiencePayloadRow(Base):
    __tablename__ = "experience_payloads"
    __table_args__ = (UniqueConstraint("query_run_id", name="uq_experience_payloads_query_run_id"),)

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    query_run_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("query_runs.id", ondelete="CASCADE"),
        nullable=False,
    )
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_embedding: Mapped[Any] = mapped_column(Vector(DEFAULT_VECTOR_DIMENSIONS), nullable=False)
    embedding_provider: Mapped[str] = mapped_column(String(128), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(256), nullable=False)
    embedding_dimensions: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_identity_version: Mapped[str] = mapped_column(String(64), nullable=False, default="1")
    trace_summary: Mapped[str] = mapped_column(Text, nullable=False)
    support_uris: Mapped[list[Any]] = mapped_column(JSONB, nullable=False, default=list)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ExperienceEdgeRow(Base):
    __tablename__ = "experience_edges"
    __table_args__ = (
        CheckConstraint("source_uri <> target_uri", name="ck_experience_edges_no_self"),
    )

    id: Mapped[Any] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    payload_id: Mapped[Any] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("experience_payloads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_uri: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    target_uri: Mapped[str] = mapped_column(String(512), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(64), nullable=False, default="active", index=True)
    source_node_id: Mapped[Any | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    target_node_id: Mapped[Any | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    source_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    target_revision: Mapped[str | None] = mapped_column(String(128), nullable=True)
    support_score: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    success_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata", JSONB, nullable=False, default=dict
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
