"""Typed application configuration from environment / .env."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vikingrag.domain.models.document import DocumentId


class ObjectStoreBackend(StrEnum):
    LOCAL = "local"
    S3 = "s3"


_ENV_FILE = ".env"


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_APP_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    name: str = "vikingrag"
    env: Literal["development", "test", "staging", "production"] = "development"
    # migrate | api | worker — production provider checks are role-scoped
    process_role: Literal["api", "migrate", "worker"] = "api"
    debug: bool = False
    log_level: str = "INFO"
    host: str = "0.0.0.0"
    port: int = 8000


class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_DATABASE_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = "postgresql+asyncpg://vikingrag:vikingrag@localhost:5432/vikingrag"
    pool_size: int = 5
    max_overflow: int = 10
    echo: bool = False


class RedisSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_REDIS_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str = "redis://localhost:6379/0"


class ObjectStoreSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_OBJECT_STORE_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    backend: ObjectStoreBackend = ObjectStoreBackend.LOCAL
    local_root: str = "./data/object_store"
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_bucket: str = "vikingrag"
    s3_region: str = "us-east-1"


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_LLM_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = "unimplemented"
    # Empty → factory uses provider preset default_model
    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    timeout_seconds: float = 30.0
    max_retries: int = 2
    temperature: float = 0.2


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_EMBEDDING_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = "unimplemented"
    # Empty → factory uses provider preset default_embedding_model
    model: str = ""
    base_url: str = "https://api.openai.com/v1"
    api_key: str | None = None
    dimensions: int = 1536
    identity_version: str = "1"
    timeout_seconds: float = 30.0
    max_retries: int = 2
    batch_size: int = 32

    @field_validator("dimensions", "batch_size")
    @classmethod
    def _positive_embed(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_AUTH_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = False
    api_key: str | None = None
    # Comma-separated UUIDs; unset = unrestricted when auth enabled;
    # empty string = allow-nothing
    allowed_document_ids: str | None = None

    def permitted_document_ids(self) -> frozenset[DocumentId] | None:
        """None unrestricted; empty frozenset deny-all; else allowlist.

        When auth is disabled, always unrestricted.
        """
        return derive_permitted_document_ids(self)


def parse_allowed_document_ids(raw: str | None) -> frozenset[DocumentId] | None:
    """Map allowlist config to permitted set.

    - ``None`` (unset) → unrestricted (``None``)
    - ``*`` / ``all`` / ``unrestricted`` → unrestricted
    - empty / whitespace → deny-all (``frozenset()``)
    - comma-separated UUIDs → frozenset of DocumentId
    """
    if raw is None:
        return None
    stripped = raw.strip()
    if stripped == "":
        return frozenset()
    if stripped.lower() in {"*", "all", "unrestricted"}:
        return None
    ids: list[DocumentId] = []
    for part in stripped.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            ids.append(DocumentId(UUID(token)))
        except ValueError as exc:
            raise ValueError(
                f"Invalid UUID in VIKINGRAG_AUTH_ALLOWED_DOCUMENT_IDS: {token!r}"
            ) from exc
    return frozenset(ids)


def derive_permitted_document_ids(auth: AuthSettings) -> frozenset[DocumentId] | None:
    if not auth.enabled:
        return None
    return parse_allowed_document_ids(auth.allowed_document_ids)


class PaperProfileSettings(BaseSettings):
    """Selectable paper-evaluation budgets (arXiv 2609.11390)."""

    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_PAPER_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    enabled: bool = False
    top_k: int = 10  # K
    chunk_token_upper_bound: int = 1000  # L
    agent_round_budget: int = 15  # B
    activation_gamma: float = 0.8  # paper gamma


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_RETRIEVAL_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    initial_top_k: int = 8
    candidate_pool_size: int = 30
    max_rounds: int = 6
    max_tool_calls: int = 12
    max_read_tokens: int = 12_000
    max_wall_time_ms: int = 12_000
    weight_document: float = 0.85
    weight_section: float = 1.0
    weight_subsection: float = 1.0
    weight_chunk: float = 1.0
    rerank_enabled: bool = False
    min_score: float | None = None
    # Evidence / primitive caps (server overrides larger client values)
    max_list_limit: int = 100
    max_grep_matches: int = 50
    max_grep_descendants: int = 1000
    max_read_tokens_per_call: int = 1000
    max_evidence_candidates: int = 12
    max_descent_depth: int = 2
    max_bundle_tokens: int = 8_000
    min_assessment_coverage: float = 1.0
    assessor_provider: str = "scripted"
    support_selector: Literal["llm", "deterministic"] = "deterministic"
    experience_gamma: float = 0.8
    experience_max_hops: int = 2
    experience_max_nodes: int = 32
    experience_max_edges: int = 64
    experience_max_support: int = 16
    finalization_llm_reserve: int = 1

    @field_validator(
        "initial_top_k",
        "candidate_pool_size",
        "max_rounds",
        "max_tool_calls",
        "max_read_tokens",
        "max_wall_time_ms",
        "max_list_limit",
        "max_grep_matches",
        "max_grep_descendants",
        "max_read_tokens_per_call",
        "max_evidence_candidates",
        "max_bundle_tokens",
        "experience_max_hops",
        "experience_max_nodes",
        "experience_max_edges",
        "experience_max_support",
    )
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value

    @field_validator("experience_gamma")
    @classmethod
    def _gamma(cls, value: float) -> float:
        if not (0.0 <= value <= 1.0):
            raise ValueError("experience_gamma must be in [0, 1]")
        return value


class IndexingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_INDEXING_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    summary_concurrency: int = 4
    summary_max_input_tokens: int = 2_000
    summary_max_output_tokens: int = 256
    summary_version: str = "1"
    preview_chars: int = 240

    @field_validator(
        "summary_concurrency",
        "summary_max_input_tokens",
        "summary_max_output_tokens",
        "preview_chars",
    )
    @classmethod
    def _positive_index(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


class WorkerSettings(BaseSettings):
    """Background workers started from API lifespan (or as sidecar)."""

    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_WORKERS_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    edge_builder_enabled: bool = True
    edge_builder_poll_seconds: float = 2.0
    edge_builder_batch_size: int = 5


class IngestionSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_INGESTION_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_strategy: Literal["skip_identical", "replace", "create_version"] = "skip_identical"
    chunk_target_tokens: int = 650
    chunk_max_tokens: int = 900
    chunk_overlap_tokens: int = 80
    max_upload_bytes: int = 20_000_000

    @field_validator(
        "chunk_target_tokens", "chunk_max_tokens", "chunk_overlap_tokens", "max_upload_bytes"
    )
    @classmethod
    def _positive_ingest(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
        return value


class Settings(BaseSettings):
    """Root settings aggregating nested groups.

    Nested models load from their own env prefixes. Top-level also accepts
    a dotenv file for local development.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app: AppSettings = Field(default_factory=AppSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    object_store: ObjectStoreSettings = Field(default_factory=ObjectStoreSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    paper: PaperProfileSettings = Field(default_factory=PaperProfileSettings)
    workers: WorkerSettings = Field(default_factory=WorkerSettings)

    @model_validator(mode="after")
    def _pool_vs_top_k(self) -> Settings:
        if self.retrieval.candidate_pool_size < self.retrieval.initial_top_k:
            raise ValueError("retrieval.candidate_pool_size must be >= initial_top_k")
        # Tests never auto-start the edge builder (prevents background DB polls).
        if self.app.env == "test":
            self.workers.edge_builder_enabled = False
        if self.paper.enabled:
            self.retrieval.initial_top_k = self.paper.top_k
            self.retrieval.candidate_pool_size = max(
                self.retrieval.candidate_pool_size, self.paper.top_k * 3
            )
            self.retrieval.max_rounds = self.paper.agent_round_budget
            self.retrieval.max_read_tokens_per_call = self.paper.chunk_token_upper_bound
            self.retrieval.experience_gamma = self.paper.activation_gamma
            # Accommodate B=15 rounds: tools + LLM + wall clock
            self.retrieval.max_tool_calls = max(self.retrieval.max_tool_calls, 60)
            self.retrieval.max_wall_time_ms = max(self.retrieval.max_wall_time_ms, 300_000)
        if self.app.env == "production":
            _reject_fake_providers_for_role(self)
            if self.app.process_role == "api" and self.auth.enabled and not self.auth.api_key:
                raise ValueError("production auth requires VIKINGRAG_AUTH_API_KEY")
        return self


_FAKE_PROVIDERS = frozenset({"fake", "test", "deterministic", "scripted", "unimplemented"})


def _reject_fake_providers_for_role(settings: Settings) -> None:
    """Production provider checks depend on process role.

    - migrate: database only (no LLM/embedding/assessor required)
    - worker: LLM required; embedding + assessor when SUPPORT=llm
    - api: full stack (LLM, embedding, assessor)
    """
    role = settings.app.process_role
    if role == "migrate":
        return
    checks: list[tuple[str, str]] = [("llm", settings.llm.provider)]
    if role == "api":
        checks.extend(
            [
                ("embedding", settings.embedding.provider),
                ("assessor", settings.retrieval.assessor_provider),
            ]
        )
    elif role == "worker":
        support = str(getattr(settings.retrieval, "support_selector", "deterministic")).lower()
        if support == "llm":
            checks.append(("embedding", settings.embedding.provider))
            # assessor not required for edge builder SUPPORT path
        else:
            # Deterministic SUPPORT still may embed query vectors for edges
            checks.append(("embedding", settings.embedding.provider))
    for label, value in checks:
        if value.lower() in _FAKE_PROVIDERS:
            raise ValueError(f"production forbids {label} provider={value!r}; use a real provider")


def _reject_fake_providers(settings: Settings) -> None:
    """Backward-compatible full-stack production rejection (api role)."""
    for label, value in (
        ("llm", settings.llm.provider),
        ("embedding", settings.embedding.provider),
        ("assessor", settings.retrieval.assessor_provider),
    ):
        if value.lower() in _FAKE_PROVIDERS:
            raise ValueError(f"production forbids {label} provider={value!r}; use a real provider")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
