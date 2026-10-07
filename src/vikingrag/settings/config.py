"""Typed application configuration from environment / .env."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    model: str = "gpt-4o-mini"
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
    model: str = "text-embedding-3-small"
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
    )
    @classmethod
    def _positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("must be >= 1")
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

    @model_validator(mode="after")
    def _pool_vs_top_k(self) -> Settings:
        if self.retrieval.candidate_pool_size < self.retrieval.initial_top_k:
            raise ValueError("retrieval.candidate_pool_size must be >= initial_top_k")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
