"""Typed application configuration from environment / .env."""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
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
    model: str = ""
    timeout_seconds: float = 30.0


class EmbeddingSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_EMBEDDING_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = "unimplemented"
    model: str = ""
    dimensions: int = 1536
    timeout_seconds: float = 30.0


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VIKINGRAG_RETRIEVAL_",
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )

    initial_top_k: int = 8
    max_rounds: int = 6
    max_tool_calls: int = 12
    max_read_tokens: int = 12_000
    max_wall_time_ms: int = 12_000

    @field_validator(
        "initial_top_k",
        "max_rounds",
        "max_tool_calls",
        "max_read_tokens",
        "max_wall_time_ms",
    )
    @classmethod
    def _positive(cls, value: int) -> int:
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
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def clear_settings_cache() -> None:
    get_settings.cache_clear()
