"""FastAPI dependency injection - resources live on app.state."""

from __future__ import annotations

from typing import Annotated, Any, cast

from fastapi import Depends, Request

from vikingrag.api.auth import AuthContext, require_auth
from vikingrag.infrastructure.cache.redis import RedisClient
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.object_store.base import ObjectStore
from vikingrag.providers.embeddings.base import EmbeddingProvider
from vikingrag.providers.factory import build_embedding_provider, build_llm_provider
from vikingrag.settings.config import Settings


def get_settings_dep(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.database)


def get_redis(request: Request) -> RedisClient:
    return cast(RedisClient, request.app.state.redis)


def get_object_store(request: Request) -> ObjectStore:
    return cast(ObjectStore, request.app.state.object_store)


def get_embedding_provider(request: Request) -> EmbeddingProvider:
    owned = getattr(request.app.state, "embedding_provider", None)
    if owned is not None:
        return cast(EmbeddingProvider, owned)
    return build_embedding_provider(cast(Settings, request.app.state.settings))


def get_llm_provider(request: Request) -> Any:
    owned = getattr(request.app.state, "llm_provider", None)
    if owned is not None:
        return owned
    return build_llm_provider(cast(Settings, request.app.state.settings))


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DatabaseDep = Annotated[Database, Depends(get_database)]
RedisDep = Annotated[RedisClient, Depends(get_redis)]
ObjectStoreDep = Annotated[ObjectStore, Depends(get_object_store)]
AuthDep = Annotated[AuthContext, Depends(require_auth)]
EmbeddingProviderDep = Annotated[EmbeddingProvider, Depends(get_embedding_provider)]
LLMProviderDep = Annotated[Any, Depends(get_llm_provider)]
