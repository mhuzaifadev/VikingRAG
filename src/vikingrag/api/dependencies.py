"""FastAPI dependency injection - resources live on app.state."""

from __future__ import annotations

from typing import Annotated, cast

from fastapi import Depends, Request

from vikingrag.infrastructure.cache.redis import RedisClient
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.object_store.base import ObjectStore
from vikingrag.settings.config import Settings


def get_settings_dep(request: Request) -> Settings:
    return cast(Settings, request.app.state.settings)


def get_database(request: Request) -> Database:
    return cast(Database, request.app.state.database)


def get_redis(request: Request) -> RedisClient:
    return cast(RedisClient, request.app.state.redis)


def get_object_store(request: Request) -> ObjectStore:
    return cast(ObjectStore, request.app.state.object_store)


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
DatabaseDep = Annotated[Database, Depends(get_database)]
RedisDep = Annotated[RedisClient, Depends(get_redis)]
ObjectStoreDep = Annotated[ObjectStore, Depends(get_object_store)]
