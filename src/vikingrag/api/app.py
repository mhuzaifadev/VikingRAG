"""FastAPI application factory and lifespan."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from vikingrag import __version__
from vikingrag.api.errors import register_exception_handlers
from vikingrag.api.routers import documents_router, health_router
from vikingrag.infrastructure.cache.redis import create_redis_client
from vikingrag.infrastructure.database.engine import create_database
from vikingrag.infrastructure.object_store.local import LocalObjectStore
from vikingrag.observability.context import bind_request_context, clear_request_context
from vikingrag.observability.logging import configure_logging, get_logger
from vikingrag.settings.config import Settings, get_settings

logger = get_logger(__name__)


class CorrelationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("x-request-id")
        trace_id = request.headers.get("x-trace-id")
        rid, tid = bind_request_context(request_id=request_id, trace_id=trace_id)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = rid
            response.headers["x-trace-id"] = tid
            return response
        finally:
            clear_request_context()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(
        level=settings.app.log_level,
        json_logs=settings.app.env != "development",
    )
    logger.info("starting_application", version=__version__, env=settings.app.env)

    database = create_database(settings.database)
    redis = create_redis_client(settings.redis)
    object_store = LocalObjectStore(settings.object_store.local_root)

    app.state.database = database
    app.state.redis = redis
    app.state.object_store = object_store

    try:
        yield
    finally:
        logger.info("shutting_down_application")
        await redis.close()
        await database.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    app = FastAPI(
        title="VikingRAG",
        description=(
            "Production-oriented hierarchical retrieval platform inspired by VikingRAG. "
            "Phase 2 adds hierarchical document ingestion and structural navigation."
        ),
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )
    app.state.settings = cfg
    app.add_middleware(CorrelationMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(documents_router)
    return app


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "vikingrag.api.app:create_app",
        factory=True,
        host=settings.app.host,
        port=settings.app.port,
        reload=settings.app.debug,
    )


# ASGI entry used by Docker / uvicorn without factory flag when needed
app: Any = None
