"""FastAPI application factory and lifespan."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from vikingrag import __version__
from vikingrag.api.errors import register_exception_handlers
from vikingrag.api.routers import (
    answers_router,
    documents_router,
    health_router,
    retrieval_router,
    search_router,
)
from vikingrag.application.experience.support_select import build_support_selector
from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.infrastructure.cache.redis import create_redis_client
from vikingrag.infrastructure.database.engine import create_database
from vikingrag.infrastructure.object_store.factory import build_object_store
from vikingrag.observability.context import bind_request_context, clear_request_context
from vikingrag.observability.logging import configure_logging, get_logger
from vikingrag.providers.factory import build_embedding_provider, build_llm_provider
from vikingrag.settings.config import Settings, get_settings
from vikingrag.workers.edge_builder import EdgeBuilderWorker

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


def _provider_configured(name: str) -> bool:
    return name.lower().strip() not in {"unimplemented", "", "none"}


async def _aclose_maybe(obj: Any) -> None:
    if obj is None:
        return
    close = getattr(obj, "aclose", None)
    if close is not None:
        await close()


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
    object_store = build_object_store(settings.object_store)

    embedding_provider = None
    llm_provider = None
    if _provider_configured(settings.embedding.provider):
        try:
            embedding_provider = build_embedding_provider(settings)
        except NotImplementedCapabilityError:
            logger.warning(
                "embedding_provider_unconfigured",
                provider=settings.embedding.provider,
            )
    if _provider_configured(settings.llm.provider):
        try:
            llm_provider = build_llm_provider(settings)
        except NotImplementedCapabilityError:
            logger.warning("llm_provider_unconfigured", provider=settings.llm.provider)

    app.state.database = database
    app.state.redis = redis
    app.state.object_store = object_store
    app.state.embedding_provider = embedding_provider
    app.state.llm_provider = llm_provider

    edge_worker: EdgeBuilderWorker | None = None
    edge_task: asyncio.Task[None] | None = None
    if settings.workers.edge_builder_enabled:
        selector = build_support_selector(
            mode=settings.retrieval.support_selector,
            llm=llm_provider,
            model=settings.llm.model,
            max_support=settings.retrieval.experience_max_support,
        )
        edge_worker = EdgeBuilderWorker(
            database=database,
            poll_interval_seconds=settings.workers.edge_builder_poll_seconds,
            batch_size=settings.workers.edge_builder_batch_size,
            max_support=settings.retrieval.experience_max_support,
            support_selector=selector,
        )
        edge_task = asyncio.create_task(edge_worker.run_forever(), name="edge_builder")
        app.state.edge_builder_worker = edge_worker
        logger.info(
            "edge_builder_worker_attached",
            poll_seconds=settings.workers.edge_builder_poll_seconds,
        )
    else:
        app.state.edge_builder_worker = None
        logger.info("edge_builder_worker_disabled")

    try:
        yield
    finally:
        logger.info("shutting_down_application")
        if edge_worker is not None:
            edge_worker.request_stop()
        if edge_task is not None:
            try:
                await asyncio.wait_for(edge_task, timeout=10.0)
            except (TimeoutError, asyncio.CancelledError):
                edge_task.cancel()
        await _aclose_maybe(embedding_provider)
        await _aclose_maybe(llm_provider)
        await _aclose_maybe(object_store)
        await redis.close()
        await database.dispose()


def create_app(settings: Settings | None = None) -> FastAPI:
    cfg = settings or get_settings()
    app = FastAPI(
        title="VikingRAG",
        description=(
            "Production-oriented hierarchical retrieval platform inspired by VikingRAG. "
            "Hierarchical ingestion, semantic Search, List/Grep/Read, and evidence sufficiency."
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
    app.include_router(search_router)
    app.include_router(retrieval_router)
    app.include_router(answers_router)
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
