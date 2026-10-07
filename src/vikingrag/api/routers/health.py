"""Liveness and readiness probes."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["health"])


class DependencyStatus(BaseModel):
    name: str
    status: Literal["ok", "error", "skipped"]
    detail: str | None = None


class LiveResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str


class ReadyResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    service: str
    checks: list[DependencyStatus] = Field(default_factory=list)


@router.get("/health/live", response_model=LiveResponse)
@router.get("/v1/health/live", response_model=LiveResponse)
async def liveness(request: Request) -> LiveResponse:
    settings = request.app.state.settings
    return LiveResponse(service=settings.app.name)


@router.get("/health/ready", response_model=ReadyResponse)
@router.get("/v1/health/ready", response_model=ReadyResponse)
async def readiness(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    checks: list[DependencyStatus] = []

    # PostgreSQL
    try:
        database = request.app.state.database
        await database.ping()
        checks.append(DependencyStatus(name="postgres", status="ok"))
    except Exception as exc:
        logger.warning("readiness_postgres_failed", error=str(exc))
        checks.append(DependencyStatus(name="postgres", status="error", detail=str(exc)))

    # Redis
    try:
        redis = request.app.state.redis
        await redis.ping()
        checks.append(DependencyStatus(name="redis", status="ok"))
    except Exception as exc:
        logger.warning("readiness_redis_failed", error=str(exc))
        checks.append(DependencyStatus(name="redis", status="error", detail=str(exc)))

    # Object store: local root is always "ready" if constructed; S3 later
    try:
        store = request.app.state.object_store
        _ = store  # ensure bound
        checks.append(DependencyStatus(name="object_store", status="ok"))
    except Exception as exc:
        checks.append(DependencyStatus(name="object_store", status="error", detail=str(exc)))

    failed = [c for c in checks if c.status == "error"]
    if failed:
        body = ReadyResponse(status="error", service=settings.app.name, checks=checks)
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=body.model_dump(),
        )

    body = ReadyResponse(status="ok", service=settings.app.name, checks=checks)
    return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump())
