"""Map domain errors to HTTP responses."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vikingrag.domain.errors import (
    ConflictError,
    DocumentAlreadyExists,
    DocumentNotFound,
    DocumentParseError,
    DomainError,
    HierarchyConstructionError,
    InvalidVikingURI,
    NodeNotFound,
    NotFoundError,
    UnsupportedDocumentType,
    ValidationDomainError,
)
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


def _payload(exc: DomainError) -> dict[str, str]:
    return {"error": exc.code, "message": exc.message}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(UnsupportedDocumentType)
    async def unsupported_type(_request: Request, exc: UnsupportedDocumentType) -> JSONResponse:
        return JSONResponse(status_code=415, content=_payload(exc))

    @app.exception_handler(DocumentParseError)
    async def parse_error(_request: Request, exc: DocumentParseError) -> JSONResponse:
        return JSONResponse(status_code=422, content=_payload(exc))

    @app.exception_handler(HierarchyConstructionError)
    async def hierarchy_error(_request: Request, exc: HierarchyConstructionError) -> JSONResponse:
        return JSONResponse(status_code=422, content=_payload(exc))

    @app.exception_handler(InvalidVikingURI)
    async def invalid_uri(_request: Request, exc: InvalidVikingURI) -> JSONResponse:
        return JSONResponse(status_code=400, content=_payload(exc))

    @app.exception_handler(DocumentAlreadyExists)
    async def already_exists(_request: Request, exc: DocumentAlreadyExists) -> JSONResponse:
        return JSONResponse(status_code=409, content=_payload(exc))

    @app.exception_handler(DocumentNotFound)
    async def document_not_found(_request: Request, exc: DocumentNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content=_payload(exc))

    @app.exception_handler(NodeNotFound)
    async def node_not_found(_request: Request, exc: NodeNotFound) -> JSONResponse:
        return JSONResponse(status_code=404, content=_payload(exc))

    @app.exception_handler(NotFoundError)
    async def not_found(_request: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content=_payload(exc))

    @app.exception_handler(ConflictError)
    async def conflict(_request: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content=_payload(exc))

    @app.exception_handler(ValidationDomainError)
    async def validation(_request: Request, exc: ValidationDomainError) -> JSONResponse:
        return JSONResponse(status_code=422, content=_payload(exc))

    @app.exception_handler(DomainError)
    async def domain(_request: Request, exc: DomainError) -> JSONResponse:
        logger.warning("domain_error", code=exc.code, message=exc.message)
        return JSONResponse(status_code=400, content=_payload(exc))
