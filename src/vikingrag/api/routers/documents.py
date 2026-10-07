"""Document ingestion and hierarchy API."""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile, status
from fastapi.responses import JSONResponse

from vikingrag.api.auth import AuthContext, require_auth
from vikingrag.api.schemas.documents import (
    DocumentResponse,
    IngestDocumentResponse,
    NodeResponse,
    TreeNodeResponse,
)
from vikingrag.application.documents import DocumentService
from vikingrag.application.navigation import StructuralNavigationService
from vikingrag.domain.errors import ValidationDomainError
from vikingrag.domain.models.document import DocumentId, IngestionStrategy, NodeId
from vikingrag.domain.models.node import DocumentNode
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository

router = APIRouter(
    prefix="/v1",
    tags=["documents"],
    dependencies=[Depends(require_auth)],
)


def _document_service(request: Request) -> DocumentService:
    return DocumentService(
        database=request.app.state.database,
        object_store=request.app.state.object_store,
        ingestion_settings=request.app.state.settings.ingestion,
    )


async def _read_upload_limited(file: UploadFile, *, max_bytes: int) -> bytes:
    """Enforce upload size while reading — never buffer unbounded content."""
    chunks: list[bytes] = []
    total = 0
    while True:
        piece = await file.read(64 * 1024)
        if not piece:
            break
        total += len(piece)
        if total > max_bytes:
            raise ValidationDomainError(f"Upload exceeds max_upload_bytes={max_bytes}")
        chunks.append(piece)
    return b"".join(chunks)


@router.post("/documents", response_model=IngestDocumentResponse)
async def ingest_document(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    strategy: Annotated[IngestionStrategy, Form()] = IngestionStrategy.SKIP_IDENTICAL,
    external_id: Annotated[str | None, Form()] = None,
) -> IngestDocumentResponse | JSONResponse:
    settings = request.app.state.settings
    max_bytes = settings.ingestion.max_upload_bytes
    # Multipart Content-Length is an upper bound when present
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            # Allow modest multipart overhead above the file itself
            if int(content_length) > max_bytes + 65_536:
                raise ValidationDomainError(f"Upload exceeds max_upload_bytes={max_bytes}")
        except ValueError as exc:
            raise ValidationDomainError("Invalid Content-Length header") from exc

    content = await _read_upload_limited(file, max_bytes=max_bytes)
    filename = file.filename or "upload.bin"
    mime_type = file.content_type or "application/octet-stream"
    service = _document_service(request)
    result = await service.ingest(
        filename=filename,
        mime_type=mime_type,
        content=content,
        external_id=external_id,
        strategy=strategy,
    )
    body = IngestDocumentResponse.from_result(result)
    if result.skipped:
        return JSONResponse(status_code=status.HTTP_200_OK, content=body.model_dump(mode="json"))
    return JSONResponse(status_code=status.HTTP_201_CREATED, content=body.model_dump(mode="json"))


@router.get("/documents/{document_id}", response_model=DocumentResponse)
async def get_document(
    request: Request,
    document_id: UUID,
    auth: Annotated[AuthContext, Depends(require_auth)],
) -> DocumentResponse:
    auth.ensure_document_allowed(document_id)
    service = _document_service(request)
    record = await service.get_document(DocumentId(document_id))
    return DocumentResponse.from_record(record)


@router.get("/documents/{document_id}/tree", response_model=TreeNodeResponse)
async def get_document_tree(
    request: Request,
    document_id: UUID,
    auth: Annotated[AuthContext, Depends(require_auth)],
    include_chunks: bool = Query(False),
) -> TreeNodeResponse:
    auth.ensure_document_allowed(document_id)
    async with request.app.state.database.session() as session:
        nav = StructuralNavigationService(
            documents=SqlDocumentRepository(session),
            nodes=SqlNodeRepository(session),
        )
        tree = await nav.get_document_tree(
            DocumentId(document_id),
            include_chunks=include_chunks,
        )
        return TreeNodeResponse.from_tree(tree)


@router.get("/nodes/{node_id}", response_model=NodeResponse)
async def get_node(
    request: Request,
    node_id: UUID,
    auth: Annotated[AuthContext, Depends(require_auth)],
    include_content: bool = Query(False),
) -> NodeResponse:
    async with request.app.state.database.session() as session:
        nav = StructuralNavigationService(
            documents=SqlDocumentRepository(session),
            nodes=SqlNodeRepository(session),
        )
        node = await nav.get_node(NodeId(node_id))
        auth.ensure_document_allowed(node.document_id)
        return NodeResponse.from_node(node, include_content=include_content)


@router.get("/uris/resolve", response_model=NodeResponse)
async def resolve_uri(
    request: Request,
    auth: Annotated[AuthContext, Depends(require_auth)],
    uri: str = Query(..., min_length=1),
    include_content: bool = Query(False),
) -> NodeResponse:
    async with request.app.state.database.session() as session:
        nav = StructuralNavigationService(
            documents=SqlDocumentRepository(session),
            nodes=SqlNodeRepository(session),
        )
        resolved = await nav.resolve_uri(uri)
        if not isinstance(resolved, DocumentNode):
            raise ValidationDomainError("URI did not resolve to a node")
        auth.ensure_document_allowed(resolved.document_id)
        return NodeResponse.from_node(resolved, include_content=include_content)
