"""Single-tenant API-key auth and server-derived document ACL."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

from fastapi import Request
from fastapi.security.utils import get_authorization_scheme_param

from vikingrag.domain.errors import AuthenticationError, ScopeDeniedError
from vikingrag.domain.models.document import DocumentId
from vikingrag.settings.config import (
    Settings,
    derive_permitted_document_ids,
    parse_allowed_document_ids,
)

__all__ = [
    "AuthContext",
    "derive_permitted_document_ids",
    "parse_allowed_document_ids",
    "require_auth",
    "resolve_auth_context",
]


@dataclass(frozen=True, slots=True)
class AuthContext:
    """Resolved auth for a request.

    ``permitted_document_ids``:
      - ``None`` — unrestricted
      - ``frozenset()`` — allow-nothing
      - nonempty frozenset — allowlist
    """

    authenticated: bool
    permitted_document_ids: frozenset[DocumentId] | None

    def ensure_document_allowed(self, document_id: UUID | DocumentId) -> None:
        did = DocumentId(document_id) if isinstance(document_id, UUID) else document_id
        if self.permitted_document_ids is None:
            return
        if did not in self.permitted_document_ids:
            raise ScopeDeniedError(f"Document {did} is outside permitted scope")


def _extract_api_key(request: Request) -> str | None:
    header = request.headers.get("x-api-key")
    if header:
        return header.strip() or None
    authorization = request.headers.get("authorization")
    if not authorization:
        return None
    scheme, param = get_authorization_scheme_param(authorization)
    if scheme.lower() != "bearer":
        return None
    return param.strip() or None


def resolve_auth_context(settings: Settings, request: Request) -> AuthContext:
    auth = settings.auth
    permitted = derive_permitted_document_ids(auth)
    if not auth.enabled:
        return AuthContext(authenticated=False, permitted_document_ids=None)

    expected = auth.api_key
    if not expected:
        raise AuthenticationError("Auth is enabled but VIKINGRAG_AUTH_API_KEY is not configured")

    provided = _extract_api_key(request)
    if provided is None or provided != expected:
        raise AuthenticationError("Invalid or missing API key")

    return AuthContext(authenticated=True, permitted_document_ids=permitted)


async def require_auth(request: Request) -> AuthContext:
    """FastAPI dependency: enforce API key when auth is enabled."""
    settings: Settings = request.app.state.settings
    return resolve_auth_context(settings, request)
