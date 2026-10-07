"""Stable Viking document/node URI build, parse, and validate.

Scheme (Phase 2 - no tenant/corpus segment yet):

    viking://documents/{document_id}
    viking://documents/{document_id}/nodes/{node_id}
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from vikingrag.domain.errors import InvalidVikingURI
from vikingrag.domain.models.document import DocumentId, NodeId

_SCHEME = "viking"
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class VikingURIKind(StrEnum):
    DOCUMENT = "document"
    NODE = "node"


@dataclass(frozen=True, slots=True)
class VikingURI:
    kind: VikingURIKind
    document_id: DocumentId
    node_id: NodeId | None = None

    def __str__(self) -> str:
        return VikingURIParser.build(self)


class VikingURIParser:
    """Build / parse / validate Viking URIs. Storage-independent."""

    @staticmethod
    def build_document_uri(document_id: DocumentId | UUID) -> str:
        return f"viking://documents/{_as_uuid(document_id)}"

    @staticmethod
    def build_node_uri(document_id: DocumentId | UUID, node_id: NodeId | UUID) -> str:
        return f"viking://documents/{_as_uuid(document_id)}/nodes/{_as_uuid(node_id)}"

    @staticmethod
    def build(uri: VikingURI) -> str:
        if uri.kind is VikingURIKind.DOCUMENT:
            return VikingURIParser.build_document_uri(uri.document_id)
        if uri.node_id is None:
            raise InvalidVikingURI("node URI requires node_id")
        return VikingURIParser.build_node_uri(uri.document_id, uri.node_id)

    @staticmethod
    def parse(value: str) -> VikingURI:
        if not isinstance(value, str) or not value:
            raise InvalidVikingURI("URI must be a non-empty string")

        raw = value.strip()
        if "://" not in raw:
            raise InvalidVikingURI("URI must include a scheme")

        scheme, rest = raw.split("://", 1)
        if scheme != _SCHEME:
            raise InvalidVikingURI(f"Unsupported URI scheme: {scheme!r}")

        if not rest or rest.startswith("/"):
            raise InvalidVikingURI("URI host/path is malformed")

        # Reject relative traversal and empty segments early
        if ".." in rest.split("/"):
            raise InvalidVikingURI("URI must not contain '..' path segments")
        if "//" in rest:
            raise InvalidVikingURI("URI must not contain empty path segments")
        if rest.endswith("/"):
            raise InvalidVikingURI("URI must not end with '/'")
        if "?" in rest or "#" in rest:
            raise InvalidVikingURI("URI must not contain query or fragment")
        if "\\" in rest or " " in rest:
            raise InvalidVikingURI("URI contains illegal characters")

        parts = rest.split("/")
        if len(parts) < 2 or parts[0] != "documents":
            raise InvalidVikingURI("URI must begin with viking://documents/{document_id}")

        document_id = _parse_uuid(parts[1], field="document_id")

        if len(parts) == 2:
            return VikingURI(kind=VikingURIKind.DOCUMENT, document_id=DocumentId(document_id))

        if len(parts) == 4 and parts[2] == "nodes":
            node_id = _parse_uuid(parts[3], field="node_id")
            return VikingURI(
                kind=VikingURIKind.NODE,
                document_id=DocumentId(document_id),
                node_id=NodeId(node_id),
            )

        raise InvalidVikingURI("URI path must be documents/{id} or documents/{id}/nodes/{node_id}")

    @staticmethod
    def validate(value: str) -> VikingURI:
        return VikingURIParser.parse(value)


def _as_uuid(value: UUID | DocumentId | NodeId) -> UUID:
    return value if isinstance(value, UUID) else UUID(str(value))


def _parse_uuid(value: str, *, field: str) -> UUID:
    if not _UUID_RE.match(value):
        raise InvalidVikingURI(f"Invalid {field}: expected UUID, got {value!r}")
    try:
        return UUID(value)
    except ValueError as exc:
        raise InvalidVikingURI(f"Invalid {field}: {value!r}") from exc
