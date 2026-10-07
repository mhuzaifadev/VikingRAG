"""Canonical hierarchy object URIs (paper Eqs 6-7).

Component-wise ancestor UUID path. Legacy ``viking://documents/{id}/nodes/{id}``
URIs remain valid aliases.

Canonical forms:

    viking://objects/{document_id}
    viking://objects/{document_id}/{ancestor}/.../{node_id}
    viking://objects/{document_id}/.../{node_id}.abstract
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from vikingrag.domain.errors import InvalidVikingURI
from vikingrag.domain.models.document import DocumentId, NodeId

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)
_ABSTRACT_SUFFIX = ".abstract"


class ObjectURIKind(StrEnum):
    DIRECTORY = "directory"
    CHUNK = "chunk"
    ABSTRACT = "abstract"
    DOCUMENT = "document"


@dataclass(frozen=True, slots=True)
class ObjectURI:
    document_id: DocumentId
    path_components: tuple[UUID, ...]  # ancestor chain including leaf node id
    kind: ObjectURIKind = ObjectURIKind.DIRECTORY

    @property
    def node_id(self) -> NodeId | None:
        if not self.path_components:
            return None
        return NodeId(self.path_components[-1])

    def __str__(self) -> str:
        return ObjectURIParser.build(self)

    def is_ancestor_of(self, other: ObjectURI) -> bool:
        """True if this path is a proper prefix of other (same document)."""
        if self.document_id != other.document_id:
            return False
        if len(self.path_components) >= len(other.path_components):
            return False
        return other.path_components[: len(self.path_components)] == self.path_components


class ObjectURIParser:
    @staticmethod
    def build(uri: ObjectURI) -> str:
        base = f"viking://objects/{uri.document_id}"
        if not uri.path_components:
            return base
        path = "/".join(str(c) for c in uri.path_components)
        suffix = _ABSTRACT_SUFFIX if uri.kind is ObjectURIKind.ABSTRACT else ""
        return f"{base}/{path}{suffix}"

    @staticmethod
    def build_from_ancestors(
        document_id: DocumentId | UUID,
        ancestor_ids: list[UUID] | tuple[UUID, ...],
        *,
        kind: ObjectURIKind = ObjectURIKind.DIRECTORY,
    ) -> str:
        return ObjectURIParser.build(
            ObjectURI(
                document_id=DocumentId(
                    document_id if isinstance(document_id, UUID) else UUID(str(document_id))
                ),
                path_components=tuple(ancestor_ids),
                kind=kind,
            )
        )

    @staticmethod
    def abstract_uri_for(directory_uri: str) -> str:
        parsed = ObjectURIParser.parse(directory_uri)
        if parsed.kind is ObjectURIKind.ABSTRACT:
            return directory_uri
        return ObjectURIParser.build(
            ObjectURI(
                document_id=parsed.document_id,
                path_components=parsed.path_components,
                kind=ObjectURIKind.ABSTRACT,
            )
        )

    @staticmethod
    def parse(value: str) -> ObjectURI:
        if not isinstance(value, str) or not value.strip():
            raise InvalidVikingURI("URI must be a non-empty string")
        raw = value.strip()
        if not raw.startswith("viking://objects/"):
            raise InvalidVikingURI("Canonical object URI must start with viking://objects/")
        rest = raw[len("viking://objects/") :]
        if ".." in rest.split("/") or "//" in rest or rest.endswith("/"):
            raise InvalidVikingURI("Malformed object URI path")
        is_abstract = rest.endswith(_ABSTRACT_SUFFIX)
        if is_abstract:
            rest = rest[: -len(_ABSTRACT_SUFFIX)]
        parts = rest.split("/") if rest else []
        if not parts or not parts[0]:
            raise InvalidVikingURI("object URI requires document_id")
        document_id = _parse_uuid(parts[0], field="document_id")
        components = tuple(_parse_uuid(p, field="path") for p in parts[1:])
        if is_abstract:
            kind = ObjectURIKind.ABSTRACT
        elif not components:
            kind = ObjectURIKind.DOCUMENT
        else:
            kind = ObjectURIKind.DIRECTORY
        return ObjectURI(
            document_id=DocumentId(document_id),
            path_components=components,
            kind=kind,
        )

    @staticmethod
    def contains_path(ancestor: tuple[UUID, ...], descendant: tuple[UUID, ...]) -> bool:
        """SQL-friendly path containment: ancestor is prefix of descendant."""
        if len(ancestor) > len(descendant):
            return False
        return descendant[: len(ancestor)] == ancestor


def _parse_uuid(value: str, *, field: str) -> UUID:
    if not _UUID_RE.match(value):
        raise InvalidVikingURI(f"Invalid {field}: expected UUID, got {value!r}")
    return UUID(value)
