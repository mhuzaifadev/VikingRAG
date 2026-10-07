"""Document hierarchy identity types."""

from __future__ import annotations

from enum import StrEnum
from typing import NewType
from uuid import UUID

DocumentId = NewType("DocumentId", UUID)
NodeId = NewType("NodeId", UUID)
ChunkId = NewType("ChunkId", UUID)


class DocumentStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"
    DELETED = "deleted"


class NodeType(StrEnum):
    DOCUMENT = "document"
    SECTION = "section"
    SUBSECTION = "subsection"
    CHUNK = "chunk"
    # Extension points for later phases
    CHAPTER = "chapter"
    APPENDIX = "appendix"
    TABLE = "table"


class AbstractStatus(StrEnum):
    NONE = "none"
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"


class IngestionStrategy(StrEnum):
    SKIP_IDENTICAL = "skip_identical"
    REPLACE = "replace"
    CREATE_VERSION = "create_version"
