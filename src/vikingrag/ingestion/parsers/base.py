"""Document parser protocol - adapters must not leak vendor types."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vikingrag.ingestion.types import ParsedDocument


@runtime_checkable
class DocumentParser(Protocol):
    name: str
    version: str

    def supports(self, *, filename: str, mime_type: str) -> bool: ...

    def parse(self, content: bytes, *, filename: str, mime_type: str) -> ParsedDocument: ...
