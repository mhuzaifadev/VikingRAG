"""Parser registry - extension point for DOCX/HTML/JSON/CSV later."""

from __future__ import annotations

from vikingrag.domain.errors import UnsupportedDocumentType
from vikingrag.ingestion.parsers.base import DocumentParser
from vikingrag.ingestion.parsers.markdown import MarkdownDocumentParser
from vikingrag.ingestion.parsers.pdf import PdfDocumentParser
from vikingrag.ingestion.parsers.text import TextDocumentParser


class ParserRegistry:
    def __init__(self, parsers: list[DocumentParser]) -> None:
        self._parsers = parsers

    def resolve(self, *, filename: str, mime_type: str) -> DocumentParser:
        for parser in self._parsers:
            if parser.supports(filename=filename, mime_type=mime_type):
                return parser
        raise UnsupportedDocumentType(
            f"No parser registered for filename={filename!r} mime_type={mime_type!r}"
        )


def build_default_parser_registry() -> ParserRegistry:
    return ParserRegistry(
        [
            MarkdownDocumentParser(),
            TextDocumentParser(),
            PdfDocumentParser(),
        ]
    )
