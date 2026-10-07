"""PDF parser adapter - isolates pypdf; domain sees ContentBlocks only."""

from __future__ import annotations

import re

from vikingrag.domain.errors import DocumentParseError
from vikingrag.ingestion.types import ContentBlock, ParsedDocument

_PARSER_VERSION = "1.0.0"
_NUMBERED = re.compile(r"^(\d+(\.\d+)*)\s+(.+)$")


class PdfDocumentParser:
    name = "pdf"
    version = _PARSER_VERSION

    def supports(self, *, filename: str, mime_type: str) -> bool:
        lower = filename.lower()
        return lower.endswith(".pdf") or mime_type in {
            "application/pdf",
            "application/x-pdf",
        }

    def parse(self, content: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        del mime_type
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover
            raise DocumentParseError("pypdf is required for PDF ingestion") from exc

        try:
            reader = PdfReader(__import__("io").BytesIO(content))
        except Exception as exc:
            raise DocumentParseError(f"Failed to open PDF: {filename}") from exc

        title = _pdf_title(reader, filename)
        blocks: list[ContentBlock] = []
        for page_index, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:
                raise DocumentParseError(
                    f"Failed to extract text from PDF page {page_index}"
                ) from exc

            for line in page_text.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                level, heading = _classify_pdf_line(stripped)
                if heading:
                    blocks.append(
                        ContentBlock(
                            text="",
                            level=level,
                            title=heading,
                            page=page_index,
                            metadata={"extraction": "heuristic_heading"},
                        )
                    )
                else:
                    blocks.append(
                        ContentBlock(
                            text=stripped,
                            level=0,
                            page=page_index,
                        )
                    )

        if not blocks:
            raise DocumentParseError(f"PDF contained no extractable text: {filename}")

        return ParsedDocument(
            title=title,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={
                "source_filename": filename,
                "page_count": len(reader.pages),
                "structure_confidence": "low",
            },
        )


def _pdf_title(reader: object, filename: str) -> str:
    meta = getattr(reader, "metadata", None)
    if meta is not None:
        raw = getattr(meta, "title", None)
        if raw:
            return str(raw)
    return filename.rsplit(".", 1)[0] or "Untitled"


def _classify_pdf_line(line: str) -> tuple[int, str | None]:
    """Best-effort heading detection - do not claim true layout understanding."""
    if len(line) > 120:
        return 0, None
    if line.isupper() and any(c.isalpha() for c in line) and len(line.split()) <= 12:
        return 1, line.title()
    match = _NUMBERED.match(line)
    if match:
        depth = match.group(1).count(".") + 1
        return min(depth, 6), match.group(3).strip()
    words = line.split()
    if (
        len(words) <= 8
        and not line.endswith(".")
        and all(w[:1].isupper() for w in words if w[:1].isalpha())
    ):
        return 2, line
    return 0, None
