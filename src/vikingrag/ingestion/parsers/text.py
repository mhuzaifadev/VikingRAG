"""Plain-text parser - paragraph structure with light heading inference."""

from __future__ import annotations

from vikingrag.domain.errors import DocumentParseError
from vikingrag.ingestion.types import ContentBlock, ParsedDocument

_PARSER_VERSION = "1.0.0"


class TextDocumentParser:
    name = "text"
    version = _PARSER_VERSION

    def supports(self, *, filename: str, mime_type: str) -> bool:
        lower = filename.lower()
        return lower.endswith(".txt") or mime_type in {
            "text/plain",
            "application/octet-stream",
        }

    def parse(self, content: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        del mime_type
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(f"Unable to decode text file as UTF-8: {filename}") from exc

        title = filename.rsplit(".", 1)[0] or "Untitled"
        blocks: list[ContentBlock] = []
        offset = 0
        paragraphs = text.split("\n\n")
        for para in paragraphs:
            raw = para
            stripped = para.strip()
            if not stripped:
                offset += len(raw) + 2
                continue
            # Infer heading: short ALL-CAPS or Markdown-less Title Case line
            level = 0
            block_title = None
            body = stripped
            if _looks_like_heading(stripped):
                level = 1 if stripped.isupper() or len(stripped.split()) <= 8 else 0
                if level == 1:
                    block_title = stripped.title() if stripped.isupper() else stripped
                    body = ""
            if block_title:
                blocks.append(
                    ContentBlock(
                        text="",
                        level=level,
                        title=block_title,
                        offset_start=offset,
                        offset_end=offset + len(raw),
                    )
                )
            if body:
                blocks.append(
                    ContentBlock(
                        text=body,
                        level=0,
                        title=None,
                        offset_start=offset,
                        offset_end=offset + len(raw),
                    )
                )
            offset += len(raw) + 2

        if not blocks:
            blocks.append(ContentBlock(text=text.strip() or "", level=0))

        return ParsedDocument(
            title=title,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={"source_filename": filename},
        )


def _looks_like_heading(line: str) -> bool:
    if "\n" in line:
        return False
    if len(line) > 80:
        return False
    if line.endswith("."):
        return False
    words = line.split()
    if not words or len(words) > 12:
        return False
    if line.isupper() and any(c.isalpha() for c in line):
        return True
    # Single-line Title Case without trailing punctuation
    return all(w[:1].isupper() for w in words if w[:1].isalpha()) and len(words) <= 8
