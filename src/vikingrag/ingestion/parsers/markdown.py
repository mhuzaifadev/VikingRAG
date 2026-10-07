"""Markdown parser - heading hierarchy from AT# markers."""

from __future__ import annotations

import re

from vikingrag.domain.errors import DocumentParseError
from vikingrag.ingestion.types import ContentBlock, ParsedDocument

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_PARSER_VERSION = "1.0.0"


class MarkdownDocumentParser:
    name = "markdown"
    version = _PARSER_VERSION

    def supports(self, *, filename: str, mime_type: str) -> bool:
        lower = filename.lower()
        return lower.endswith((".md", ".markdown")) or mime_type in {
            "text/markdown",
            "text/x-markdown",
        }

    def parse(self, content: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        del mime_type
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentParseError(f"Unable to decode markdown as UTF-8: {filename}") from exc

        lines = text.splitlines(keepends=True)
        blocks: list[ContentBlock] = []
        buffer: list[str] = []
        offset = 0
        buffer_start = 0
        doc_title = filename.rsplit(".", 1)[0] or "Untitled"

        def flush_buffer() -> None:
            nonlocal buffer, buffer_start
            if not buffer:
                return
            body = "".join(buffer).strip()
            if body:
                end = buffer_start + sum(len(x) for x in buffer)
                blocks.append(
                    ContentBlock(
                        text=body,
                        level=0,
                        offset_start=buffer_start,
                        offset_end=end,
                    )
                )
            buffer = []

        in_fence = False
        fence_marker = ""
        for line in lines:
            stripped = line.rstrip("\n")
            # Preserve fenced code: heading-like lines inside fences are body text
            fence_open = re.match(r"^(```|~~~)(.*)$", stripped)
            if fence_open:
                marker = fence_open.group(1)
                if not in_fence:
                    in_fence = True
                    fence_marker = marker
                elif stripped.startswith(fence_marker):
                    in_fence = False
                    fence_marker = ""
                if not buffer:
                    buffer_start = offset
                buffer.append(line)
                offset += len(line)
                continue

            match = None if in_fence else _HEADING.match(stripped)
            if match:
                flush_buffer()
                level = len(match.group(1))
                title = match.group(2).strip()
                if level == 1 and doc_title == filename.rsplit(".", 1)[0]:
                    doc_title = title
                blocks.append(
                    ContentBlock(
                        text="",
                        level=level,
                        title=title,
                        offset_start=offset,
                        offset_end=offset + len(line),
                    )
                )
            else:
                if not buffer:
                    buffer_start = offset
                buffer.append(line)
            offset += len(line)

        flush_buffer()
        if not blocks:
            blocks.append(ContentBlock(text=text.strip(), level=0))

        return ParsedDocument(
            title=doc_title,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={"source_filename": filename},
        )
