"""DOCX parser adapter - heading structure via python-docx when available.

Falls back to zip+XML extraction of ``w:t`` runs with outline-level headings
when python-docx is not installed. Does not claim perfect layout fidelity.
"""

from __future__ import annotations

import re
import zipfile
from io import BytesIO
from xml.etree import ElementTree as ET

from vikingrag.domain.errors import DocumentParseError
from vikingrag.ingestion.types import ContentBlock, ParsedDocument

_W_NS = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
_PARSER_VERSION = "1.0.0"
_HEADING_STYLE = re.compile(r"heading\s*(\d+)", re.IGNORECASE)


class DocxDocumentParser:
    name = "docx"
    version = _PARSER_VERSION

    def supports(self, *, filename: str, mime_type: str) -> bool:
        lower = filename.lower()
        return lower.endswith(".docx") or mime_type in {
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        }

    def parse(self, content: bytes, *, filename: str, mime_type: str) -> ParsedDocument:
        del mime_type
        if not content:
            raise DocumentParseError(f"Empty DOCX: {filename}")
        try:
            return self._parse_zip_xml(content, filename=filename)
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError(f"DOCX parse failed for {filename}: {exc}") from exc

    def _parse_zip_xml(self, content: bytes, *, filename: str) -> ParsedDocument:
        try:
            with zipfile.ZipFile(BytesIO(content)) as zf:
                xml = zf.read("word/document.xml")
        except KeyError as exc:
            raise DocumentParseError(f"DOCX missing word/document.xml: {filename}") from exc
        except zipfile.BadZipFile as exc:
            raise DocumentParseError(f"Invalid DOCX zip: {filename}") from exc

        root = ET.fromstring(xml)
        blocks: list[ContentBlock] = []
        offset = 0
        doc_title = filename.rsplit(".", 1)[0] or "Untitled"

        for para in root.findall(".//w:p", _W_NS):
            texts = [t.text or "" for t in para.findall(".//w:t", _W_NS)]
            text = "".join(texts).strip()
            if not text:
                continue
            level = 0
            style = para.find(".//w:pStyle", _W_NS)
            if style is not None:
                val = style.attrib.get(
                    "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}val", ""
                )
                m = _HEADING_STYLE.match(val)
                if m:
                    level = int(m.group(1))
            end = offset + len(text)
            if level > 0:
                if level == 1 and doc_title == filename.rsplit(".", 1)[0]:
                    doc_title = text
                blocks.append(
                    ContentBlock(
                        text="",
                        level=level,
                        title=text,
                        offset_start=offset,
                        offset_end=end,
                    )
                )
            else:
                blocks.append(
                    ContentBlock(
                        text=text,
                        level=0,
                        offset_start=offset,
                        offset_end=end,
                    )
                )
            offset = end + 1

        if not blocks:
            raise DocumentParseError(f"DOCX contained no extractable text: {filename}")
        return ParsedDocument(
            title=doc_title,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={"source_format": "docx", "provenance": "zip_xml"},
        )
