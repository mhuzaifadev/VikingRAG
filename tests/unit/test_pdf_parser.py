from __future__ import annotations

from pathlib import Path

from vikingrag.ingestion.hierarchy import build_hierarchy
from vikingrag.ingestion.parsers.pdf import PdfDocumentParser

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_pdf_parser_extracts_text() -> None:
    content = (FIXTURES / "sample.pdf").read_bytes()
    parsed = PdfDocumentParser().parse(content, filename="sample.pdf", mime_type="application/pdf")
    assert parsed.blocks
    joined = " ".join(b.title or b.text for b in parsed.blocks)
    assert "ARCHITECTURE" in joined.upper() or "PostgreSQL" in joined
    hierarchy = build_hierarchy(parsed)
    assert hierarchy.nodes
