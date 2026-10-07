from __future__ import annotations

from pathlib import Path

from vikingrag.domain.models.document import NodeType
from vikingrag.ingestion.chunking import ChunkingConfig, StructureAwareChunkingPolicy
from vikingrag.ingestion.hierarchy import build_hierarchy
from vikingrag.ingestion.parsers.markdown import MarkdownDocumentParser
from vikingrag.ingestion.parsers.text import TextDocumentParser
from vikingrag.ingestion.tokenization import ApproxWhitespaceTokenizer

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_markdown_parser_preserves_headings() -> None:
    content = (FIXTURES / "architecture.md").read_bytes()
    parsed = MarkdownDocumentParser().parse(
        content, filename="architecture.md", mime_type="text/markdown"
    )
    titles = [b.title for b in parsed.blocks if b.title]
    assert "System Architecture" in titles
    assert "Storage" in titles
    assert "PostgreSQL" in titles
    assert "Evidence Verification" in titles


def test_hierarchy_from_architecture_fixture() -> None:
    content = (FIXTURES / "architecture.md").read_bytes()
    parsed = MarkdownDocumentParser().parse(
        content, filename="architecture.md", mime_type="text/markdown"
    )
    hierarchy = build_hierarchy(parsed)
    by_title = {n.title: n for n in hierarchy.nodes if n.title}
    root = by_title["System Architecture"]
    assert root.node_type is NodeType.DOCUMENT
    storage = by_title["Storage"]
    postgres = by_title["PostgreSQL"]
    assert storage.parent_temp_id == root.temp_id
    assert postgres.parent_temp_id == storage.temp_id
    assert by_title["Semantic Search"].parent_temp_id == by_title["Retrieval"].temp_id


def test_structure_aware_chunking_stays_in_region() -> None:
    content = (FIXTURES / "architecture.md").read_bytes()
    parsed = MarkdownDocumentParser().parse(
        content, filename="architecture.md", mime_type="text/markdown"
    )
    hierarchy = build_hierarchy(parsed)
    # Force small chunks
    policy = StructureAwareChunkingPolicy(
        config=ChunkingConfig(target_tokens=20, max_tokens=40, overlap_tokens=5),
        tokenizer=ApproxWhitespaceTokenizer(),
    )
    chunked, chunks = policy.chunk(hierarchy)
    chunk_nodes = [n for n in chunked.nodes if n.node_type is NodeType.CHUNK]
    assert chunk_nodes
    assert chunks
    parents = {n.parent_temp_id for n in chunk_nodes}
    # Every chunk parent is a structural node in the same draft
    structural_ids = {n.temp_id for n in chunked.nodes if n.node_type is not NodeType.CHUNK}
    assert parents <= structural_ids


def test_text_parser_paragraphs() -> None:
    raw = b"INTRODUCTION\n\nHello world.\n\nMore text here."
    parsed = TextDocumentParser().parse(raw, filename="note.txt", mime_type="text/plain")
    assert parsed.blocks
    hierarchy = build_hierarchy(parsed)
    assert any(n.node_type is NodeType.DOCUMENT for n in hierarchy.nodes)
