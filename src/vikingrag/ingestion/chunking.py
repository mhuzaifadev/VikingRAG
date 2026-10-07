"""Structure-aware chunking - never crosses structural parent boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import UUID, uuid4

from vikingrag.domain.models.document import NodeType
from vikingrag.ingestion.tokenization import Tokenizer, create_tokenizer
from vikingrag.ingestion.types import ChunkDraft, HierarchyDraft, HierarchyDraftNode


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    target_tokens: int = 650
    max_tokens: int = 900
    overlap_tokens: int = 80


@runtime_checkable
class ChunkingPolicy(Protocol):
    def chunk(self, hierarchy: HierarchyDraft) -> tuple[HierarchyDraft, list[ChunkDraft]]: ...


class StructureAwareChunkingPolicy:
    """Split leaf structural node text into CHUNK children within the same region."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._config = config or ChunkingConfig()
        self._tokenizer = tokenizer or create_tokenizer()
        if self._config.overlap_tokens >= self._config.target_tokens:
            raise ValueError("overlap_tokens must be < target_tokens")
        if self._config.target_tokens > self._config.max_tokens:
            raise ValueError("target_tokens must be <= max_tokens")

    def chunk(self, hierarchy: HierarchyDraft) -> tuple[HierarchyDraft, list[ChunkDraft]]:
        nodes = list(hierarchy.nodes)
        chunks: list[ChunkDraft] = []
        children_by_parent: dict[UUID | None, list[HierarchyDraftNode]] = {}
        for node in nodes:
            children_by_parent.setdefault(node.parent_temp_id, []).append(node)

        for node in list(nodes):
            if node.node_type is NodeType.CHUNK:
                continue
            # Chunk direct body text even when the node also has structural children
            # (paper: sections may own body text and child sections).
            text = (node.text or "").strip()
            if not text:
                continue

            pieces = self._split_text(text)
            # Ordinals must be unique among ALL siblings (structural + chunk).
            sibling_ordinals = [c.ordinal for c in children_by_parent.get(node.temp_id, [])]
            next_ordinal = (max(sibling_ordinals) + 1) if sibling_ordinals else 0
            parent_temp_id = node.temp_id
            parent_depth = node.depth
            parent_title = node.title

            def _emit(
                piece_text: str,
                token_count: int,
                *,
                _parent_temp_id: UUID = parent_temp_id,
                _parent_depth: int = parent_depth,
                _parent_title: str | None = parent_title,
            ) -> None:
                nonlocal next_ordinal
                chunk_id = uuid4()
                ordinal = next_ordinal
                next_ordinal += 1
                chunk_node = HierarchyDraftNode(
                    temp_id=chunk_id,
                    parent_temp_id=_parent_temp_id,
                    node_type=NodeType.CHUNK,
                    title=None,
                    ordinal=ordinal,
                    depth=_parent_depth + 1,
                    text=piece_text,
                    metadata={
                        "parent_title": _parent_title,
                        "chunk_index": ordinal,
                        "owns_parent_body": True,
                    },
                )
                nodes.append(chunk_node)
                chunks.append(
                    ChunkDraft(
                        parent_temp_id=_parent_temp_id,
                        ordinal=ordinal,
                        text=piece_text,
                        token_count=token_count,
                        metadata=dict(chunk_node.metadata),
                    )
                )

            for piece, _token_count in pieces:
                recounted = self._tokenizer.count(piece)
                if recounted > self._config.max_tokens:
                    for sub_piece, sub_count in self._window_split(piece):
                        _emit(sub_piece, sub_count)
                else:
                    _emit(piece, recounted)
            # Structural node keeps title/metadata; body lives in chunks
            node.text = None

        return HierarchyDraft(document_title=hierarchy.document_title, nodes=nodes), chunks

    def _split_text(self, text: str) -> list[tuple[str, int]]:
        cfg = self._config
        total = self._tokenizer.count(text)
        if total <= cfg.max_tokens:
            return [(text, total)]

        # Prefer sentence-ish splits, then pack to target size
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text]

        units: list[str] = []
        for para in paragraphs:
            sentences = _split_sentences(para)
            units.extend(sentences if sentences else [para])

        packed: list[tuple[str, int]] = []
        current: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self._tokenizer.count(unit)
            if unit_tokens > cfg.max_tokens:
                # Hard-split oversized unit by token window
                if current:
                    packed.append(("\n\n".join(current), current_tokens))
                    current, current_tokens = [], 0
                packed.extend(self._window_split(unit))
                continue

            if current and current_tokens + unit_tokens > cfg.target_tokens:
                packed.append(("\n\n".join(current), current_tokens))
                # Overlap: keep trailing units within same region
                if cfg.overlap_tokens > 0 and current:
                    overlap = _take_overlap(current, cfg.overlap_tokens, self._tokenizer)
                    current = overlap
                    current_tokens = self._tokenizer.count("\n\n".join(current)) if current else 0
                else:
                    current, current_tokens = [], 0

            current.append(unit)
            current_tokens += unit_tokens

        if current:
            packed.append(("\n\n".join(current), current_tokens))
        return packed

    def _window_split(self, text: str) -> list[tuple[str, int]]:
        """Token-window split for oversized units; overlap stays inside this unit."""
        cfg = self._config
        try:
            tokens = self._tokenizer.encode(text)
            decode = self._tokenizer.decode
        except Exception:
            # Character fallback for approx tokenizer
            step = max(1, cfg.target_tokens * 4)
            overlap = max(0, cfg.overlap_tokens * 4)
            out: list[tuple[str, int]] = []
            i = 0
            while i < len(text):
                piece = text[i : i + step]
                out.append((piece, self._tokenizer.count(piece)))
                if i + step >= len(text):
                    break
                i = max(i + step - overlap, i + 1)
            return out

        out = []
        start = 0
        n = len(tokens)
        while start < n:
            end = min(start + cfg.target_tokens, n)
            # Prefer not exceeding max
            end = min(end, start + cfg.max_tokens)
            piece_tokens = tokens[start:end]
            piece = decode(piece_tokens)
            out.append((piece, len(piece_tokens)))
            if end >= n:
                break
            start = max(end - cfg.overlap_tokens, start + 1)
        return out


def _split_sentences(text: str) -> list[str]:
    import re

    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _take_overlap(units: list[str], overlap_tokens: int, tokenizer: Tokenizer) -> list[str]:
    taken: list[str] = []
    total = 0
    for unit in reversed(units):
        t = tokenizer.count(unit)
        if taken and total + t > overlap_tokens:
            break
        taken.insert(0, unit)
        total += t
    return taken
