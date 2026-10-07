"""Assemble and validate EvidenceBundle from authoritative Reads."""

from __future__ import annotations

from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.evidence import (
    EvidenceBundle,
    ExcludedEvidence,
    ExclusionReason,
    RetrievedEvidence,
)
from vikingrag.domain.models.primitives import OffsetSystem, ReadResponse


def evidence_id_for(index: int) -> str:
    return f"e{index + 1}"


def evidence_from_read(
    read: ReadResponse,
    *,
    evidence_id: str,
    provenance: tuple[str, ...],
    discovery_score: float | None = None,
) -> RetrievedEvidence | None:
    if not read.has_direct_content or not read.text.strip():
        return None
    return RetrievedEvidence(
        evidence_id=evidence_id,
        uri=read.uri,
        text=read.text,
        token_count=read.token_count,
        document_id=read.document_id,
        node_id=read.node_id,
        content_hash=read.content_hash,
        start_offset=read.start_offset,
        end_offset=read.end_offset,
        offset_system=read.offset_system,
        discovery_score=discovery_score,
        score=discovery_score,
        provenance=provenance,
        metadata=dict(read.source_metadata),
    )


def assemble_evidence_bundle(
    candidates: list[RetrievedEvidence],
    *,
    max_tokens: int,
) -> EvidenceBundle:
    """Deduplicate identical ranges, collapse overlaps per node revision, enforce token cap."""
    excluded: list[ExcludedEvidence] = []
    accepted: list[RetrievedEvidence] = []
    seen_ranges: set[tuple[DocumentId | None, NodeId | None, str, int, int]] = set()
    # Track revision per node - reject mixed hashes
    node_hash: dict[NodeId, str] = {}
    # Per-node accepted intervals for overlap collapse
    node_intervals: dict[NodeId, list[tuple[int, int]]] = {}

    total = 0
    truncated = False
    next_id = 0

    for item in candidates:
        if not item.text.strip():
            excluded.append(
                ExcludedEvidence(uri=item.uri, reason=ExclusionReason.EMPTY, detail="empty text")
            )
            continue
        if item.node_id is not None and item.content_hash:
            prior = node_hash.get(item.node_id)
            if prior is not None and prior != item.content_hash:
                excluded.append(
                    ExcludedEvidence(
                        uri=item.uri,
                        reason=ExclusionReason.MIXED_REVISION,
                        detail="mixed content_hash for same node",
                    )
                )
                continue
            node_hash[item.node_id] = item.content_hash

        key = (
            item.document_id,
            item.node_id,
            item.content_hash,
            item.start_offset,
            item.end_offset,
        )
        if key in seen_ranges:
            excluded.append(
                ExcludedEvidence(
                    uri=item.uri,
                    reason=ExclusionReason.DUPLICATE_RANGE,
                    detail="identical source range",
                )
            )
            continue

        # Retain uncovered tails when new interval partially overlaps existing ones
        pieces: list[tuple[int, int, str, int]]
        if item.node_id is not None:
            intervals = node_intervals.setdefault(item.node_id, [])
            pieces = _uncovered_segments(
                item.start_offset,
                item.end_offset,
                item.text,
                item.token_count,
                intervals,
            )
            if not pieces:
                excluded.append(
                    ExcludedEvidence(
                        uri=item.uri,
                        reason=ExclusionReason.OVERLAP_COLLAPSED,
                        detail="fully covered by existing intervals on same node",
                    )
                )
                continue
        else:
            pieces = [(item.start_offset, item.end_offset, item.text, item.token_count)]

        for seg_start, seg_end, seg_text, seg_tokens in pieces:
            if total + seg_tokens > max_tokens:
                excluded.append(
                    ExcludedEvidence(
                        uri=item.uri,
                        reason=ExclusionReason.BUNDLE_TOKEN_LIMIT,
                        detail=f"would exceed max_tokens={max_tokens}",
                    )
                )
                truncated = True
                continue

            eid = item.evidence_id or evidence_id_for(next_id)
            next_id += 1
            if len(pieces) > 1 or (seg_start, seg_end) != (item.start_offset, item.end_offset):
                eid = f"{eid}_t{seg_start}"
            normalized = RetrievedEvidence(
                evidence_id=eid,
                uri=item.uri,
                text=seg_text,
                token_count=seg_tokens,
                document_id=item.document_id,
                node_id=item.node_id,
                chunk_id=item.chunk_id,
                content_hash=item.content_hash,
                start_offset=seg_start,
                end_offset=seg_end,
                offset_system=item.offset_system or OffsetSystem.UNICODE_CODE_POINT,
                discovery_score=item.discovery_score
                if item.discovery_score is not None
                else item.score,
                score=item.score,
                provenance=item.provenance,
                metadata={**dict(item.metadata), "overlap_tail": True}
                if (seg_start, seg_end) != (item.start_offset, item.end_offset)
                else dict(item.metadata),
            )
            accepted.append(normalized)
            seen_ranges.add(
                (
                    item.document_id,
                    item.node_id,
                    item.content_hash,
                    seg_start,
                    seg_end,
                )
            )
            if item.node_id is not None:
                node_intervals.setdefault(item.node_id, []).append((seg_start, seg_end))
            total += seg_tokens

    return EvidenceBundle.from_items(
        accepted,
        excluded=excluded,
        truncated=truncated,
        complete=not truncated,
    )


def _overlaps(intervals: list[tuple[int, int]], start: int, end: int) -> bool:
    return any(start < b and end > a for a, b in intervals)


def _uncovered_segments(
    start: int,
    end: int,
    text: str,
    token_count: int,
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int, str, int]]:
    """Return disjoint sub-ranges of [start,end) not covered by existing intervals.

    Text is sliced by unicode offsets relative to the original evidence span.
    Token counts stay exact when unsplit; proportional when split.
    """
    span = max(1, end - start)
    if not _overlaps(intervals, start, end):
        return [(start, end, text, token_count)]

    covered = sorted(intervals)
    cursor = start
    out: list[tuple[int, int, str, int]] = []
    for a, b in covered:
        if b <= cursor:
            continue
        if a > cursor:
            seg_start, seg_end = cursor, min(a, end)
            if seg_start < seg_end:
                local = text[seg_start - start : seg_end - start]
                tokens = (
                    max(1, (token_count * (seg_end - seg_start)) // span) if local.strip() else 0
                )
                out.append((seg_start, seg_end, local, tokens))
        cursor = max(cursor, b)
        if cursor >= end:
            break
    if cursor < end:
        local = text[cursor - start : end - start]
        tokens = max(1, (token_count * (end - cursor)) // span) if local.strip() else 0
        out.append((cursor, end, local, tokens))
    return out
