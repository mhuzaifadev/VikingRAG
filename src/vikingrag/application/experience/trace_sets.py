"""Extract U_src / U_cand / U_edge URI sets from a retrieval event trace (Alg 2)."""

from __future__ import annotations

from dataclasses import dataclass

from vikingrag.domain.models.experience import RetrievalEvent, RetrievalEventType


@dataclass(frozen=True, slots=True)
class TraceUriSets:
    """Paper Algorithm 2 URI partitions for a completed trace τ."""

    u_src: frozenset[str]
    u_cand: frozenset[str]
    u_edge: frozenset[str]

    @property
    def u_tgt_candidates(self) -> frozenset[str]:
        """Support selection starts from U_cand; U_edge is excluded after selection."""
        return self.u_cand


def _refs(event: RetrievalEvent) -> list[str]:
    return [u for u in event.result_refs if isinstance(u, str) and u.strip()]


def extract_trace_uri_sets(events: list[RetrievalEvent]) -> TraceUriSets:
    """Build U_src, U_cand, U_edge from immutable retrieval events.

    - U_src: union of URIs returned by semantic Search interactions.
    - U_cand: union of Grep-returned URIs and Read-target URIs.
    - U_edge: URIs reached through already existing experience edges.
    """
    u_src: set[str] = set()
    u_cand: set[str] = set()
    u_edge: set[str] = set()

    for event in events:
        refs = _refs(event)
        if event.event_type is RetrievalEventType.SEARCH:
            u_src.update(refs)
        elif event.event_type is RetrievalEventType.GREP:
            u_cand.update(refs)
        elif event.event_type is RetrievalEventType.READ:
            # Read targets may appear in arguments or result_refs
            target = event.arguments.get("uri") or event.arguments.get("target_uri")
            if isinstance(target, str) and target.strip():
                u_cand.add(target.strip())
            u_cand.update(refs)
        elif event.event_type is RetrievalEventType.EDGE_EXPAND:
            u_edge.update(refs)

    return TraceUriSets(
        u_src=frozenset(u_src),
        u_cand=frozenset(u_cand),
        u_edge=frozenset(u_edge),
    )


def summarize_trace(events: list[RetrievalEvent], *, max_chars: int = 1_000) -> str:
    """Compact action/observation summary preserved on the experience payload."""
    parts: list[str] = []
    for event in events:
        refs = _refs(event)[:5]
        parts.append(f"r{event.round_no}:{event.event_type.value}[{','.join(refs)}]")
    summary = " | ".join(parts)
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3] + "..."
