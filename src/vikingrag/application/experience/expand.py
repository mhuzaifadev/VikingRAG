"""Conditioned multi-hop experience-edge expansion (Algorithm 3)."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Sequence
from typing import TYPE_CHECKING, Protocol, TypeVar, runtime_checkable

from vikingrag.application.experience.activate import should_activate
from vikingrag.domain.errors import BudgetExhaustedError, DomainError, InvalidVikingURI
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.experience import (
    ActivatedEdge,
    ExpansionResult,
    ExperienceEdge,
    ExperienceExpansionLimits,
    ExperiencePayload,
)
from vikingrag.domain.models.representation import EmbeddingIdentity
from vikingrag.domain.uri.object_uri import ObjectURIParser
from vikingrag.ingestion.tokenization import ApproxWhitespaceTokenizer

if TYPE_CHECKING:
    from vikingrag.application.budget import RetrievalContext

_TOKENIZER = ApproxWhitespaceTokenizer()
_T = TypeVar("_T")


def document_id_from_uri(uri: str) -> DocumentId | None:
    """Best-effort document id extraction for scope checks."""
    try:
        return ObjectURIParser.parse(uri).document_id
    except (InvalidVikingURI, ValueError, TypeError):
        pass
    # Legacy / test URIs: viking://objects/{uuid}/... or embed UUID after documents/
    import re
    from uuid import UUID

    m = re.search(
        r"(?:objects|documents)/([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
        r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12})",
        uri,
    )
    if m:
        return DocumentId(UUID(m.group(1)))
    return None


def uri_in_scope(uri: str, permitted: frozenset[DocumentId] | None) -> bool:
    """None permitted = unrestricted; empty = deny all."""
    if permitted is None:
        return True
    if not permitted:
        return False
    doc_id = document_id_from_uri(uri)
    if doc_id is None:
        # Cannot prove scope — do not expose (fail closed for experience targets)
        return False
    return doc_id in permitted


@runtime_checkable
class EdgeLookup(Protocol):
    """Outgoing edge lookup used by expansion (SQL or in-memory)."""

    async def find_active_by_source(
        self,
        source_uri: str,
        *,
        identity: EmbeddingIdentity,
    ) -> list[tuple[ExperienceEdge, ExperiencePayload]]: ...

    async def count_active(self) -> int: ...


def estimate_payload_tokens(payload: ExperiencePayload) -> int:
    """Approximate text tokens charged against ``max_tokens`` for an activated payload."""
    text = f"{payload.query_text}\n{payload.trace_summary}"
    return max(1, _TOKENIZER.count(text))


async def expand_experience_edges(
    *,
    seed_uris: Sequence[str],
    query_embedding: list[float],
    identity: EmbeddingIdentity,
    edges: EdgeLookup,
    limits: ExperienceExpansionLimits | None = None,
    ctx: RetrievalContext | None = None,
    permitted_document_ids: frozenset[DocumentId] | None = None,
) -> ExpansionResult:
    """Frontier-by-frontier expansion with visited/cycle tracking and hard caps.

    Empty active edge store → cold path: seeds only, no activated edges.
    ``max_tokens`` charges approximate text tokens from activated payloads.
    Edge I/O honors ``RetrievalContext.await_with_deadline`` when ``ctx`` is set.
    When ``permitted_document_ids`` is set (or taken from ``ctx``), targets outside
    scope are never activated or expanded.
    """
    caps = limits or ExperienceExpansionLimits()
    seeds = tuple(dict.fromkeys(u for u in seed_uris if u and u.strip()))
    if permitted_document_ids is None and ctx is not None:
        permitted_document_ids = ctx.permitted_document_ids

    async def _await_edges(awaitable: Awaitable[_T]) -> _T:
        if ctx is not None:
            return await ctx.await_with_deadline(awaitable)
        return await awaitable

    if await _await_edges(edges.count_active()) == 0:
        return ExpansionResult(
            seed_uris=seeds,
            expanded_uris=(),
            activated=(),
            hops_taken=0,
            edges_considered=0,
            edges_activated=0,
        )

    started = time.perf_counter()
    visited: set[str] = set(seeds)
    frontier: set[str] = set(seeds)
    activated: list[ActivatedEdge] = []
    expanded: list[str] = []
    edges_considered = 0
    hops_taken = 0
    truncated = False
    truncation_reason: str | None = None
    tokens_used = 0

    def _over_deadline() -> bool:
        if ctx is not None:
            try:
                ctx.check_deadline()
            except DomainError:
                return True
        return (time.perf_counter() - started) * 1000.0 >= caps.max_wall_time_ms

    for hop in range(1, caps.max_hops + 1):
        if not frontier:
            break
        if _over_deadline():
            truncated = True
            truncation_reason = "deadline"
            break
        if len(visited) >= caps.max_nodes:
            truncated = True
            truncation_reason = "max_nodes"
            break
        if edges_considered >= caps.max_edges:
            truncated = True
            truncation_reason = "max_edges"
            break

        next_frontier: set[str] = set()
        for source in sorted(frontier):
            if _over_deadline():
                truncated = True
                truncation_reason = "deadline"
                break
            try:
                outgoing = await _await_edges(
                    edges.find_active_by_source(source, identity=identity)
                )
            except BudgetExhaustedError:
                truncated = True
                truncation_reason = "deadline"
                break
            for edge, payload in outgoing:
                edges_considered += 1
                if edges_considered > caps.max_edges:
                    truncated = True
                    truncation_reason = "max_edges"
                    break

                if not uri_in_scope(edge.target_uri, permitted_document_ids):
                    continue
                if not uri_in_scope(edge.source_uri, permitted_document_ids):
                    continue

                ok, sim = should_activate(
                    current_query_embedding=query_embedding,
                    current_identity=identity,
                    payload=payload,
                    edge=edge,
                    gamma=caps.gamma,
                )
                if not ok:
                    continue

                cost = estimate_payload_tokens(payload)
                if tokens_used + cost > caps.max_tokens:
                    truncated = True
                    truncation_reason = "max_tokens"
                    break

                tokens_used += cost
                activated.append(
                    ActivatedEdge(edge=edge, payload=payload, query_similarity=sim, hop=hop)
                )
                target = edge.target_uri
                if target not in visited:
                    visited.add(target)
                    next_frontier.add(target)
                    expanded.append(target)
                    if len(visited) >= caps.max_nodes:
                        truncated = True
                        truncation_reason = "max_nodes"
                        break
            if truncated:
                break

        hops_taken = hop
        frontier = next_frontier
        if truncated:
            break

    return ExpansionResult(
        seed_uris=seeds,
        expanded_uris=tuple(expanded),
        activated=tuple(activated),
        hops_taken=hops_taken,
        edges_considered=edges_considered,
        edges_activated=len(activated),
        truncated=truncated,
        truncation_reason=truncation_reason,
    )
