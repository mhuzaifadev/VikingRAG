"""Conditioned multi-hop experience-edge expansion (Algorithm 3)."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from vikingrag.application.experience.activate import should_activate
from vikingrag.domain.models.experience import (
    ActivatedEdge,
    ExpansionResult,
    ExperienceEdge,
    ExperienceExpansionLimits,
    ExperiencePayload,
)
from vikingrag.domain.models.representation import EmbeddingIdentity


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


async def expand_experience_edges(
    *,
    seed_uris: Sequence[str],
    query_embedding: list[float],
    identity: EmbeddingIdentity,
    edges: EdgeLookup,
    limits: ExperienceExpansionLimits | None = None,
) -> ExpansionResult:
    """Frontier-by-frontier expansion with visited/cycle tracking and hard caps.

    Empty active edge store → cold path: seeds only, no activated edges.
    """
    caps = limits or ExperienceExpansionLimits()
    seeds = tuple(dict.fromkeys(u for u in seed_uris if u and u.strip()))

    if await edges.count_active() == 0:
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

    def _over_deadline() -> bool:
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
            outgoing = await edges.find_active_by_source(source, identity=identity)
            for edge, payload in outgoing:
                edges_considered += 1
                if edges_considered > caps.max_edges:
                    truncated = True
                    truncation_reason = "max_edges"
                    break

                ok, sim = should_activate(
                    current_query_embedding=query_embedding,
                    current_identity=identity,
                    payload=payload,
                    edge=edge,
                    gamma=caps.gamma,
                )
                if not ok:
                    continue

                # Token budget: approximate by counting activated payloads
                if len(activated) >= caps.max_tokens:
                    truncated = True
                    truncation_reason = "max_tokens"
                    break

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
