"""In-memory experience edge store for unit tests and local cold/warm simulation."""

from __future__ import annotations

from vikingrag.domain.models.experience import (
    ExperienceEdge,
    ExperienceEdgeId,
    ExperienceEdgeStatus,
    ExperiencePayload,
    ExperiencePayloadId,
    QueryRunId,
)
from vikingrag.domain.models.representation import EmbeddingIdentity


class InMemoryExperienceStore:
    """Thread-unsafe in-process store implementing ExperienceEdgeRepository + EdgeLookup."""

    def __init__(self) -> None:
        self.payloads: dict[ExperiencePayloadId, ExperiencePayload] = {}
        self.payloads_by_run: dict[QueryRunId, ExperiencePayloadId] = {}
        self.edges: dict[ExperienceEdgeId, ExperienceEdge] = {}

    async def get_payload_for_run(self, run_id: QueryRunId) -> ExperiencePayload | None:
        pid = self.payloads_by_run.get(run_id)
        return self.payloads.get(pid) if pid else None

    async def create_payload(self, payload: ExperiencePayload) -> ExperiencePayload:
        self.payloads[payload.id] = payload
        self.payloads_by_run[payload.query_run_id] = payload.id
        return payload

    async def create_edges(self, edges: list[ExperienceEdge]) -> list[ExperienceEdge]:
        for edge in edges:
            self.edges[edge.id] = edge
        return list(edges)

    async def find_active_by_source(
        self,
        source_uri: str,
        *,
        identity: EmbeddingIdentity,
    ) -> list[tuple[ExperienceEdge, ExperiencePayload]]:
        out: list[tuple[ExperienceEdge, ExperiencePayload]] = []
        for edge in self.edges.values():
            if edge.source_uri != source_uri:
                continue
            if edge.status is not ExperienceEdgeStatus.ACTIVE:
                continue
            payload = self.payloads.get(edge.payload_id)
            if payload is None:
                continue
            if not payload.embedding_identity.compatible_with(identity):
                continue
            out.append((edge, payload))
        return out

    async def find_active_pair(
        self,
        source_uri: str,
        target_uri: str,
        *,
        identity: EmbeddingIdentity,
    ) -> ExperienceEdge | None:
        for edge, _payload in await self.find_active_by_source(source_uri, identity=identity):
            if edge.target_uri == target_uri:
                return edge
        return None

    async def supersede(self, edge_id: ExperienceEdgeId) -> None:
        edge = self.edges.get(edge_id)
        if edge is not None:
            edge.status = ExperienceEdgeStatus.SUPERSEDED

    async def invalidate_by_uris(self, uris: set[str]) -> int:
        n = 0
        for edge in self.edges.values():
            if edge.status is not ExperienceEdgeStatus.ACTIVE:
                continue
            if edge.source_uri in uris or edge.target_uri in uris:
                edge.status = ExperienceEdgeStatus.INVALIDATED
                n += 1
        return n

    async def count_active(self) -> int:
        return sum(1 for e in self.edges.values() if e.status is ExperienceEdgeStatus.ACTIVE)
