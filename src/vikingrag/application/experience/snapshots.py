"""Experience snapshot create / freeze / rollback (in-memory + SQL-backed)."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from vikingrag.domain.models.experience import (
    ExperienceEdge,
    ExperienceEdgeId,
    ExperienceEdgeStatus,
    ExperienceSnapshot,
    ExperienceSnapshotId,
    SnapshotStatus,
    new_experience_snapshot_id,
)
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


class SnapshotEdgeSource(Protocol):
    async def list_active_edges(self) -> list[ExperienceEdge]: ...

    async def set_edge_status(
        self, edge_id: ExperienceEdgeId, status: ExperienceEdgeStatus
    ) -> None: ...


class InMemorySnapshotStore:
    """Unit-test snapshot registry."""

    def __init__(self) -> None:
        self.snapshots: dict[ExperienceSnapshotId, ExperienceSnapshot] = {}
        self.by_name: dict[str, ExperienceSnapshotId] = {}

    async def create(
        self,
        *,
        name: str,
        edges: list[ExperienceEdge],
        corpus_id: UUID | None = None,
    ) -> ExperienceSnapshot:
        snap = ExperienceSnapshot(
            id=new_experience_snapshot_id(),
            name=name,
            status=SnapshotStatus.ACTIVE,
            corpus_id=corpus_id,
            edge_ids=tuple(e.id for e in edges),
            payload_ids=tuple(sorted({e.payload_id for e in edges}, key=str)),
            metadata={"edge_count": len(edges)},
        )
        self.snapshots[snap.id] = snap
        self.by_name[name] = snap.id
        return snap

    async def get(self, snapshot_id: ExperienceSnapshotId) -> ExperienceSnapshot | None:
        return self.snapshots.get(snapshot_id)

    async def get_by_name(self, name: str) -> ExperienceSnapshot | None:
        sid = self.by_name.get(name)
        return self.snapshots.get(sid) if sid else None

    async def freeze(self, snapshot_id: ExperienceSnapshotId) -> ExperienceSnapshot:
        snap = self.snapshots[snapshot_id]
        snap.status = SnapshotStatus.FROZEN
        logger.info("experience_snapshot_frozen", snapshot_id=str(snapshot_id), name=snap.name)
        return snap

    async def rollback(
        self,
        snapshot_id: ExperienceSnapshotId,
        *,
        edges: SnapshotEdgeSource,
    ) -> ExperienceSnapshot:
        """Restore active set to snapshot membership without re-embedding."""
        snap = self.snapshots[snapshot_id]
        keep = set(snap.edge_ids)
        active = await edges.list_active_edges()
        for edge in active:
            if edge.id not in keep:
                await edges.set_edge_status(edge.id, ExperienceEdgeStatus.SUPERSEDED)
        # Reactivate snapshot edges that were superseded
        for edge_id in snap.edge_ids:
            await edges.set_edge_status(edge_id, ExperienceEdgeStatus.ACTIVE)
        snap.status = SnapshotStatus.ACTIVE
        logger.info("experience_snapshot_rollback", snapshot_id=str(snapshot_id))
        return snap
