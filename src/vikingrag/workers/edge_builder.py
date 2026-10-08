"""Durable Algorithm-2 edge builder - polls query_runs.edge_build_status.

Not fire-and-forget asyncio.create_task. Jobs survive process restarts because
state lives on ``query_runs.edge_build_status`` (PENDING -> PROCESSING -> COMPLETED/FAILED).
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import Any

from vikingrag.application.experience.builder import ExperienceEdgeBuilder
from vikingrag.application.experience.policy import worker_may_build
from vikingrag.domain.models.experience import EdgeBuildStatus, QueryRunStatus
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.experience import (
    SqlExperienceEdgeRepository,
    SqlQueryRunRepository,
    SqlRetrievalEventRepository,
)
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)

SessionFactory = Callable[[], Any]


class EdgeBuilderWorker:
    """Poll DB for PENDING edge-build jobs and process them durably."""

    def __init__(
        self,
        *,
        database: Database,
        poll_interval_seconds: float = 2.0,
        batch_size: int = 5,
        max_support: int = 16,
        support_selector: Any | None = None,
    ) -> None:
        self._database = database
        self._poll_interval = poll_interval_seconds
        self._batch_size = batch_size
        self._max_support = max_support
        self._support_selector = support_selector
        self._stop = asyncio.Event()

    def request_stop(self) -> None:
        self._stop.set()

    async def run_forever(self) -> None:
        logger.info("edge_builder_worker_started", poll_interval=self._poll_interval)
        while not self._stop.is_set():
            try:
                processed = await self.poll_once()
                if processed == 0:
                    with contextlib.suppress(TimeoutError):
                        await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval)
            except Exception as exc:
                logger.exception("edge_builder_poll_error", error=str(exc))
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._stop.wait(), timeout=self._poll_interval)
        logger.info("edge_builder_worker_stopped")

    async def poll_once(self) -> int:
        """Claim and process up to ``batch_size`` PENDING jobs. Returns count processed."""
        async with self._database.session() as session:
            runs_repo = SqlQueryRunRepository(session)
            events_repo = SqlRetrievalEventRepository(session)
            edges_repo = SqlExperienceEdgeRepository(session)
            builder = ExperienceEdgeBuilder(
                runs=runs_repo,
                events=events_repo,
                edges=edges_repo,
                max_support=self._max_support,
                support_selector=self._support_selector,
            )
            claimed = await runs_repo.claim_pending_edge_jobs(limit=self._batch_size)
            if not claimed:
                return 0

            done = 0
            for run in claimed:
                if not worker_may_build(run):
                    run.edge_build_status = EdgeBuildStatus.SKIPPED
                    run.edge_build_error = (
                        f"learning_policy_{run.learning_policy.value}_refused_by_worker"
                    )
                    await runs_repo.update(run)
                    logger.info(
                        "edge_builder_job_refused",
                        run_id=str(run.id),
                        learning_policy=run.learning_policy.value,
                    )
                    done += 1
                    continue
                try:
                    result = await builder.build_for_run(run)
                    logger.info(
                        "edge_builder_job_done",
                        run_id=str(run.id),
                        skipped=result.skipped,
                        edges=len(result.edges_created),
                        suppressed=result.edges_suppressed,
                    )
                    done += 1
                except Exception as exc:
                    logger.exception(
                        "edge_builder_job_failed",
                        run_id=str(run.id),
                        error=str(exc),
                    )
                    run.edge_build_status = EdgeBuildStatus.FAILED
                    run.edge_build_error = str(exc)[:2_000]
                    if run.status is not QueryRunStatus.SUCCEEDED:
                        run.status = QueryRunStatus.SUCCEEDED
                    await runs_repo.update(run)
            return done


async def process_pending_edge_jobs(
    database: Database,
    *,
    batch_size: int = 5,
) -> int:
    """One-shot helper for tests / CLI - does not start a long-running loop."""
    worker = EdgeBuilderWorker(database=database, batch_size=batch_size)
    return await worker.poll_once()
