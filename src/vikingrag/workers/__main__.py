"""Run the edge-builder worker as a sidecar process.

Usage::

    uv run python -m vikingrag.workers

Prefer either the in-process API worker OR this sidecar — not both against
the same database unless you accept concurrent claim races (claim is atomic).
"""

from __future__ import annotations

import asyncio
import contextlib
import signal

from vikingrag.application.experience.support_select import build_support_selector
from vikingrag.infrastructure.database.engine import create_database
from vikingrag.observability.logging import configure_logging, get_logger
from vikingrag.providers.factory import build_llm_provider
from vikingrag.settings.config import get_settings
from vikingrag.workers.edge_builder import EdgeBuilderWorker

logger = get_logger(__name__)


async def _amain() -> None:
    settings = get_settings()
    configure_logging(level=settings.app.log_level, json_logs=settings.app.env != "development")
    database = create_database(settings.database)
    llm = None
    try:
        if settings.retrieval.support_selector == "llm":
            llm = build_llm_provider(settings)
    except Exception as exc:
        logger.warning("worker_llm_unavailable", error=str(exc))
    selector = build_support_selector(
        mode=settings.retrieval.support_selector,
        llm=llm,
        model=settings.llm.model,
        max_support=settings.retrieval.experience_max_support,
    )
    worker = EdgeBuilderWorker(
        database=database,
        poll_interval_seconds=settings.workers.edge_builder_poll_seconds,
        batch_size=settings.workers.edge_builder_batch_size,
        max_support=settings.retrieval.experience_max_support,
        support_selector=selector,
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, worker.request_stop)

    try:
        await worker.run_forever()
    finally:
        await database.dispose()
        close = getattr(llm, "aclose", None)
        if close is not None:
            await close()


def main() -> None:
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
