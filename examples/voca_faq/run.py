#!/usr/bin/env python3
"""One-loop SDK demo: ingest → index → ask → explain → aclose."""

from __future__ import annotations

import asyncio
from pathlib import Path

from vikingrag import VikingRAGClient

FIXTURE = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "voca_faq" / "corpus"


async def main() -> None:
    docs = sorted(FIXTURE.glob("*.md"))
    if not docs:
        raise SystemExit(f"No corpus markdown under {FIXTURE}")

    async with VikingRAGClient.from_settings() as rag:
        refs = []
        for path in docs:
            ref = await rag.ingest(path)
            await rag.index(ref.id)
            refs.append(ref)
            print(f"indexed {path.name} id={ref.id} skipped={ref.skipped}")

        answer = await rag.ask(
            "What is the cancellation window for a Standard plan?",
            document_ids=[r.id for r in refs],
            mode="vikingrag",
            learning_policy="record_only",
        )
        print(f"status={answer.status} route={answer.route}")
        print((answer.answer or "")[:400])
        run_id = answer.metadata.get("experience_query_run_id") or str(answer.query_id)
        explanation = await rag.explain(run_id)
        print(
            f"explain learning_policy={explanation.get('learning_policy')} "
            f"edges={len(explanation.get('contributing_edges') or [])} "
            f"replay={explanation.get('replay')}"
        )


if __name__ == "__main__":
    asyncio.run(main())
