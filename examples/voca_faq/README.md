# Voca FAQ example (sanitized policy docs)

Minimal SDK path: ingest → index → ask → explain → close.

Uses the fixture corpus under `tests/fixtures/voca_faq/`. **No measured scores** are claimed here — run your own eval exports for numbers.

## Prerequisites

- Postgres + pgvector, Redis
- Env: `VIKINGRAG_DATABASE_URL`, `VIKINGRAG_REDIS_URL`, LLM + embedding providers
- Migrations applied: `vikingrag-migrate upgrade head`

## Run

```bash
# from repo root
uv run python examples/voca_faq/run.py
```

Held-out scoring should use `learning_policy="frozen"` (see `docs/EVALUATION.md`).
