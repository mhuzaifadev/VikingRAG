# Operations

Deploy, configure, and verify a VikingRAG stack.

## Install from PyPI

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U vikingrag
# optional: pip install "vikingrag[docx]"
```

Copy env from the [repo `.env.example`](https://github.com/mhuzaifadev/VikingRAG/blob/main/.env.example), then:

```bash
export VIKINGRAG_DATABASE_URL=postgresql+asyncpg://vikingrag:vikingrag@localhost:5432/vikingrag
export VIKINGRAG_REDIS_URL=redis://localhost:6379/0
export VIKINGRAG_LLM_PROVIDER=fake          # development only
export VIKINGRAG_EMBEDDING_PROVIDER=deterministic

vikingrag-migrate upgrade head
vikingrag-api
# http://localhost:8000/docs
```

You still need **Postgres + pgvector** and **Redis** running (Compose below is the easiest path).

## Local development (clone)

```bash
cp .env.example .env
make docker-up
make migrate   # vikingrag-migrate upgrade head
make dev
```

- `make docker-down` — stop containers, **keep volumes**
- `make docker-reset` — destroy volumes (`docker compose down -v`)

## Deploy checklist

### 1. Offline quality (no paid APIs)

```bash
make install
make lint
make typecheck
make test
uv run vikingrag-eval smoke --offline
```

Expect: lint/typecheck clean; unit + evaluation tests green; smoke `status: "ok"` with `"measured_scores": null`.

### 2. Integration (Postgres + Redis)

```bash
make docker-up
make migrate
make test-integration
```

### 3. Local API smoke (fake providers OK for wiring)

```bash
VIKINGRAG_APP_ENV=development
VIKINGRAG_LLM_PROVIDER=fake
VIKINGRAG_EMBEDDING_PROVIDER=deterministic
VIKINGRAG_AUTH_ENABLED=false
```

```bash
make dev
curl -s http://localhost:8000/health/live
curl -s http://localhost:8000/health/ready
# ingest → index → search → answers (README Quick start)
```

### 4. Production-shaped stack (real providers)

```bash
export VIKINGRAG_APP_ENV=production
export VIKINGRAG_LLM_PROVIDER=openai   # or deepseek | gemini | anthropic
export VIKINGRAG_EMBEDDING_PROVIDER=openai
export VIKINGRAG_AUTH_ENABLED=true
export VIKINGRAG_AUTH_API_KEY=...      # never commit
export VIKINGRAG_WORKERS_EDGE_BUILDER_ENABLED=false  # use Compose sidecar
docker compose -f docker-compose.prod.yml up -d --build
```

Confirm:

- App refuses `fake` / `deterministic` / `scripted` providers in production
- `/health/ready` healthy after migrate
- `edge-builder` sidecar running (or in-process worker — not both)
- Authenticated `POST /v1/answers` with `execution_mode=vikingrag_e_plus` returns a cited answer or a clear abstention

### 5. SDK embed check

```python
import asyncio
from vikingrag import VikingRAGClient

async def main() -> None:
    client = VikingRAGClient.from_settings()
    try:
        # await client.search.search(...)
        # await client.answer_generator().generate(...)
        pass
    finally:
        await client.aclose()

asyncio.run(main())
```

### 6. Evaluation honesty

| Command | Produces |
|---|---|
| `vikingrag-eval smoke --offline` | Fixture integrity JSON — **not** paper scores |
| `vikingrag-eval list` | Adapter catalog |
| `vikingrag-eval prepare` / `warmup` / `run` | Require corpora + credentials; otherwise report as unmeasured |

Do not invent paper table numbers. See [EVALUATION.md](EVALUATION.md).

## Production profile

`docker-compose.prod.yml`:

- No source mounts / reload
- No example secrets in committed env
- Non-root process
- Migrations as an explicit job, then readiness
- `edge-builder` sidecar; API sets `VIKINGRAG_WORKERS_EDGE_BUILDER_ENABLED=false`

## Auth (single-tenant)

- Shared API key: `VIKINGRAG_AUTH_API_KEY`
- Document allowlist: `VIKINGRAG_AUTH_ALLOWED_DOCUMENT_IDS` (comma-separated UUIDs)
- Unset allowlist with auth enabled → unrestricted; empty allowlist → allow-nothing

## Object store

`VIKINGRAG_OBJECT_STORE_BACKEND=local|s3`. Unsupported backends fail at startup.

## Vector dimensions

Storage is fixed at **1536**. Changing dimensions requires an explicit storage migration.

## LLM / embedding providers

| `VIKINGRAG_LLM_PROVIDER` | Notes |
|---|---|
| `openai` / `openai_compatible` / `vllm` | Chat Completions + tools |
| `deepseek` / `gemini` | OpenAI-compatible presets |
| `anthropic` | Native Messages API + `tool_use` |
| `fake` | Dev/tests only — rejected in `production` |

## Backup / restore

1. Stop writers
2. `pg_dump` database; archive object-store root or S3 bucket
3. Restore DB then objects; run migrations forward only
4. Schema rollback needs a DB snapshot taken before the migration

## Health

- `GET /health/live` — liveness
- `GET /health/ready` — DB + Redis readiness
