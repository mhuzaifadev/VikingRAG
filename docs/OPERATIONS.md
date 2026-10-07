# Operations

## Install from PyPI (other laptops)

```bash
pip install -U vikingrag
# set VIKINGRAG_DATABASE_URL / REDIS / LLM env (see .env.example on GitHub)
vikingrag-migrate upgrade head
vikingrag-api
```

## Local development

```bash
cp .env.example .env
make docker-up
make migrate   # vikingrag-migrate upgrade head
make dev
```

`make docker-down` stops containers and **preserves volumes**.  
`make docker-reset` destroys volumes (`docker compose down -v`).

## Pre-live checklist

Run these on a clean machine (or CI) before tagging / deploying `0.4.0` as your go-live candidate.

### 1. Offline quality (no paid APIs)

```bash
make install
make lint
make typecheck
make test
uv run vikingrag-eval smoke --offline
```

Expected: lint/typecheck clean; unit + evaluation tests green; smoke `status: "ok"` with `"measured_scores": null`.

### 2. Integration (Postgres + Redis)

```bash
make docker-up
make migrate
make test-integration
```

Expected: Alembic head applied; integration tests pass. Fail the gate if services are missing.

### 3. Local API smoke (fake providers OK for wiring)

In `.env` (never production values):

```bash
VIKINGRAG_APP_ENV=development
VIKINGRAG_LLM_PROVIDER=fake
VIKINGRAG_EMBEDDING_PROVIDER=deterministic
VIKINGRAG_AUTH_ENABLED=false
VIKINGRAG_WORKERS_EDGE_BUILDER_ENABLED=true
```

```bash
make dev
# separate terminal:
curl -s http://localhost:8000/health/live
curl -s http://localhost:8000/health/ready
# ingest → index → search → answers (see README Quick start)
```

### 4. Production-shaped stack (real providers)

```bash
# Set real keys / providers; never commit them
export VIKINGRAG_APP_ENV=production
export VIKINGRAG_LLM_PROVIDER=openai   # or deepseek | gemini | anthropic
export VIKINGRAG_EMBEDDING_PROVIDER=openai
export VIKINGRAG_AUTH_ENABLED=true
export VIKINGRAG_AUTH_API_KEY=...
export VIKINGRAG_WORKERS_EDGE_BUILDER_ENABLED=false  # use Compose sidecar
docker compose -f docker-compose.prod.yml up -d --build
```

Confirm:

- App refuses to start with `fake` / `deterministic` / `scripted` providers
- `/health/ready` is healthy after migrate
- `edge-builder` sidecar is running (or in-process worker enabled intentionally — not both)
- One authenticated `POST /v1/answers` with `execution_mode=vikingrag_e_plus` returns a cited answer or a clear abstention

### 5. SDK embed check

```python
from vikingrag import VikingRAGClient

client = VikingRAGClient.from_settings()
# client.search / client.answer_generator()
```

Close with `await client.aclose()` in async apps.

### 6. Evaluation results (honest)

| Command | Produces |
|---|---|
| `uv run vikingrag-eval smoke --offline` | Fixture integrity JSON; **not** paper scores |
| `uv run vikingrag-eval list` | Adapter catalog |
| `uv run vikingrag-eval prepare --dataset syllabusqa --download` | Blocked until you place corpora |
| `uv run vikingrag-eval warmup --dataset syllabusqa --m 1000` | Plan + **BLOCKED** until corpus + LLM |
| `uv run vikingrag-eval run` | Explicitly blocked |

Never invent Tables 3–5 numbers. Record unpaid/unrun gates as **BLOCKED**.

## Production profile

Use `docker-compose.prod.yml`:

- No source mounts / reload
- No example secrets in committed env
- Non-root process
- Migrations as an explicit job, then readiness probe
- `edge-builder` sidecar; API sets `VIKINGRAG_WORKERS_EDGE_BUILDER_ENABLED=false`

## Auth (single-tenant)

- Shared API key via `VIKINGRAG_AUTH_API_KEY`
- Document allowlist via `VIKINGRAG_AUTH_ALLOWED_DOCUMENT_IDS` (comma-separated UUIDs) or unrestricted when unset and auth-enabled
- Empty allowlist means allow-nothing when auth mode requires ACL

## Object store

`VIKINGRAG_OBJECT_STORE_BACKEND=local|s3`. Unsupported backends fail at startup.

## Vector dimensions

Storage is fixed at **1536**. Changing dimensions requires an explicit storage migration; unsupported provider dims fail validation.

## LLM / embedding providers

| `VIKINGRAG_LLM_PROVIDER` | Notes |
|---|---|
| `openai` / `openai_compatible` / `vllm` | Chat Completions + tools |
| `deepseek` / `gemini` | OpenAI-compatible presets (default base URL) |
| `anthropic` | Native Messages API + `tool_use` |
| `fake` | Dev/tests only — rejected in `production` |

## Backup / restore

1. Stop writers
2. `pg_dump` database; archive object-store root or S3 bucket
3. Restore DB then objects; run `alembic upgrade head` only forward
4. Schema rollback may require restoring a DB snapshot taken before the migration

## Health

- `GET /health/live` (and legacy aliases if present) — liveness
- `GET /health/ready` — DB + Redis (+ optional object-store) readiness
