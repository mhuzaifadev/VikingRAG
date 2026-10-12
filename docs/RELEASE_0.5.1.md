# VikingRAG v0.5.1 release report

Focused correctness release: experience edges when documents change.

## Scope

- Transactional REPLACE / soft-delete experience retirement on every resolve path
- Revision-specific source object keys
- Stale PENDING job skip + builder endpoint/revision gates
- Snapshot rollback never reactivates `INVALIDATED` edges
- Postgres integration regression suite

Baseline: `v0.5.0` / `c4f792b`.

## Migrations / compat

- No new Alembic revision required (uses existing edge status + URI columns)
- Object keys for **new** puts use `documents/{id}/source/{content_hash}/{filename}`; existing rows keep prior keys until next REPLACE/ingest

## Commands / results

| Gate | Result |
|---|---|
| `ruff` + `mypy src` | pass |
| `pytest tests/unit tests/evaluation` | **183 passed** |
| `vikingrag-eval smoke --offline` | status ok, measured_scores null |
| `uv build` + wheel contains migrations/fixtures | pass |
| Clean uv venv `pip install` wheel + version + smoke | **0.5.1**, smoke ok |
| `make test-integration` (Postgres/Redis) | **blocked** — Docker daemon not running on release host |

## Measured / claims

- No new quality benchmarks in this release
- No superiority claim vs official AGPL tree

## Blocked

- Postgres integration suite until Docker is available (`make docker-up && make test-integration`)
- PyPI publish if `UV_PUBLISH_TOKEN` / credentials unavailable at release time
