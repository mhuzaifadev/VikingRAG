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

## Release status

| Step | Status |
|---|---|
| Commit on `main` | `b3483d4` |
| Tag / GitHub Release | [v0.5.1](https://github.com/mhuzaifadev/VikingRAG/releases/tag/v0.5.1) |
| PyPI `vikingrag==0.5.1` | **blocked** — `uv publish` needs credentials (`UV_PUBLISH_TOKEN` / trusted publishing) |
| Postgres integration suite | **blocked** — Docker daemon not running |

```bash
# When PyPI token is available:
uv publish
pip install vikingrag==0.5.1
python -c "import vikingrag; print(vikingrag.__version__)"
vikingrag-eval smoke --offline
```
