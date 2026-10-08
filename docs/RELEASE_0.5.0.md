# VikingRAG v0.5.0 release report

**Do not tag / push / publish until instructed.**

## Scope

Milestones A–E from the v0.5.0 roadmap: controllable learning, SDK workflow, replayable traces, credible export harness, positioning gates.

Positioning: independent Postgres/pgvector VikingRAG with a Python SDK and explicit control over retrieval budgets and experience learning. Link [rucdatascience/VikingRAG](https://github.com/rucdatascience/VikingRAG). Paper charts ≠ this repo’s measurements. No superiority claim without regenerable exports.

## Migrations / compat

- Alembic `20261008_0005_learning_policy_snapshots`: `learning_policy`, `snapshot_id`, `corpus_id` on query runs; `experience_snapshots` table
- Existing `learn`-path behavior remains the default for production answers
- Eval held-out defaults to `frozen`

## Commands / results (offline gates)

```bash
make lint
make typecheck
make test   # 180 unit+evaluation passed locally
uv run vikingrag-eval smoke --offline   # status ok, measured_scores null
uv run pytest tests/unit/test_learning_policy.py tests/unit/test_explain_trace.py tests/unit/test_sdk_ask_policy.py -q
# Prod compose config requires secret env stubs (CI sets dummy values):
POSTGRES_PASSWORD=… VIKINGRAG_LLM_PROVIDER=… VIKINGRAG_LLM_API_KEY=… \
  VIKINGRAG_EMBEDDING_PROVIDER=… VIKINGRAG_EMBEDDING_API_KEY=… \
  VIKINGRAG_RETRIEVAL_ASSESSOR_PROVIDER=… VIKINGRAG_AUTH_API_KEY=… \
  docker compose -f docker-compose.prod.yml config -q
```

Paid / live Postgres measured scores: **blocked until providers + corpus run** — leave EVALUATION measured table blank.

Official AGPL E+: **not_run** (never vendored).

## Measured scope

| Item | Status |
|---|---|
| Fixture corpus `tests/fixtures/voca_faq/` | present |
| Flat + three VikingRAG methods | wired |
| Export JSON/CSV latency + citation rates | wired |
| Published quality numbers | **unmeasured** (fill from exports only) |
| Official E+ pin | **not_run** |

## Blocked (not passed)

- Live held-out quality scores on real LLM/embed providers
- Official research-repo E+ comparison container
- Full six paper-dataset download automation
- Tag / PyPI publish (awaiting explicit instruction)

## Files of note

- Domain: `LearningPolicy`, snapshots, scoped `expand.py`
- SDK: `client.py` ingest/index/ask/explain
- API: answer `learning_policy`, `GET …/explain`
- Eval: frozen default, `flat_rag`, exports, `explain` CLI, one-loop warmup
- Docs: README, PAPER_PARITY, EVALUATION, OPERATIONS, CHANGELOG, this report
