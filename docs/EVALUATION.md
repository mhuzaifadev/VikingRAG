# Evaluation

Harness for paper-shaped corpora and method comparison. Measured scores are only emitted from real runs — never invent them.

Paper-cited charts in the README are **paper results**, not measurements from this repository.

## Positioning

This repo is an **independent Postgres/pgvector** VikingRAG. The official research tree ([rucdatascience/VikingRAG](https://github.com/rucdatascience/VikingRAG), AGPL) is linked for attribution only and is **never vendored**. Official E+ rows in exports default to `not_run` unless you pin a separate checkout/container.

## Datasets

Adapters (primary sources, checksums, immutable revisions):

- VersionQA
- SyllabusQA
- QASPER
- HotpotQA
- LegalBench-CUAD
- FinanceBench

Untouched original data is retained; adapter transforms are documented per dataset.

### Fixture workload (v0.5.0)

Sanitized FAQ/policy corpus for local comparison:

```text
tests/fixtures/voca_faq/
  corpus/                 # cancellation.md, billing.md, support.md
  historical_questions.json
  heldout_manifest.json   # gold ≠ historical; learning_policy_for_scoring=frozen
```

Example SDK path: `examples/voca_faq/`.

### Source document layout (warm-up / indexing)

Place plain-text corpus files under the adapter data root:

```text
data/eval/<dataset>/
  corpus/          # or documents/
    *.md | *.txt
  qa.json          # gold — never read as warm-up corpus
  *.jsonl          # gold — never read as warm-up corpus
```

Adapters expose `iter_source_documents()` over `documents/` and `corpus/` (`.md`/`.txt` only). Gold QA JSON/JSONL is ignored for historical-question generation.

## Methods

- `vikingrag` — Algorithm 1 + Search (default `vikingrag-eval run`)
- `vikingrag_e` — Algorithm 1 + Search+
- `vikingrag_e_plus` — Section 5 one-round Search+ with agentic fallback
- `flat_rag` — ordinary Search baseline (no experience edges; instructions forbid edge use)
- `echo` — wiring-only placeholder (`--method echo`)

Held-out scoring defaults to `LearningPolicy.FROZEN` so edge counts cannot drift during measurement.

Use identical backbone prompts/budgets when comparing; record unavoidable deviations in run manifests.

Optional `--judge scripted` records deterministic `non_empty_answer_rate` and `citation_presence_rate` only — **not** LLM answer accuracy.

Malformed `document_ids` in a manifest fail closed before any provider call. Run status is `failed` / `partial` / `completed` / `completed_unjudged`; the CLI exits nonzero on `failed` or `partial`.

## Warm-up rules

- Historical questions only from corpus evidence (never eval questions, gold answers, or paraphrases)
- Source text from `documents/` / `corpus/` only
- Materialize with `learning_policy=learn` via E+/E answers + edge-builder drain (one asyncio loop)
- Freeze snapshot / set policy `frozen` before held-out scoring
- Isolate per-method/corpus indexes and histories
- Paper default M=1000; smaller runs must be labeled **smoke** (`--smoke-max`)
- Fail if `materialize=True` but no client/answer path is available

## Commands

```bash
vikingrag-eval list
vikingrag-eval smoke --offline
vikingrag-eval prepare --dataset syllabusqa --verify
vikingrag-eval warmup --dataset syllabusqa --m 100 --smoke-max 10
vikingrag-eval run --manifest tests/fixtures/voca_faq/heldout_manifest.json \
  --method flat_rag,vikingrag,vikingrag_e,vikingrag_e_plus \
  --judge scripted --output report.json --csv report.csv
vikingrag-eval explain --query-id <experience_query_run_id>
```

From a clone with `uv`:

```bash
uv run vikingrag-eval smoke --offline
```

Offline smoke uses `tests/fixtures/system_architecture.md` as a syllabus-like corpus. It validates scaffolding only (`measured_scores` is always `null`). It is **not** a paper-table result.

`prepare --download` raises a clear `dataset_not_available` until each corpus is provisioned with checksums under `data/eval/<name>/`.

Full paper experiments need corpora + `VIKINGRAG_EVAL_*` / provider credentials; otherwise leave status unmeasured.

CI runs `vikingrag-eval smoke --offline` and `pytest tests/evaluation`.

## Pre-declared quality tolerances

Before a scored run, record tolerances in the manifest (see `heldout_manifest.json`):

| Tolerance | Fixture default | Meaning |
|---|---|---|
| `non_empty_answer_rate_min` | `0.0` | Raise before claiming quality |
| `citation_presence_rate_min` | `0.0` | Raise before claiming quality |

Do not invent post-hoc thresholds after looking at numbers.

## Measured results (this repo)

Status: **unmeasured** until you fill a row from a regenerable export. Do not copy paper table numbers here.

| Field | Value |
|---|---|
| Modes | `flat_rag` / `vikingrag` / `vikingrag_e` / `vikingrag_e_plus` |
| Corpus | voca_faq fixture-v1 _(or dataset + revision)_ |
| N (questions) | |
| Model (LLM / embed) | |
| Hardware | |
| Cold / warm | warm after snapshot freeze |
| Learning during score | `frozen` |
| p50 latency (ms) | _(from `export_metrics`)_ |
| p95 latency (ms) | |
| Tokens (in/out) | |
| Quality (judge scale / notes) | |
| Citation validity rate | |
| Official E+ | `not_run` (AGPL not vendored) |
| Date / commit | |
| Notes | Leave blank cells empty; never invent |

### Measure checklist

1. Provision corpus under `data/eval/<dataset>/corpus/` (or use `tests/fixtures/voca_faq/`)
2. `vikingrag-eval warmup …` with materialize (`learn`); optionally freeze a snapshot
3. Score with held-out manifest + `frozen` (runner default)
4. `vikingrag-eval run --manifest … --method flat_rag,vikingrag,vikingrag_e,vikingrag_e_plus --judge scripted --output … --csv …`
5. Confirm job count / active-edge count unchanged vs freeze point
6. Copy `export_metrics` + `measured_scores` into the table above — never invent cells

## Aggregation

Judge scores use the supplement’s 0–4 gold-answer scale with independently written prompts when an LLM judge is wired. Aggregation semantics are declared per export; no implicit accuracy threshold is invented. Scripted judge = citation / non-empty rates only.
