# Evaluation

Harness for paper-shaped corpora and method comparison. Measured scores are only emitted from real runs — never invent them.

Paper-cited charts in the README are **paper results**, not measurements from this repository.

## Datasets

Adapters (primary sources, checksums, immutable revisions):

- VersionQA
- SyllabusQA
- QASPER
- HotpotQA
- LegalBench-CUAD
- FinanceBench

Untouched original data is retained; adapter transforms are documented per dataset.

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
- `echo` — wiring-only placeholder (`--method echo`)

Use identical backbone prompts/budgets when comparing; record unavoidable deviations in run manifests.

Optional `--judge scripted` records deterministic `non_empty_answer_rate` and `citation_presence_rate` only — **not** LLM answer accuracy.

## Warm-up rules

- Historical questions only from corpus evidence (never eval questions, gold answers, or paraphrases)
- Source text from `documents/` / `corpus/` only
- After generation, warm-up runs `vikingrag_e_plus` (or `vikingrag_e`) and drains the edge builder so experience edges materialize
- Freeze edge store before held-out scoring
- Disable learning during scoring
- Isolate per-method/corpus indexes and histories
- Paper default M=1000; smaller runs must be labeled **smoke** (`--smoke-max`)

## Commands

```bash
vikingrag-eval list
vikingrag-eval smoke --offline
vikingrag-eval prepare --dataset syllabusqa --verify
vikingrag-eval warmup --dataset syllabusqa --m 100 --smoke-max 10
vikingrag-eval run --manifest path/to/manifest.json
vikingrag-eval run --manifest path/to/manifest.json \
  --method vikingrag,vikingrag_e,vikingrag_e_plus \
  --judge scripted --output report.json
```

From a clone with `uv`:

```bash
uv run vikingrag-eval smoke --offline
```

Offline smoke uses `tests/fixtures/system_architecture.md` as a syllabus-like corpus. It validates scaffolding only (`measured_scores` is always `null`). It is **not** a paper-table result.

`prepare --download` raises a clear `dataset_not_available` until each corpus is provisioned with checksums under `data/eval/<name>/`.

Full paper experiments need corpora + `VIKINGRAG_EVAL_*` / provider credentials; otherwise leave status unmeasured.

CI runs `vikingrag-eval smoke --offline` and `pytest tests/evaluation`.

## Measured results (this repo)

Status: **unmeasured** until you fill a row from a real run. Do not copy paper table numbers here.

| Field | Value |
|---|---|
| Modes | `vikingrag` / `vikingrag_e` / `vikingrag_e_plus` |
| Corpus | _(dataset + revision)_ |
| N (questions) | |
| Model (LLM / embed) | |
| Hardware | |
| Cold / warm | |
| p50 latency (ms) | |
| p95 latency (ms) | |
| Tokens (in/out) | |
| Quality (judge scale / notes) | |
| Citation validity rate | |
| Date / commit | |
| Notes | Leave blank cells empty; never invent |

### Measure checklist

1. Provision corpus under `data/eval/<dataset>/corpus/` (or `documents/`)
2. `vikingrag-eval warmup --dataset … --m …` (materialize edges; `--smoke-max` for small runs)
3. Freeze edge store; disable learning for scoring
4. `vikingrag-eval run --manifest … --method vikingrag,vikingrag_e,vikingrag_e_plus`
5. Optionally `--judge scripted` for citation/non-empty rates
6. Record hardware, models, cold/warm, latency, tokens, and quality into the table above

## Aggregation

Judge scores use the supplement’s 0–4 gold-answer scale with independently written prompts. Aggregation semantics are declared per export; no implicit accuracy threshold is invented.
