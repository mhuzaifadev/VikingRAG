# Evaluation

Harness for paper-shaped corpora and method comparison. Measured scores are only emitted from real runs — never invent them.

## Datasets

Adapters (primary sources, checksums, immutable revisions):

- VersionQA
- SyllabusQA
- QASPER
- HotpotQA
- LegalBench-CUAD
- FinanceBench

Untouched original data is retained; adapter transforms are documented per dataset.

## Methods

- `flat_rag` — baseline
- `vikingrag` — Algorithm 1 + Search
- `vikingrag_e` — Algorithm 1 + Search+
- `vikingrag_e_plus` — Section 5 route

Use identical backbone prompts/budgets when comparing; record unavoidable deviations in run manifests.

## Warm-up rules

- Historical questions only from corpus evidence (never eval questions, gold answers, or paraphrases)
- Freeze edge store before held-out scoring
- Disable learning during scoring
- Isolate per-method/corpus indexes and histories
- Paper default M=1000; smaller runs must be labeled **smoke**

## Commands

```bash
vikingrag-eval list
vikingrag-eval smoke --offline
vikingrag-eval prepare --dataset syllabusqa --verify
vikingrag-eval run --manifest path/to/manifest.json
```

From a clone with `uv`:

```bash
uv run vikingrag-eval smoke --offline
```

Offline smoke uses `tests/fixtures/system_architecture.md` as a syllabus-like corpus. It validates scaffolding only (`measured_scores` is always `null`). It is **not** a paper-table result.

`prepare --download` raises a clear `dataset_not_available` until each corpus is provisioned with checksums under `data/eval/<name>/`.

Full paper experiments need corpora + `VIKINGRAG_EVAL_*` / provider credentials; otherwise leave status unmeasured.

CI runs `vikingrag-eval smoke --offline` and `pytest tests/evaluation`.

## Aggregation

Judge scores use the supplement’s 0–4 gold-answer scale with independently written prompts. Aggregation semantics are declared per export; no implicit accuracy threshold is invented.
