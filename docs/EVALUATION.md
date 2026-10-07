# Evaluation

Harness for paper-shaped corpora and method comparison. **Never invent measured scores.**

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

Identical backbone prompts/budgets when comparing; unavoidable deviations recorded in run manifests.

## Warm-up rules

- Historical questions only from corpus evidence (never eval questions, gold answers, or paraphrases thereof)
- Freeze edge store before held-out scoring
- Disable learning during scoring
- Isolate per-method/corpus indexes and histories
- Paper default M=1000; smaller runs must be labeled **smoke**

## Commands

```bash
uv run vikingrag-eval list
uv run vikingrag-eval smoke --offline
uv run vikingrag-eval prepare --dataset syllabusqa --verify
uv run vikingrag-eval run --manifest path/to/manifest.json
```

Offline smoke uses `tests/fixtures/system_architecture.md` as a syllabus-like corpus. It validates scaffolding integrity only (`measured_scores` is always `null`). It is **not** a paper-table result.

Adapter `prepare --download` currently raises a clear `dataset_not_available` error until each corpus is provisioned with checksums under `data/eval/<name>/`. Never invent scores when data is missing.

Real-provider and full paper experiments run only when resources and `VIKINGRAG_EVAL_*` settings permit; otherwise status is **BLOCKED/unmeasured**.

CI runs `vikingrag-eval smoke --offline` and `pytest tests/evaluation`.

## Aggregation

Judge scores use the supplement’s 0–4 gold-answer scale with independently written prompts. Aggregation semantics are declared per export; no implicit accuracy threshold is invented.
