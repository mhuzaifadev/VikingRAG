# Changelog

## v0.4.2

Correctness hotfix after PyPI `0.4.1` review (scope, E+, Compose, packaging).

### Fixed

- Answer modes honor `document_ids` (intersect with auth allowlist across Search/List/Grep/Read)
- E+ no longer runs Algorithm 1 twice on escalation; `initial_evidence` / gaps / instructions are carried
- E+ one-round path returns citations + usage and enqueues experience learning
- Search+ traces emit separate `SEARCH` vs `EDGE_EXPAND` events for Algorithm 2
- E+ assessor receives the candidate answer for claim-aware sufficiency
- Production Compose: migrate/worker process roles; allowlist `*` = unrestricted; model/base-URL env forwarded
- Provider model defaults follow presets when `VIKINGRAG_LLM_MODEL` is empty (Gemini → `gemini-2.5-flash`)
- Wheel ships offline smoke fixture; CI installs the wheel outside the repo

### Added

- Executable warm-up when corpus + LLM are present; `vikingrag-eval run --manifest` records unjudged results

### Changed

- README: production-oriented wording; Docker vs `make dev` split; working SDK example

## v0.4.1

Docs and packaging polish after the first PyPI publish of `vikingrag`.

### Changed

- Public docs pass: architecture intro (removed coding-agent prompts), retrieval/ops/eval hubs
- README Quick start includes `POST /v1/answers`; SDK example is async-safe
- Pre-commit mypy uses the project venv (`uv run mypy src`)

### Fixed

- Invalid YAML indentation in `.pre-commit-config.yaml`

## v0.4.0

First PyPI release (`pip install vikingrag`). Algorithm-complete after v0.3.0
(Search / List / Grep / Read / evidence). Not claimed SemVer 1.0.

### Added

- Wheel packaging: Alembic migrations + `vikingrag-migrate` CLI
- Docs hub, paper parity ledger, AGENTS.md, EVALUATION.md, OPERATIONS.md
- Execution modes: `vikingrag`, `vikingrag_e`, `vikingrag_e_plus`
- Agentic Algorithm 1, answers API, experience edges (Alg 2), Search+ (Alg 3)
- Section 5 E+ candidate answer + constraint-aware strict sufficiency gate
- Search `scope_uri` subtree containment (`path_ids @>`)
- Alg 2 LLM SUPPORT selector (`VIKINGRAG_RETRIEVAL_SUPPORT_SELECTOR`)
- In-process edge builder + Compose `edge-builder` sidecar
- Multi-provider LLM presets: openai / deepseek / gemini / anthropic / vllm
- Library facade `VikingRAGClient`; eval `warmup` CLI scaffolding
- API-key auth + document allowlist; production Compose profile

### Fixed

- E+ soft-escalate when assessor was missing
- Production rejection of fake/scripted providers
- Empty allowlist vs unrestricted scope semantics
- Exact Read token caps; Grep cursor/Unicode; atomic budget reserve
- Summary fingerprint vs content hash; assessor truncation disclosure
- Object-store backend selection; non-destructive `docker-down`

## v0.3.0

Retrieval primitives (List/Grep/Read) and preliminary evidence sufficiency.

## v0.2.0

Multi-granular abstracts, embeddings, semantic Search.

## v0.1.0

Hierarchical ingestion foundation.
