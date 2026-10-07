# Changelog

## v0.4.0

Algorithm-complete milestone after v0.3.0 (Search / List / Grep / Read / evidence).
Not yet a claimed SemVer 1.0 stable release. Intended for PyPI as `vikingrag`.

### Added

- `pip install vikingrag` packaging: migrations in wheel + `vikingrag-migrate` CLI
- Paper parity ledger, AGENTS.md, EVALUATION.md, OPERATIONS.md, docs hub
- Execution modes: `vikingrag`, `vikingrag_e`, `vikingrag_e_plus`
- Agentic Algorithm 1 loop, answers API, experience edges (Alg 2), Search+ (Alg 3)
- Section 5 E+ candidate answer + constraint-aware strict sufficiency gate
- Search `scope_uri` subtree containment (`path_ids @>`)
- Alg 2 LLM SUPPORT selector (`VIKINGRAG_RETRIEVAL_SUPPORT_SELECTOR=llm|deterministic`)
- In-process edge builder lifespan + Compose `edge-builder` sidecar
- Multi-provider LLM presets: openai / deepseek / gemini / anthropic / vllm
- Library facade `VikingRAGClient`; eval `warmup` CLI scaffolding
- Evaluation scaffolding and offline smoke
- API-key auth + document allowlist; production Compose profile

### Fixed

- E+ soft-escalate when `assessor=None` (now wires real assessor)
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
