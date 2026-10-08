# Paper parity ledger (arXiv 2609.11390v1)

Status values: `implemented` | `partial` | `deviation` | `blocked`.

Faithful algorithm behavior is separated from production extensions. Deviations are deliberate and labeled.

| Paper ref | Requirement | Code | Test | Status |
|---|---|---|---|---|
| §3.1 / Eqs 6-7 | Hierarchical storage + component-wise ancestor URI paths | `domain/uri/viking_uri.py`, `object_uri.py`, ingestion | `test_uri.py`, object URI unit | implemented |
| §3.1 | Bottom-up abstracts from owned content | `application/indexing.py` | indexing units | implemented |
| §3.1 | Distinct abstract object URIs | `domain/uri/object_uri.py` (`.abstract`) | URI unit | implemented |
| §3.2 | Search / List / Grep / Read | `application/search.py`, `*_primitive.py` | unit + integration | implemented |
| §3.2 | Search subtree `scope_uri` / path containment | `SearchRequest.scope_uri` + `path_ids @>` SQL | `test_search_scope_uri.py` | implemented |
| Alg 1 | Agentic multi-round tool retrieval | `application/agent/loop.py` | `test_agent_loop.py` | implemented |
| Alg 2 | Experience construction `U_src/U_cand/U_edge/U_tgt` | `application/experience/*`, migration `0004` | `test_experience_edges.py` | implemented |
| Alg 2 SUPPORT | LLM evidence-selection (with deterministic fallback) | `support_select.py` (`llm` \| `deterministic`) | `test_support_selector.py` | implemented |
| Alg 3 | Search+ γ-gated edge expansion | `application/search_plus.py`, `experience/expand.py` | experience + acceptance | implemented |
| §5 | VikingRAG-E+ one-round + candidate + strict sufficiency | `orchestration/query.py`, `candidate.py` | `test_eplus_strict_gate.py` | implemented |
| Supplement | Constraint-aware sufficiency | `assessment.py` v2 (entity/time/scope slots) | assessment units | implemented |
| §6 / datasets | Six corpus adapters + judge scaffolding | `evaluation/adapters/*`, CLI | offline smoke | implemented (adapters; full download **blocked** without data) |
| §6 warm-up | Historical M questions + edge warm-up | `evaluation/warmup.py`, `vikingrag-eval warmup` | unit + CLI | partial (live gen needs corpus+LLM) |
| Paper profile | K=10, L=1000, B=15, γ=0.8 | `PaperProfileSettings`, `BudgetLimits.paper_profile` | settings/auth units | implemented |
| Held-out isolation | No learning during scoring | `LearningPolicy.FROZEN` default in eval runner | `test_learning_policy.py` | implemented |
| Flat baseline | Non-experience RAG | `flat_rag` method in `runner.py` | `test_explain_trace.py` | implemented |

## Execution modes

| Mode | Behavior |
|---|---|
| `vikingrag` | Algorithm 1 + ordinary Search |
| `vikingrag_e` | Algorithm 1 + Search+ |
| `vikingrag_e_plus` | One-round Search+ → candidate answer → strict sufficiency → Alg 1 fallback |

## Production extensions (not claimed as paper)

- API-key auth + server document allowlist (single-tenant)
- Durable Postgres learning jobs (`workers/edge_builder.py`); in-process lifespan or Compose sidecar
- Multi-provider LLM presets: `openai`, `deepseek`, `gemini`, `anthropic`, `vllm`
- Library facade: `VikingRAGClient` ingest/index/ask/explain + async context manager
- `LearningPolicy` + experience snapshots; scoped edge expansion
- Offline explain API/CLI (`GET /v1/answers/{id}/explain`, `vikingrag-eval explain`)
- Rate/upload limits, redacted logs, production Compose
- Fixed vector dimension 1536 with startup validation

## Deliberate deviations

| Topic | Paper | This repo | Reason |
|---|---|---|---|
| Vector dim | provider-dependent | Fixed 1536 unless migrated | Operational simplicity |
| Prod budgets | B=15 paper eval | Tighter serving defaults; paper profile selectable | Cost/latency |
| Full 8 baselines | Comparative tables | Flat RAG + three VikingRAG modes in harness | Separate validation gate |
| Official E+ runner | Research AGPL tree | Never vendored; export row `not_run` | License + independence |
| SUPPORT default | Always LLM | Default `deterministic`; set `VIKINGRAG_RETRIEVAL_SUPPORT_SELECTOR=llm` for paper fidelity | Offline/CI safety |

## Unmeasured (until you run them)

- Full paid paper-table reproduction — requires corpora + real LLM runs
- Real-provider production smoke — requires credentials
- Official AGPL E+ pin — separate checkout only; leave `not_run` if blocked
- CI integration — wired in `.github/workflows/ci.yml` (Postgres/Redis services)
