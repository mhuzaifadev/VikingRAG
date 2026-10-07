# AGENTS.md — VikingRAG contributor / agent invariants

Engineering authority: `docs/ARCHITECTURE.md`. Paper intent: arXiv 2609.11390v1.
Parity ledger: `docs/PAPER_PARITY.md`. Do not copy AGPL research-repo code.

## Invariants

1. **Cheap path first** — retrieve narrowly → evaluate evidence → answer if sufficient → escalate only when necessary.
2. **Authoritative Read** — final claims and citations use original source ranges, never abstracts masquerading as source.
3. **No fake production providers** — `scripted` / `fake` / `deterministic` LLM, embedding, or assessor providers are rejected when `VIKINGRAG_APP_ENV=production`.
4. **Scope semantics** — `permitted_document_ids=None` is unrestricted; empty frozenset is allow-nothing (short-circuit before embed/SQL).
5. **No silent fakes** — unimplemented behavior raises a domain error or is explicitly unsupported.
6. **Deterministic where possible** — URI validation, budgets, offsets, citation ID checks, edge activation thresholds.
7. **Revision-consistent publication** — never expose new text with an obsolete vector; invalidate edges on endpoint replace/delete.
8. **Auth before use-cases** — data/query/index/trace/answer APIs require authentication; ACL is server-derived.

## Commands

```bash
make install          # uv sync + hooks
make lint             # ruff check + format check
make typecheck        # mypy src
make test             # unit tests
make test-integration # integration (requires Postgres/Redis)
make migrate          # alembic upgrade head
make docker-up        # compose up
make docker-down      # compose down (volumes preserved)
make docker-reset     # compose down -v (destructive)
```

Offline evaluation smoke:

```bash
uv run vikingrag-eval smoke --offline
```

## Execution modes

| Mode | Behavior |
|---|---|
| `vikingrag` | Algorithm 1 + ordinary Search |
| `vikingrag_e` | Algorithm 1 + Search+ (experience edges) |
| `vikingrag_e_plus` | One-round Search+ → strict sufficiency → agentic fallback if needed |

Paper evaluation profile (configurable): `K=10`, `L=1000`, `B=15`, `gamma=0.8`.

## Release gates

Do not mark skipped/failed/absent checks as passed.

1. `make lint` / `make typecheck` / `make test` / `make test-integration`
2. Clean-DB Alembic migrate + upgrade-from-v0.3
3. Offline evaluation smoke
4. Real-provider / production smoke only when configured; otherwise **BLOCKED**
5. Algorithm complete ≠ operationally verified ≠ experimentally reproduced — report separately

Authorship on this repo is always the owner (`mhuzaifadev`), never Cursor Agent.
