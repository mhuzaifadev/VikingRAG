# Contributor invariants

Engineering authority: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).  
Paper intent: [arXiv:2609.11390](https://arxiv.org/abs/2609.11390).  
Parity ledger: [`docs/PAPER_PARITY.md`](docs/PAPER_PARITY.md).

Do not copy AGPL research-repo code into this tree.

## Invariants

1. **Cheap path first** — retrieve narrowly → evaluate evidence → answer if sufficient → escalate only when necessary.
2. **Authoritative Read** — final claims and citations use original source ranges, never abstracts as source.
3. **No fake production providers** — `scripted` / `fake` / `deterministic` providers are rejected when `VIKINGRAG_APP_ENV=production`.
4. **Scope semantics** — `permitted_document_ids=None` is unrestricted; empty frozenset is allow-nothing.
5. **No silent fakes** — unimplemented behavior raises a domain error or is explicitly unsupported.
6. **Deterministic where possible** — URI validation, budgets, offsets, citation checks, edge thresholds.
7. **Revision-consistent publication** — never expose new text with an obsolete vector; invalidate edges on replace/delete.
8. **Auth before use-cases** — data/query/index/answer APIs require authentication when enabled; ACL is server-derived.

## Commands

```bash
make install          # uv sync --all-extras + hooks
make lint             # ruff
make typecheck        # mypy src
make test             # unit + evaluation tests
make test-integration # Postgres + Redis
make migrate          # vikingrag-migrate upgrade head
make docker-up        # compose up
make docker-down      # compose down (volumes preserved)
make docker-reset     # compose down -v (destructive)
```

```bash
uv run vikingrag-eval smoke --offline
```

## Execution modes

| Mode | Behavior |
|---|---|
| `vikingrag` | Algorithm 1 + ordinary Search |
| `vikingrag_e` | Algorithm 1 + Search+ |
| `vikingrag_e_plus` | One-round Search+ → strict sufficiency → agentic fallback |

Paper evaluation profile (configurable): `K=10`, `L=1000`, `B=15`, `gamma=0.8`.

## Release notes

- Algorithm-complete ≠ operationally verified ≠ paper tables reproduced — report each separately.
- Unrun paid experiments stay unmeasured; do not invent scores.
