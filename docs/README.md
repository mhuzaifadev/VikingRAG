# VikingRAG documentation

| Doc | What it answers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design authority — modes, storage, retrieval, edges |
| [RETRIEVAL.md](RETRIEVAL.md) | Search / List / Grep / Read, evidence, answers |
| [PAPER_PARITY.md](PAPER_PARITY.md) | Paper § / Alg → code → test → status |
| [EVALUATION.md](EVALUATION.md) | Adapters, offline smoke, unmeasured experiments |
| [OPERATIONS.md](OPERATIONS.md) | Install, deploy, auth, backup |
| [CHANGELOG.md](../CHANGELOG.md) | Release history |
| [RELEASE_0.5.0.md](RELEASE_0.5.0.md) | v0.5.0 release gate report (measured vs blocked) |
| [ADR/](ADR/) | Architecture decision records |
| [assets/](assets/) | Optional README charts (drop PNGs here) |

Contributor invariants: [`AGENTS.md`](../AGENTS.md). Internal planning/comparison specs are gitignored and not shipped as product docs.

---

## Current release

**`0.5.0`** — independent Postgres/pgvector VikingRAG with controllable learning, SDK lifecycle (`ingest` / `index` / `ask` / `explain`), and honest eval exports. PyPI package: [`vikingrag`](https://pypi.org/project/vikingrag/).

```bash
pip install -U vikingrag
vikingrag-migrate upgrade head
vikingrag-api
```

Official research repo: [rucdatascience/VikingRAG](https://github.com/rucdatascience/VikingRAG) (AGPL — not vendored). Paper charts ≠ this repo’s measurements — see [EVALUATION.md](EVALUATION.md). Not claimed SemVer 1.0.

---

## Quick links

- Paper: [arXiv:2609.11390](https://arxiv.org/abs/2609.11390)
- Local API docs: `http://localhost:8000/docs`
- SDK: `async with VikingRAGClient.from_settings() as rag:`
- Example: [`examples/voca_faq/`](../examples/voca_faq/)
