# VikingRAG documentation

Public engineering docs for the production platform. Start here, then drill into a topic.

| Doc | What it answers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design authority — modes, storage, retrieval, edges |
| [RETRIEVAL.md](RETRIEVAL.md) | Search / List / Grep / Read, evidence, cheap path |
| [PAPER_PARITY.md](PAPER_PARITY.md) | Paper § / Alg → code → test → status |
| [EVALUATION.md](EVALUATION.md) | Adapters, offline smoke, what is BLOCKED |
| [OPERATIONS.md](OPERATIONS.md) | Deploy, auth, backup, pre-live checklist |
| [ADR/](ADR/) | Architecture decision records |

Contributor invariants and release gates: [`AGENTS.md`](../AGENTS.md) (repo root).

---

## Release status

This tree is **`0.4.0`** (algorithm-complete milestone after 0.3.0). That means:

- Core paper modes are implemented and unit-tested
- Full paper-table reproduction and paid-provider smoke stay **BLOCKED** until you run them with real data/credentials
- Do not treat this as SemVer 1.0 / “stable forever” until the [pre-live checklist](OPERATIONS.md#pre-live-checklist) passes on your stack
- Install: `pip install vikingrag` (migrations via `vikingrag-migrate`)

---

## Quick links

- Paper: [arXiv:2609.11390](https://arxiv.org/abs/2609.11390)
- Interactive API (local): `http://localhost:8000/docs`
- SDK entry: `from vikingrag import VikingRAGClient`
