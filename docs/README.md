# VikingRAG documentation

| Doc | What it answers |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design authority — modes, storage, retrieval, edges |
| [RETRIEVAL.md](RETRIEVAL.md) | Search / List / Grep / Read, evidence, answers |
| [PAPER_PARITY.md](PAPER_PARITY.md) | Paper § / Alg → code → test → status |
| [EVALUATION.md](EVALUATION.md) | Adapters, offline smoke, unmeasured experiments |
| [OPERATIONS.md](OPERATIONS.md) | Install, deploy, auth, backup |
| [ADR/](ADR/) | Architecture decision records |
| [assets/](assets/) | Optional README charts (drop PNGs here) |

Contributor invariants: [`AGENTS.md`](../AGENTS.md).

---

## Current release

**`0.4.3`** on PyPI as [`vikingrag`](https://pypi.org/project/vikingrag/):

```bash
pip install -U vikingrag
vikingrag-migrate upgrade head
vikingrag-api
```

Core paper modes are implemented and unit-tested. Full paper-table reproduction needs your corpora and paid providers — see [EVALUATION.md](EVALUATION.md). This is not claimed SemVer 1.0.

---

## Quick links

- Paper: [arXiv:2609.11390](https://arxiv.org/abs/2609.11390)
- Local API docs: `http://localhost:8000/docs`
- SDK: `from vikingrag import VikingRAGClient`
