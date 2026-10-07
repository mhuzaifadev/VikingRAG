<p align="center">
  <a href="https://github.com/mhuzaifadev/VikingRAG">
    <img src="https://img.shields.io/badge/VikingRAG-0.3.0-1f6feb?style=for-the-badge&labelColor=0d1117" alt="VikingRAG 0.3.0" />
  </a>
</p>

<h1 align="center">VikingRAG</h1>

<p align="center">
  <em>Structure-aware retrieval for documents that actually have structure.</em>
</p>

<p align="center">
  Production hierarchical RAG inspired by the
  <a href="https://arxiv.org/abs/2609.11390"><strong>VikingRAG paper</strong></a>.<br/>
  Hierarchy-preserving ingestion. Multi-granular semantic Search. Built to extend.
</p>

<p align="center">
  <a href="https://github.com/mhuzaifadev/VikingRAG/stargazers"><img src="https://img.shields.io/github/stars/mhuzaifadev/VikingRAG?style=social" alt="GitHub stars" /></a>
  &nbsp;
  <a href="https://github.com/mhuzaifadev/VikingRAG/network/members"><img src="https://img.shields.io/github/forks/mhuzaifadev/VikingRAG?style=social" alt="GitHub forks" /></a>
  &nbsp;
  <a href="https://github.com/mhuzaifadev/VikingRAG/watchers"><img src="https://img.shields.io/github/watchers/mhuzaifadev/VikingRAG?style=social" alt="GitHub watchers" /></a>
</p>

<p align="center">
  <a href="https://github.com/mhuzaifadev/VikingRAG/releases/tag/v0.3.0"><img src="https://img.shields.io/badge/release-v0.3.0-blue?logo=github" alt="Release v0.3.0" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?logo=apache" alt="Apache 2.0" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white" alt="FastAPI" /></a>
  <a href="https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL + pgvector" /></a>
  <a href="https://redis.io/"><img src="https://img.shields.io/badge/Redis-ready-DC382D?logo=redis&logoColor=white" alt="Redis" /></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/Ruff-lint-261230?logo=ruff&logoColor=white" alt="Ruff" /></a>
  <a href="https://arxiv.org/abs/2609.11390"><img src="https://img.shields.io/badge/arXiv-2609.11390-b31b1b?logo=arxiv&logoColor=white" alt="arXiv paper" /></a>
</p>

<p align="center">
  <a href="#features"><strong>Features</strong></a>
  ·
  <a href="#architecture"><strong>Architecture</strong></a>
  ·
  <a href="#quick-start"><strong>Quick start</strong></a>
  ·
  <a href="#api"><strong>API</strong></a>
  ·
  <a href="#citation"><strong>Cite</strong></a>
  ·
  <a href="#license"><strong>License</strong></a>
</p>

<p align="center">
  <a href="https://www.linkedin.com/in/mhuzaifadev">
    <img src="https://img.shields.io/badge/LinkedIn-mhuzaifadev-0A66C2?style=flat-square&logo=linkedin&logoColor=white" alt="LinkedIn" />
  </a>
  &nbsp;
  <a href="https://mhuzaifa.com">
    <img src="https://img.shields.io/badge/Website-mhuzaifa.com-111111?style=flat-square&logo=googlechrome&logoColor=white" alt="mhuzaifa.com" />
  </a>
  &nbsp;
  <a href="https://github.com/mhuzaifadev">
    <img src="https://img.shields.io/badge/GitHub-mhuzaifadev-181717?style=flat-square&logo=github&logoColor=white" alt="GitHub" />
  </a>
</p>

---

## Why VikingRAG?

Classic RAG often flattens documents into unrelated chunks. **Structure is lost** - sections, subsections, and provenance disappear into a top-k bag of text.

**VikingRAG** (Gao et al., 2026) showed that hierarchy-preserving storage, multi-granular indexing, evidence-gap tools (`Search` / `List` / `Grep` / `Read`), and adaptive escalation can improve accuracy **while cutting tokens**.

This project turns those ideas into a production-ready stack:

| Research idea | This repo today |
|---|---|
| Hierarchical documents | `document → section → subsection → chunk` with stable URIs |
| Multi-granular indexing | Bottom-up abstracts + embeddings at document/section/chunk |
| Semantic Search | pgvector cosine discovery returning `viking://` URIs |
| Cheap path / agents | Evidence collect + sufficiency now; agent loop planned |
| Evidence-first grounding | Authoritative Reads + assessed EvidenceBundle (no answers yet) |
| Experience edges | Planned - reuse successful retrieval traces |
| Production constraints | Typed config, migrations, health probes, adapters, tests |

> **Independent implementation** inspired by the paper - not a fork of the AGPL research artifacts.

---

## Features

- **Hierarchical documents** - preserve section / subsection / chunk structure instead of flattening everything
- **Stable URIs** - address any node with `viking://documents/...` (storage-independent)
- **Structure-aware ingestion** - Markdown, TXT, and PDF with region-local chunking
- **Idempotent uploads** - `skip_identical`, `replace`, or `create_version`
- **Structural navigation** - parent, children, ancestors, descendants, and URI resolve
- **Hierarchical abstracts** - bottom-up summaries with token budgets and idempotent reuse
- **Multi-granular embeddings** - chunk content + structural summaries with explicit embedding identity
- **Semantic Search** - pgvector cosine retrieval with granularity weights and URI-addressable hits
- **List / Grep / Read** - structural children, scoped literal Grep, authoritative Read with provenance
- **Evidence collection** - bounded Search→List/Read pipeline assembling deduplicated evidence bundles
- **Evidence sufficiency** - structured assessment (`sufficient` / `insufficient` / `unknown`) with citation checks
- **Shared budgets** - cumulative tool/embedding/read/LLM limits across a request
- **Replaceable providers** - OpenAI-compatible LLM/embeddings, deterministic fakes for offline tests
- **Production foundation** - FastAPI, PostgreSQL + pgvector, Redis, object store, typed settings, health checks

**Coming soon:** bounded agentic multi-round retrieval, experience edges, answer generation, and evaluation vs flat RAG.

See [`docs/RETRIEVAL.md`](docs/RETRIEVAL.md) for indexing and Search details.

---

## Architecture

```mermaid
flowchart TB
    Client[Client / SDK / UI]
    API[FastAPI Gateway]

    subgraph Ingest[Ingestion]
      Parse[Parsers TXT / MD / PDF]
      Hier[Hierarchy builder]
      Chunk[Structure-aware chunker]
      Persist[(PostgreSQL + object store)]
    end

    subgraph Retrieve[Retrieval]
      Search[Search]
      List[List]
      Grep[Grep]
      Read[Read]
      Judge[Evidence sufficiency]
      Agent[Bounded agent]
    end

    Client --> API
    API --> Ingest
    API --> Retrieve
    Parse --> Hier --> Chunk --> Persist
    Search --> Persist
    List --> Persist
    Grep --> Persist
    Read --> Persist
    Judge -->|insufficient| Agent
```

### Document model

Documents are **not** flat chunk lists:

```mermaid
flowchart TD
    D["DOCUMENT<br/>viking://documents/{id}"]
    S1[SECTION · Storage]
    S2[SECTION · Retrieval]
    S3[SECTION · Operations]
    SS1[SUBSECTION · PostgreSQL]
    SS2[SUBSECTION · Object Storage]
    SS3[SUBSECTION · Semantic Search]
    C1[CHUNK]
    C2[CHUNK]

    D --> S1
    D --> S2
    D --> S3
    S1 --> SS1
    S1 --> SS2
    S2 --> SS3
    SS1 --> C1
    SS1 --> C2
```

Stable URIs:

```text
viking://documents/{document_id}
viking://documents/{document_id}/nodes/{node_id}
```

### Ingestion pipeline

```mermaid
flowchart LR
    A[Upload] --> B[Hash / idempotency]
    B --> C[Parse]
    C --> D[Structure extract]
    D --> E[Normalize]
    E --> F[Hierarchy]
    F --> G[Chunk in-region]
    G --> H[Persist nodes + blob]
```

### Layout

```text
src/vikingrag/
  api/             FastAPI routes & schemas
  application/     Use-cases (ingest, navigation)
  domain/          Models, URIs, errors
  ingestion/       Parse → hierarchy → chunk pipeline
  infrastructure/  Postgres, Redis, object store
  providers/       LLM / embedding / rerank Protocols
  observability/   Structured logging + correlation
  settings/        Typed configuration
```

---

## Quick start

### Prerequisites

| Tool | Notes |
|---|---|
| ![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white) | Required |
| ![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?logo=uv&logoColor=white) | Recommended |
| ![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white) | Optional; or local Postgres + Redis |

### Install & run

```bash
git clone https://github.com/mhuzaifadev/VikingRAG.git
cd VikingRAG
cp .env.example .env

make install
make migrate
make docker-up
# or: make dev
```

Health:

```bash
curl -s http://localhost:8000/v1/health/live  | jq
curl -s http://localhost:8000/v1/health/ready | jq
```

Ingest a document, index it, and search:

```bash
curl -s -X POST http://localhost:8000/v1/documents \
  -F "file=@tests/fixtures/system_architecture.md;type=text/markdown" \
  -F "strategy=skip_identical" | jq

DOC_ID="<document_id from response>"
curl -s "http://localhost:8000/v1/documents/${DOC_ID}/tree" | jq

# Requires VIKINGRAG_LLM_PROVIDER=fake and VIKINGRAG_EMBEDDING_PROVIDER=deterministic
# (see .env.example) or real OpenAI-compatible credentials.
curl -s -X POST "http://localhost:8000/v1/documents/${DOC_ID}/index" \
  -H "Content-Type: application/json" \
  -d '{"force_summaries":false,"force_embeddings":false}' | jq

curl -s -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"How do we verify retrieved evidence?","top_k":5}' | jq
```

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/v1/health/live` | Liveness |
| `GET` | `/v1/health/ready` | Readiness (Postgres + Redis) |
| `POST` | `/v1/documents` | Multipart ingest (`.md` / `.txt` / `.pdf`) |
| `GET` | `/v1/documents/{id}` | Document metadata |
| `GET` | `/v1/documents/{id}/tree` | Structural tree |
| `POST` | `/v1/documents/{id}/index` | Summarize + embed (idempotent) |
| `GET` | `/v1/documents/{id}/index-status` | Index stage and counts |
| `POST` | `/v1/search` | Semantic Search |
| `POST` | `/v1/retrieval/list` | List direct children |
| `POST` | `/v1/retrieval/grep` | Scoped literal Grep |
| `POST` | `/v1/retrieval/read` | Authoritative Read |
| `POST` | `/v1/retrieval/evidence` | Collect evidence + assess sufficiency |
| `GET` | `/v1/nodes/{id}` | Node metadata (+ optional content) |
| `GET` | `/v1/uris/resolve?uri=` | Resolve a `viking://…` URI |

---

## Development

| Command | Purpose |
|---|---|
| `make install` | Sync deps (`uv`) + pre-commit |
| `make dev` | API with reload |
| `make lint` | Ruff |
| `make typecheck` | mypy (strict) |
| `make test` | Unit tests |
| `make test-integration` | Integration tests (Postgres + Redis) |
| `make migrate` | Apply Alembic migrations |
| `make docker-up` / `make docker-down` | Compose stack |

---

## Citation

If you use ideas from VikingRAG or this software in research, please cite the paper:

```bibtex
@article{gao2026vikingrag,
  title   = {VikingRAG: Accurate and Token-efficient Retrieval-augmented
             Generation over Structured Documents},
  author  = {Gao, Peiyuan and others},
  year    = {2026},
  eprint  = {2609.11390},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url     = {https://arxiv.org/abs/2609.11390}
}
```

Paper: [arXiv:2609.11390](https://arxiv.org/abs/2609.11390) · Machine-readable: [`CITATION.cff`](CITATION.cff) · Legal notices: [`NOTICE`](NOTICE)

---

## License

**Apache License 2.0** - see [`LICENSE`](LICENSE).

VikingRAG is independently authored for production use. Do **not** copy code from AGPL research repositories into this tree unless you intentionally accept those obligations.

---

## Contributing

Issues and PRs are welcome. Please:

1. Keep layers clean (API ≠ domain ≠ adapters)
2. Prefer typed contracts over `dict[str, Any]`
3. Add tests with features
4. Run `make lint && make typecheck && make test` before opening a PR

⭐ **If this project helps you, starring the repo helps others find it.**

<br/>

---


<p align="center">
  <a href="https://www.linkedin.com/in/mhuzaifadev">LinkedIn</a>
  ·
  <a href="https://mhuzaifa.com">mhuzaifa.com</a>
  ·
  <a href="https://github.com/mhuzaifadev">GitHub</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/built_with-curiosity-ff69b4?style=flat-square" alt="built with curiosity" />
  &nbsp;
  <img src="https://img.shields.io/badge/shipped_with-love-e25555?style=flat-square" alt="shipped with love" />
</p>
