<p align="center">
  <a href="https://github.com/mhuzaifadev/VikingRAG">
    <img src="https://img.shields.io/badge/VikingRAG-0.4.0-1f6feb?style=for-the-badge&labelColor=0d1117" alt="VikingRAG 0.4.0" />
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
  <a href="https://github.com/mhuzaifadev/VikingRAG/releases/tag/v0.4.0"><img src="https://img.shields.io/badge/release-v0.4.0-blue?logo=github" alt="Release v0.4.0" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?logo=apache" alt="Apache 2.0" /></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white" alt="Python 3.12+" /></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-async-009688?logo=fastapi&logoColor=white" alt="FastAPI" /></a>
  <a href="https://www.postgresql.org/"><img src="https://img.shields.io/badge/PostgreSQL-pgvector-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL + pgvector" /></a>
  <a href="https://redis.io/"><img src="https://img.shields.io/badge/Redis-ready-DC382D?logo=redis&logoColor=white" alt="Redis" /></a>
  <a href="https://pypi.org/project/vikingrag/"><img src="https://img.shields.io/pypi/v/vikingrag?logo=pypi&logoColor=white&label=PyPI" alt="PyPI" /></a>
  <a href="https://arxiv.org/abs/2609.11390"><img src="https://img.shields.io/badge/arXiv-2609.11390-b31b1b?logo=arxiv&logoColor=white" alt="arXiv paper" /></a>
</p>

<p align="center">
  <a href="#install"><strong>Install</strong></a>
  ·
  <a href="#features"><strong>Features</strong></a>
  ·
  <a href="#architecture"><strong>Architecture</strong></a>
  ·
  <a href="#quick-start"><strong>Quick start</strong></a>
  ·
  <a href="#sdk"><strong>SDK</strong></a>
  ·
  <a href="#api"><strong>API</strong></a>
  ·
  <a href="docs/README.md"><strong>Docs</strong></a>
  ·
  <a href="#citation"><strong>Cite</strong></a>
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
| Hierarchical documents | `document → section → subsection → chunk` with stable + canonical object URIs |
| Multi-granular indexing | Bottom-up abstracts + embeddings at document/section/chunk |
| Semantic Search | pgvector cosine discovery returning `viking://` URIs |
| Cheap path / agents | Algorithm 1 agentic loop; E+ one-round fast path then escalate |
| Evidence-first grounding | Authoritative Reads, citations, `POST /v1/answers` |
| Experience edges | Algorithm 2 construction + Algorithm 3 Search+ (γ-gated) |
| Production constraints | Auth allowlist, migrations, prod Compose, eval smoke, typed settings |

> **Independent implementation** inspired by the paper - not a fork of the AGPL research artifacts.

### Comparison visuals

Drop charts into [`docs/assets/`](docs/assets/) (see that folder’s README), then uncomment:

<!--
<p align="center">
  <img src="docs/assets/comparison-tokens.png" alt="Token cost vs baselines" width="720" />
</p>
<p align="center">
  <img src="docs/assets/comparison-accuracy.png" alt="Accuracy vs baselines" width="720" />
</p>
-->

Best place for side-by-side paper-style figures is **right here** (under Why), or a short **Results** section after Features. Use only real measurements or clearly cited paper figures — never invent scores.

---

## Install

### Option A — PyPI (fastest on any laptop)

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U vikingrag
# optional richer DOCX parsing:
# pip install "vikingrag[docx]"
```

You still need **Postgres + pgvector** and **Redis**, plus a `.env` (copy from the [repo `.env.example`](https://github.com/mhuzaifadev/VikingRAG/blob/main/.env.example)):

```bash
export VIKINGRAG_DATABASE_URL=postgresql+asyncpg://vikingrag:vikingrag@localhost:5432/vikingrag
export VIKINGRAG_REDIS_URL=redis://localhost:6379/0
export VIKINGRAG_LLM_PROVIDER=fake          # or openai / deepseek / gemini / anthropic
export VIKINGRAG_EMBEDDING_PROVIDER=deterministic

vikingrag-migrate upgrade head
vikingrag-api
# API: http://localhost:8000/docs
```

SDK:

```python
from vikingrag import VikingRAGClient
client = VikingRAGClient.from_settings()
```

### Option B — Clone + uv (full stack / contributors)

```bash
git clone https://github.com/mhuzaifadev/VikingRAG.git
cd VikingRAG
cp .env.example .env
# Python 3.12+, https://docs.astral.sh/uv/
make install && make docker-up && make migrate && make dev
```

### Option C — pip from GitHub (pre-release / specific tag)

```bash
pip install "git+https://github.com/mhuzaifadev/VikingRAG.git@v0.4.0"
```

### Option D — Docker Compose

```bash
git clone https://github.com/mhuzaifadev/VikingRAG.git && cd VikingRAG
cp .env.example .env
docker compose up -d --build
```

Production-shaped: `docker compose -f docker-compose.prod.yml up -d --build` — see [`docs/OPERATIONS.md`](docs/OPERATIONS.md).

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
- **Agentic Algorithm 1** - tool-calling Search/List/Grep/Read loop with shared budgets and finalization reserve
- **Answers API** - `POST /v1/answers` and `POST /v1/query` with modes `vikingrag` / `vikingrag_e` / `vikingrag_e_plus`
- **Experience edges** - Algorithm 2 learning jobs + Algorithm 3 Search+ expansion
- **Evaluation scaffolding** - six dataset adapters + offline smoke (`vikingrag-eval smoke --offline`)
- **Auth** - optional API-key + server document allowlist (single-tenant)
- **Shared budgets** - cumulative tool/embedding/read/LLM limits across a request
- **Replaceable providers** - `openai` / `deepseek` / `gemini` / `anthropic` / `vllm`; fakes rejected in production
- **SDK facade** - `VikingRAGClient` for embed-in-process use without FastAPI
- **Production foundation** - FastAPI, PostgreSQL + pgvector, Redis, object store, typed settings, health checks

**Docs hub:** [`docs/README.md`](docs/README.md) · [`docs/RETRIEVAL.md`](docs/RETRIEVAL.md) · [`docs/PAPER_PARITY.md`](docs/PAPER_PARITY.md) · [`AGENTS.md`](AGENTS.md)

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

After [Install](#install), check health:

```bash
curl -s http://localhost:8000/health/live  | jq
curl -s http://localhost:8000/health/ready | jq
```

Ingest a document, index it, and search (with `.env` defaults: `VIKINGRAG_LLM_PROVIDER=fake`, `VIKINGRAG_EMBEDDING_PROVIDER=deterministic`):

```bash
curl -s -X POST http://localhost:8000/v1/documents \
  -F "file=@tests/fixtures/system_architecture.md;type=text/markdown" \
  -F "strategy=skip_identical" | jq

DOC_ID="<document_id from response>"
curl -s "http://localhost:8000/v1/documents/${DOC_ID}/tree" | jq

curl -s -X POST "http://localhost:8000/v1/documents/${DOC_ID}/index" \
  -H "Content-Type: application/json" \
  -d '{"force_summaries":false,"force_embeddings":false}' | jq

curl -s -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query":"How do we verify retrieved evidence?","top_k":5}' | jq
```

Interactive API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Verify before go-live

`0.4.0` is an **algorithm-complete milestone**, not “I ran paper Tables 3–5.” Use this ladder:

```bash
# 1) Offline gates (always)
make install && make lint && make typecheck && make test
uv run vikingrag-eval smoke --offline

# 2) Integration (Postgres + Redis)
make docker-up && make migrate && make test-integration

# 3) Local API with fake/deterministic providers (wiring only)
make dev
# then ingest → index → search → POST /v1/answers (see Quick start)

# 4) Production-shaped stack with real providers + auth
#    (see docs/OPERATIONS.md — rejects fake/scripted/deterministic)
docker compose -f docker-compose.prod.yml up -d --build
```

Full checklist and result meanings: [`docs/OPERATIONS.md`](docs/OPERATIONS.md#pre-live-checklist).

**What counts as a result today**

| Output | How | Claims |
|---|---|---|
| Unit / lint / mypy | `make test` etc. | Code quality |
| Offline eval smoke JSON | `vikingrag-eval smoke --offline` | Fixture integrity only (`measured_scores: null`) |
| Integration pass | `make test-integration` | DB/migrate/API contracts |
| Paper accuracy/token tables | Paid corpora + real LLM runs | **BLOCKED** until you run them — never invent |

---

## SDK

Embed VikingRAG in your process (same settings / providers as the API):

```python
from vikingrag import VikingRAGClient

client = VikingRAGClient.from_settings()
# await client.search.search(...)
# await client.answer_generator().generate(...)
# await client.aclose()
```

Providers via env: `VIKINGRAG_LLM_PROVIDER=openai|deepseek|gemini|anthropic|vllm` (and matching embeddings). Production forbids `fake` / `deterministic` / `scripted`.

```bash
pip install vikingrag
```

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health/live` | Liveness |
| `GET` | `/health/ready` | Readiness (Postgres + Redis) |
| `POST` | `/v1/documents` | Multipart ingest (`.md` / `.txt` / `.pdf` / `.docx`) |
| `GET` | `/v1/documents/{id}` | Document metadata |
| `GET` | `/v1/documents/{id}/tree` | Structural tree |
| `POST` | `/v1/documents/{id}/index` | Summarize + embed (idempotent) |
| `GET` | `/v1/documents/{id}/index-status` | Index stage and counts |
| `POST` | `/v1/search` | Semantic Search (`scope_uri` optional) |
| `POST` | `/v1/retrieval/list` | List direct children |
| `POST` | `/v1/retrieval/grep` | Scoped Grep |
| `POST` | `/v1/retrieval/read` | Authoritative Read |
| `POST` | `/v1/retrieval/evidence` | Collect evidence + assess sufficiency |
| `POST` | `/v1/answers` | Cited answer (`vikingrag` / `vikingrag_e` / `vikingrag_e_plus`) |
| `POST` | `/v1/query` | Mode-switched query orchestration |
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
| `make docker-up` / `make docker-down` | Compose stack (volumes preserved on down) |
| `uv run vikingrag-eval smoke --offline` | Offline evaluation scaffolding smoke |

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
