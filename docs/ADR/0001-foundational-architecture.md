# ADR 0001: Foundational layered architecture and provider abstractions

- Status: Accepted
- Date: 2026-10-07
- Phase: 1

## Context

VikingRAG-X is a production retrieval platform inspired by the VikingRAG paper, not a research reproduction. Early implementation must establish boundaries that preserve reliability, provider independence, and maintainability before ingestion or retrieval logic lands.

Without explicit layering, FastAPI routes and domain logic tend to couple to a single LLM vendor, SQLAlchemy models, and storage details - making later adaptive retrieval, experience edges, and multi-provider support expensive to retrofit.

## Decision

Adopt a layered architecture for Phase 1:

1. **API** (`vikingrag.api`) - HTTP, OpenAPI, health probes, request correlation.
2. **Application** (`vikingrag.application`) - reserved for use-cases (empty in Phase 1).
3. **Domain** (`vikingrag.domain`) - typed models and errors with no FastAPI/SQLAlchemy/vendor imports.
4. **Infrastructure** - PostgreSQL (async SQLAlchemy), Redis, object storage adapters.
5. **Providers** - `Protocol` contracts for LLM, embeddings, and reranking (unimplemented until later phases).

Infrastructure and providers are accessed through interfaces (`ObjectStore`, `DocumentRepository`, `VectorRepository`, `LLMProvider`, etc.). Concrete adapters (e.g. `LocalObjectStore`, `SqlDocumentRepository`) sit outside domain logic.

Configuration is typed via pydantic-settings. Schema changes go through Alembic migrations; pgvector is enabled in the first migration even though vector tables arrive later.

## Consequences

### Positive

- Domain retrieval contracts can evolve without rewriting HTTP or ORM layers.
- Vendor SDKs can be swapped without touching orchestration.
- Health/readiness and Docker Compose give a reproducible boot path before feature work.

### Negative / trade-offs

- Slightly more boilerplate than a single-module prototype.
- Some repository interfaces remain unimplemented (`VectorRepository`) until later phases - callers must not fake results.

## Alternatives considered

- **Monolithic FastAPI module with inline OpenAI/SQL calls** - rejected; violates provider independence and testability.
- **LangChain / LlamaIndex as the core abstraction** - rejected; hides retrieval semantics and couples us to framework lifecycle.
- **Copying AGPL research repositories** - rejected; independent implementation from `docs/ARCHITECTURE.md` and the paper.
