# ADR 0004: pgvector cosine search with HNSW

- Status: Accepted
- Date: 2026-10-07

## Context

Phase 3 needs production-shaped vector similarity over multi-granular node embeddings. Architecture allows pgvector or an external vector DB; we already run PostgreSQL.

## Decision

1. Store embeddings in Postgres via **pgvector**.
2. Similarity metric: **cosine** (`vector_cosine_ops`). Search exposes both raw `similarity` and weighted `score`.
3. Index: **HNSW** (`m=16`, `ef_construction=64`) on `node_embeddings.embedding`.
4. Fixed column dimension **1536** aligned with the default embedding settings (`text-embedding-3-small` / deterministic test space). Changing dimensions requires a new migration and re-index; row-level `dimensions` + identity still prevent silent mixing.
5. Search filters by embedding identity, optional `document_ids`, `node_types`, `representation_types`, and `min_score`.

## Consequences

- No external vector service in the default path.
- HNSW rebuild cost exists on bulk reload; acceptable for v0.1.
- Alternative engines (Qdrant) remain possible behind `VectorRepository` later without changing Search URIs.
