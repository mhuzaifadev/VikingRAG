# ADR 0003: Node representations and embedding identity

- Status: Accepted
- Date: 2026-10-07

## Context

Semantic retrieval must operate on multi-granular representations (document / section / subsection summaries and chunk content) without mixing incompatible embedding spaces. Flat nullable columns on `document_nodes` are insufficient for versioning, idempotency, and provider identity.

## Decision

1. Persist **`node_representations`** separately from `document_nodes`:
   - unique `(node_id, representation_type)`
   - store `content`, `content_hash`, `generator`, `generator_model`, `version`
2. Persist **`node_embeddings`** keyed by representation + embedding identity:
   - unique `(representation_id, provider, model, dimensions, identity_version)`
   - store the vector and identity fields explicitly
3. Domain type **`EmbeddingIdentity`** is required for every search and upsert. Incompatible identities raise `EmbeddingIdentityMismatch` - never silent compare.
4. Representation types for this phase: `node_content` (chunks) and `node_summary` (structural nodes).
5. `document_nodes.abstract_text` remains a convenience mirror of the latest `NODE_SUMMARY` for navigation; `node_representations` is authoritative for indexing.

## Consequences

- Indexing can retry without corrupting hierarchy ingestion.
- Re-embedding after model change is deliberate (new identity rows).
- Vector column width is fixed at 1536 in migration `20261007_0003` (see ADR 0004).
