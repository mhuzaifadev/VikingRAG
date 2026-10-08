# System Architecture

Overview of the retrieval platform.

## Storage

Persistence components.

### PostgreSQL

Primary metadata and hierarchy store with pgvector for similarity search.

### Object Storage

Binary blobs and original uploads live in object storage.

## Retrieval

How discovery and verification work.

### Semantic Search

Semantic Search finds relevant structural regions using multi-granular embeddings.

### Structural Search

Structural navigation lists parents, children, and descendants by URI.

### Evidence Verification

Evidence Verification maps each claim to retrieved evidence, checks citations,
and rejects unsupported statements before final answers are returned.
How do we verify retrieved evidence? By claim-evidence pairing and citation checks.
