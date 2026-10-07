# System Architecture

Overview of the VikingRAG platform.

## Storage

Durable state and blobs.

### PostgreSQL

Primary metadata and hierarchy store.

### Object Storage

Source files and large binaries.

## Retrieval

How evidence is found.

### Semantic Search

Vector similarity over chunks and abstracts.

### Structural Search

Navigate the document tree with List and Read.

### Evidence Verification

Claim to evidence mapping before answering.

## Operations

Running the system in production.

### Observability

Traces, metrics, and structured logs.

### Deployment

Containers, migrations, and health probes.
