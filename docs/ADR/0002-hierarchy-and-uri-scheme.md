# ADR 0002: Hierarchical document nodes and URI scheme

- Status: Accepted
- Date: 2026-10-07
- Phase: 2

## Context

VikingRAG-style retrieval depends on hierarchy-preserving storage and addressable structural regions. Phase 1 only had a flat `documents` table. Phase 2 must introduce a durable node tree and stable URIs without locking into table names or embedding storage details into identity.

`docs/ARCHITECTURE.md` sketches a tenant/corpus-prefixed URI namespace. Multi-tenancy is not yet implemented.

## Decision

1. **Single table `document_nodes`** holds DOCUMENT / SECTION / SUBSECTION / CHUNK (and future types). Chunks are first-class nodes under structural parents - not a separate premature `objects` table.

2. **URI scheme (Phase 2):**

   ```text
   viking://documents/{document_id}
   viking://documents/{document_id}/nodes/{node_id}
   ```

   Identity is UUID-based. Titles/slugs never form identity. No SQL table names appear in URIs.

3. **Ancestry:** store `parent_id`, `depth`, `ordinal`, and `path_ids UUID[]` for efficient filtering. Traversal uses explicit repository methods and recursive CTEs - not ORM lazy loading.

4. **Abstracts:** nullable `abstract_text` + `abstract_status` columns prepare hierarchical abstracts without calling an LLM in Phase 2.

5. **Cascade:** deleting a document cascades to all nodes. Deleting a parent node cascades to descendants.

6. **Tenant/corpus** segments are deferred until auth/tenancy lands; URIs will gain optional prefixes via a documented migration rather than encoding placeholders now.

## Consequences

- List/Read primitives can resolve URIs without knowing Postgres layout.
- Later `objects` / embedding tables can reference `document_nodes.id` without URI changes.
- Existing ARCHITECTURE examples with `section-1/` path segments remain conceptual; production identity is UUID nodes.

## Alternatives considered

- Path-segment URIs built from heading titles - rejected; titles change and collide.
- Separate `chunks` table only - rejected for Phase 2; over-splits the hierarchy before retrieval needs it.
- Premature tenant/corpus URI prefixes with sentinel UUIDs - rejected; invents tenancy we do not have.
