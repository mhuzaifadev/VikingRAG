# ADR 0005: Retrieval primitives and evidence sufficiency

- Status: Accepted
- Date: 2026-10-07

## Context

Semantic Search alone returns discovery previews and summaries. VikingRAG needs authoritative evidence for answering, with explicit List / Grep / Read tools, shared budgets, and a structured sufficiency judgment - without an autonomous agent loop yet.

## Decision

1. **Four primitives** with typed contracts: Search (existing), List, Grep, Read.
2. **Authoritative evidence** comes only from Read of stored node `content` - never from Search previews or generated abstracts.
3. **Shared `RetrievalContext`**: immutable `BudgetLimits` + mutable `UsageLedger` with atomic reservations; nested calls share one ledger and deadline.
4. **Bounded collector**: one Search → optional Grep → List children / Read excerpts within depth and token caps → `EvidenceBundle` with provenance and dedupe.
5. **Assessment**: `EvidenceAssessor` returns `SUFFICIENT | INSUFFICIENT | UNKNOWN`. Final status is computed by deterministic policy after validating evidence IDs and quotes. Empty bundles short-circuit without an LLM.
6. **Scope**: `permitted_document_ids` on the context; client filters may only narrow. No claim of multitenant auth in this milestone.
7. **APIs**: `POST /v1/retrieval/{list,grep,read,evidence}` alongside existing `POST /v1/search`.

## Consequences

- Future agent loops can reuse the same primitives and budget.
- Scripted/fake assessors are explicit config (`VIKINGRAG_RETRIEVAL_ASSESSOR_PROVIDER`) - never silent production fallbacks.
- Quote/ID validation is necessary but not sufficient for semantic entailment.
