# VikingRAG v0.5.2 release report

CI-verified line of the v0.5.1 experience-retirement work.

## Why 0.5.2

- PyPI `0.5.1` stays on the original product build
- `0.5.2` includes integration-fixture fixes so Postgres regression checks pass on CI

## Product (unchanged from 0.5.1 intent)

- REPLACE / soft-delete invalidate incident experience edges in one DB transaction
- Revision-specific source object keys
- Stale PENDING jobs skipped; builder stamps live revisions; snapshot rollback never reactivates `INVALIDATED`

## Verification

| Gate | Result |
|---|---|
| Unit + evaluation | pass |
| Integration (GitHub Actions) | pass — [run 38195739774](https://github.com/mhuzaifadev/VikingRAG/actions/runs/38195739774) |
| Offline smoke | pass |

## Install

```bash
pip install -U vikingrag==0.5.2
```
