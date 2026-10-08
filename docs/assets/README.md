# README visuals

| File | Content |
|---|---|
| `comparison-tokens.png` | Paper Table 3 — VikingRAG / E / E+ token % vs gold baseline |
| `comparison-accuracy.png` | Paper Table 3 — E+ vs gold & silver baselines (accuracy claimed competitive in paper Fig. 3) |

**Provenance:** numbers from Gao et al., [arXiv:2609.11390](https://arxiv.org/abs/2609.11390). They are **not** runs of this Apache-2.0 repo.

Regenerate:

```bash
uv run python scripts/render_paper_comparison_charts.py
```

When you have your own eval exports, replace these PNGs and update the README caption so it no longer says “paper experiments.”
