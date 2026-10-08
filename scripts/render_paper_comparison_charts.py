#!/usr/bin/env python3
"""Render README comparison charts from published paper numbers only.

Source: Gao et al., VikingRAG, arXiv:2609.11390v1, Table 3 + abstract.
These are NOT measurements from this Apache-2.0 repository.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

DATASETS = [
    "VersionQA",
    "SyllabusQA",
    "QASPER",
    "HotpotQA",
    "LegalBench",
    "FinanceBench",
]

TABLE3_GOLD = {
    "VikingRAG": [36.9, 40.3, 48.7, 28.1, 49.8, 14.4],
    "VikingRAG-E": [31.0, 31.2, 39.3, 24.8, 38.6, 9.7],
    "VikingRAG-E+": [23.1, 25.2, 31.4, 12.3, 30.7, 7.9],
}
TABLE3_SILVER_EPLUS = [8.0, 32.5, 22.1, 5.1, 24.8, 10.0]
TABLE3_GOLD_EPLUS = TABLE3_GOLD["VikingRAG-E+"]

OUT = Path(__file__).resolve().parents[1] / "docs" / "assets"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
        }
    )


def render_tokens() -> Path:
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.2), dpi=160)
    x = np.arange(len(DATASETS))
    width = 0.25
    colors = {"VikingRAG": "#1f6feb", "VikingRAG-E": "#8250df", "VikingRAG-E+": "#1a7f37"}

    for i, (name, vals) in enumerate(TABLE3_GOLD.items()):
        ax.bar(x + (i - 1) * width, vals, width, label=name, color=colors[name], edgecolor="white")

    ax.set_ylabel("% of gold-baseline tokens (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=18, ha="right")
    ax.set_ylim(0, 60)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.set_title("Token cost vs highest-accuracy baseline")
    fig.text(
        0.5,
        0.01,
        "Gao et al., arXiv:2609.11390, Table 3 — paper results, not this repository",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#57606a",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = OUT / "comparison-tokens.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def render_accuracy() -> Path:
    """E+ vs gold/silver baselines — paper pairs these with competitive accuracy (Fig. 3)."""
    _style()
    fig, ax = plt.subplots(figsize=(10.5, 5.0), dpi=160)
    x = np.arange(len(DATASETS))
    width = 0.35

    ax.bar(
        x - width / 2,
        TABLE3_GOLD_EPLUS,
        width,
        label="vs gold (1st accuracy)",
        color="#1a7f37",
        edgecolor="white",
    )
    ax.bar(
        x + width / 2,
        TABLE3_SILVER_EPLUS,
        width,
        label="vs silver (2nd accuracy)",
        color="#3fb950",
        edgecolor="white",
    )

    ax.set_ylabel("% of baseline tokens used by VikingRAG-E+")
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=18, ha="right")
    ax.set_ylim(0, 40)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title("VikingRAG-E+ — paper reports competitive accuracy at these token ratios")
    ax.text(
        0.02,
        0.95,
        "Abstract: E+ uses 5.1%-32.5% of high-accuracy\nbaseline tokens while remaining competitive.\nNumeric accuracy bars: paper Fig. 3.",
        transform=ax.transAxes,
        va="top",
        fontsize=8.5,
        color="#24292f",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "edgecolor": "#d0d7de"},
    )
    fig.text(
        0.5,
        0.01,
        "Gao et al., arXiv:2609.11390, Table 3 + abstract — paper results, not this repository",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#57606a",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    path = OUT / "comparison-accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print(render_tokens())
    print(render_accuracy())


if __name__ == "__main__":
    main()
