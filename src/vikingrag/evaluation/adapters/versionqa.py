"""VersionQA adapter stub."""

from __future__ import annotations

from vikingrag.evaluation.base import DatasetManifest, StubDatasetAdapter


class VersionQAAdapter(StubDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(
            DatasetManifest(
                name="versionqa",
                version="paper-rev-unset",
                primary_source_url="https://github.com/rucdatascience/VikingRAG",
                expected_sha256=None,
                local_relative_path="versionqa",
                notes="Provision immutable revision under data/eval/versionqa with SHA256SUMS.",
            ),
            _download_hint="See docs/EVALUATION.md; do not invent scores without verified data.",
        )
