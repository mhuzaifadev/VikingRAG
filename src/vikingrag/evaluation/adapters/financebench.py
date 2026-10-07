"""FinanceBench adapter stub."""

from __future__ import annotations

from vikingrag.evaluation.base import DatasetManifest, StubDatasetAdapter


class FinanceBenchAdapter(StubDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(
            DatasetManifest(
                name="financebench",
                version="paper-rev-unset",
                primary_source_url="https://github.com/patronus-ai/financebench",
                expected_sha256=None,
                local_relative_path="financebench",
                notes="FinanceBench requires explicit license compliance before local caching.",
            ),
            _download_hint="Clone/cache FinanceBench only when license permits; verify digests.",
        )
