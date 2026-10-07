"""HotpotQA adapter stub."""

from __future__ import annotations

from vikingrag.evaluation.base import DatasetManifest, StubDatasetAdapter


class HotpotQAAdapter(StubDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(
            DatasetManifest(
                name="hotpotqa",
                version="paper-rev-unset",
                primary_source_url="https://hotpotqa.github.io/",
                expected_sha256=None,
                local_relative_path="hotpotqa",
                notes="Multi-hop corpus; warm-up must never use held-out eval questions.",
            ),
            _download_hint="Download HotpotQA from the official site; verify before scoring.",
        )
