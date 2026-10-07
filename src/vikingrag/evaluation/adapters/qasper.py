"""QASPER adapter stub."""

from __future__ import annotations

from vikingrag.evaluation.base import DatasetManifest, StubDatasetAdapter


class QASPERAdapter(StubDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(
            DatasetManifest(
                name="qasper",
                version="paper-rev-unset",
                primary_source_url="https://allenai.org/data/qasper",
                expected_sha256=None,
                local_relative_path="qasper",
                notes="Retain untouched original QASPER revision; document transforms separately.",
            ),
            _download_hint="Obtain QASPER from the official release and verify checksums.",
        )
