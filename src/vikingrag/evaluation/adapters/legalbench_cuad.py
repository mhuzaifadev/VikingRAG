"""LegalBench-CUAD adapter stub."""

from __future__ import annotations

from vikingrag.evaluation.base import DatasetManifest, StubDatasetAdapter


class LegalBenchCUADAdapter(StubDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(
            DatasetManifest(
                name="legalbench_cuad",
                version="paper-rev-unset",
                primary_source_url="https://github.com/TheAtticusProject/cuad",
                expected_sha256=None,
                local_relative_path="legalbench_cuad",
                notes="CUAD / LegalBench licensing and revision pins must be recorded in manifests.",
            ),
            _download_hint="Provision CUAD under data/eval/legalbench_cuad with SHA256SUMS.",
        )
