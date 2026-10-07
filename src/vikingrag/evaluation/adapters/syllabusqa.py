"""SyllabusQA adapter stub (offline smoke uses repo fixture, not this download)."""

from __future__ import annotations

from vikingrag.evaluation.base import DatasetManifest, StubDatasetAdapter


class SyllabusQAAdapter(StubDatasetAdapter):
    def __init__(self) -> None:
        super().__init__(
            DatasetManifest(
                name="syllabusqa",
                version="paper-rev-unset",
                primary_source_url="https://github.com/rucdatascience/VikingRAG",
                expected_sha256=None,
                local_relative_path="syllabusqa",
                notes=(
                    "Full SyllabusQA requires DOCX/PDF corpora from the paper release. "
                    "Offline smoke uses tests/fixtures/system_architecture.md only."
                ),
            ),
            _download_hint="Place verified SyllabusQA files under data/eval/syllabusqa.",
        )
