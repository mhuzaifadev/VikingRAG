"""Paper corpus adapters (stubs until datasets are provisioned)."""

from __future__ import annotations

from vikingrag.evaluation.adapters.financebench import FinanceBenchAdapter
from vikingrag.evaluation.adapters.hotpotqa import HotpotQAAdapter
from vikingrag.evaluation.adapters.legalbench_cuad import LegalBenchCUADAdapter
from vikingrag.evaluation.adapters.qasper import QASPERAdapter
from vikingrag.evaluation.adapters.syllabusqa import SyllabusQAAdapter
from vikingrag.evaluation.adapters.versionqa import VersionQAAdapter
from vikingrag.evaluation.base import StubDatasetAdapter

ADAPTERS: dict[str, StubDatasetAdapter] = {
    "versionqa": VersionQAAdapter(),
    "syllabusqa": SyllabusQAAdapter(),
    "qasper": QASPERAdapter(),
    "hotpotqa": HotpotQAAdapter(),
    "legalbench_cuad": LegalBenchCUADAdapter(),
    "financebench": FinanceBenchAdapter(),
}


def get_adapter(name: str) -> StubDatasetAdapter:
    key = name.lower().strip().replace("-", "_")
    if key not in ADAPTERS:
        known = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"Unknown dataset {name!r}; known: {known}")
    return ADAPTERS[key]


__all__ = [
    "ADAPTERS",
    "FinanceBenchAdapter",
    "HotpotQAAdapter",
    "LegalBenchCUADAdapter",
    "QASPERAdapter",
    "SyllabusQAAdapter",
    "VersionQAAdapter",
    "get_adapter",
]
