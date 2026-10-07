"""Evaluation harness - adapters, smoke runs, manifests.

Never invent measured scores. Missing datasets raise clear errors.
"""

from vikingrag.evaluation.base import (
    DatasetAdapter,
    DatasetManifest,
    DatasetNotAvailableError,
)

__all__ = [
    "DatasetAdapter",
    "DatasetManifest",
    "DatasetNotAvailableError",
]
