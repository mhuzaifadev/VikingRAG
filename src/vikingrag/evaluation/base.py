"""Dataset adapter contracts - download/verify with checksums; no invented results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from vikingrag.domain.errors import DomainError


class DatasetNotAvailableError(DomainError):
    """Raised when evaluation data is missing or fails checksum verification."""

    def __init__(self, dataset: str, message: str) -> None:
        super().__init__(
            f"Dataset '{dataset}' unavailable: {message}",
            code="dataset_not_available",
        )
        self.dataset = dataset


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable descriptor for an evaluation corpus revision."""

    name: str
    version: str
    primary_source_url: str
    expected_sha256: str | None = None
    local_relative_path: str = ""
    notes: str = ""
    checksum_fields: tuple[str, ...] = ("expected_sha256",)


@dataclass(slots=True)
class VerifyResult:
    ok: bool
    path: Path | None
    message: str
    sha256: str | None = None


@runtime_checkable
class DatasetAdapter(Protocol):
    """Download/verify interface for paper-shaped corpora.

    Implementations must not fabricate questions, gold answers, or scores.
    """

    @property
    def manifest(self) -> DatasetManifest: ...

    def data_root(self, base: Path) -> Path: ...

    def is_present(self, base: Path) -> bool: ...

    def verify(self, base: Path) -> VerifyResult: ...

    def download(self, base: Path, *, force: bool = False) -> Path:
        """Fetch immutable revision into ``base``. Raises if not implemented or fails."""
        ...

    def require_present(self, base: Path) -> Path:
        """Return dataset path or raise DatasetNotAvailableError."""
        ...


@dataclass
class StubDatasetAdapter:
    """Base stub: documents source/checksums; refuses silent fake results."""

    _manifest: DatasetManifest
    _download_hint: str = field(default="")

    @property
    def manifest(self) -> DatasetManifest:
        return self._manifest

    def data_root(self, base: Path) -> Path:
        rel = self._manifest.local_relative_path or self._manifest.name
        return (base / rel).resolve()

    def is_present(self, base: Path) -> bool:
        root = self.data_root(base)
        return root.is_dir() and any(root.iterdir())

    def verify(self, base: Path) -> VerifyResult:
        root = self.data_root(base)
        if not root.exists():
            return VerifyResult(
                ok=False,
                path=None,
                message=f"missing path {root}; {self._download_hint}".strip(),
            )
        if self._manifest.expected_sha256 is None:
            return VerifyResult(
                ok=True,
                path=root,
                message="present (no checksum configured; treat as unverified layout)",
            )
        # Directory-level checksums require a published artifact hash file.
        marker = root / "SHA256SUMS"
        if not marker.is_file():
            return VerifyResult(
                ok=False,
                path=root,
                message=(
                    f"expected SHA256SUMS with {self._manifest.expected_sha256}; "
                    "refusing to invent verification"
                ),
            )
        content = marker.read_text(encoding="utf-8").strip()
        if self._manifest.expected_sha256 not in content:
            return VerifyResult(
                ok=False,
                path=root,
                message="SHA256SUMS does not contain expected revision digest",
            )
        return VerifyResult(
            ok=True,
            path=root,
            message="checksum marker matched",
            sha256=self._manifest.expected_sha256,
        )

    def download(self, base: Path, *, force: bool = False) -> Path:
        del force
        raise DatasetNotAvailableError(
            self._manifest.name,
            (
                f"automatic download is not implemented for this adapter. "
                f"Obtain data from {self._manifest.primary_source_url}. "
                f"{self._download_hint}"
            ).strip(),
        )

    def require_present(self, base: Path) -> Path:
        result = self.verify(base)
        if not result.ok or result.path is None:
            raise DatasetNotAvailableError(
                self._manifest.name,
                result.message
                or f"place files under {self.data_root(base)} after verifying checksums",
            )
        return result.path
