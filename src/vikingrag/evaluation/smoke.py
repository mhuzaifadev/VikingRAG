"""Deterministic offline evaluation smoke using the fixture syllabus corpus.

Labeled **smoke** — not a paper-table measurement. Never invent accuracy scores.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import resources
from pathlib import Path

from vikingrag.ingestion.parsers.markdown import MarkdownDocumentParser

# Fixed smoke queries answered only from fixture text (historical-style, not gold eval).
_SMOKE_QUERIES: tuple[tuple[str, str], ...] = (
    ("What stores primary metadata?", "PostgreSQL"),
    ("How do we verify retrieved evidence?", "claim-evidence"),
    ("What finds structural regions?", "Semantic Search"),
)


@dataclass(frozen=True, slots=True)
class SmokeCheck:
    name: str
    ok: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SmokeReport:
    status: str  # "ok" | "failed"
    label: str
    fixture_path: str
    fixture_sha256: str
    checks: tuple[SmokeCheck, ...]
    notes: str

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "label": self.label,
            "fixture_path": self.fixture_path,
            "fixture_sha256": self.fixture_sha256,
            "checks": [asdict(c) for c in self.checks],
            "notes": self.notes,
            "measured_scores": None,  # explicitly unmeasured
        }


def _packaged_fixture_bytes() -> tuple[bytes, str] | None:
    try:
        pkg = resources.files("vikingrag.evaluation.fixtures")
        packaged = pkg.joinpath("system_architecture.md")
        if packaged.is_file():
            return packaged.read_bytes(), "vikingrag.evaluation.fixtures/system_architecture.md"
    except (TypeError, FileNotFoundError, ModuleNotFoundError, AttributeError):
        return None
    return None


def default_fixture_path() -> Path:
    """Prefer packaged wheel resource (copied to temp), then repo ``tests/fixtures``."""
    packaged = _packaged_fixture_bytes()
    if packaged is not None:
        import tempfile

        raw, _label = packaged
        tmp = Path(tempfile.gettempdir()) / "vikingrag_smoke_system_architecture.md"
        tmp.write_bytes(raw)
        return tmp

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "fixtures" / "system_architecture.md",
        here.parents[3] / "tests" / "fixtures" / "system_architecture.md",
        here.parents[4] / "tests" / "fixtures" / "system_architecture.md",
        Path.cwd() / "tests" / "fixtures" / "system_architecture.md",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Offline smoke fixture system_architecture.md not found in package or "
        "tests/fixtures/. Reinstall vikingrag or run from the repository root."
    )


def run_offline_smoke(*, fixture: Path | None = None) -> SmokeReport:
    if fixture is not None:
        path = fixture
        raw = path.read_bytes()
        path_label = str(path)
    else:
        packaged = _packaged_fixture_bytes()
        if packaged is not None:
            raw, path_label = packaged
            path = Path(path_label)
        else:
            path = default_fixture_path()
            raw = path.read_bytes()
            path_label = str(path)
    digest = hashlib.sha256(raw).hexdigest()
    checks: list[SmokeCheck] = []

    parser = MarkdownDocumentParser()
    try:
        parsed = parser.parse(raw, filename="system_architecture.md", mime_type="text/markdown")
        checks.append(
            SmokeCheck(
                name="parse_fixture",
                ok=len(parsed.blocks) > 0,
                detail=f"blocks={len(parsed.blocks)} title={parsed.title!r}",
            )
        )
    except Exception as exc:
        checks.append(SmokeCheck(name="parse_fixture", ok=False, detail=str(exc)))
        return SmokeReport(
            status="failed",
            label="smoke",
            fixture_path=path_label,
            fixture_sha256=digest,
            checks=tuple(checks),
            notes="Parse failed; no scores produced.",
        )

    text = raw.decode("utf-8")
    for query, needle in _SMOKE_QUERIES:
        found = needle.casefold() in text.casefold()
        checks.append(
            SmokeCheck(
                name=f"fixture_contains:{needle}",
                ok=found,
                detail=("support span present in fixture" if found else "missing support span"),
            )
        )
        _ = query

    for heading in ("PostgreSQL", "Semantic Search", "Evidence Verification"):
        ok = heading in text
        checks.append(
            SmokeCheck(
                name=f"heading:{heading}",
                ok=ok,
                detail="present" if ok else "absent",
            )
        )

    all_ok = all(c.ok for c in checks)
    return SmokeReport(
        status="ok" if all_ok else "failed",
        label="smoke",
        fixture_path=path_label,
        fixture_sha256=digest,
        checks=tuple(checks),
        notes=(
            "Offline smoke validates fixture integrity and scaffolding only. "
            "measured_scores is null — not a paper benchmark result."
        ),
    )


def report_json(report: SmokeReport) -> str:
    return json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
