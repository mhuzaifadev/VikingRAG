"""Packaged migrate helper resolves a migrations directory."""

from __future__ import annotations

from pathlib import Path

from vikingrag.migrate import _script_location


def test_script_location_finds_repo_or_package_migrations() -> None:
    loc = _script_location()
    assert loc.is_dir()
    assert (loc / "env.py").is_file()
    assert any(loc.glob("versions/*.py"))
    # Sanity: Path is absolute
    assert Path(loc).is_absolute()
