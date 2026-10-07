"""CI-friendly offline evaluation smoke."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vikingrag.evaluation.adapters import ADAPTERS, get_adapter
from vikingrag.evaluation.base import DatasetNotAvailableError
from vikingrag.evaluation.cli import main
from vikingrag.evaluation.smoke import run_offline_smoke


def test_offline_smoke_ok() -> None:
    report = run_offline_smoke()
    assert report.status == "ok"
    assert report.label == "smoke"
    assert report.to_dict()["measured_scores"] is None
    assert report.fixture_sha256
    assert all(c.ok for c in report.checks)


def test_cli_smoke_offline(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["smoke", "--offline"])
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["status"] == "ok"
    assert payload["measured_scores"] is None


def test_adapters_refuse_missing_data(tmp_path: Path) -> None:
    assert len(ADAPTERS) == 6
    adapter = get_adapter("syllabusqa")
    with pytest.raises(DatasetNotAvailableError):
        adapter.require_present(tmp_path)
    with pytest.raises(DatasetNotAvailableError):
        adapter.download(tmp_path)


def test_adapter_verify_missing(tmp_path: Path) -> None:
    adapter = get_adapter("qasper")
    result = adapter.verify(tmp_path)
    assert result.ok is False
