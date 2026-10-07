"""Literal Grep matching helpers."""

from __future__ import annotations

from vikingrag.application.grep_primitive import _excerpt, _find_literal_positions


def test_literal_positions_case_sensitive() -> None:
    text = "Evidence Verification and evidence verification"
    pos = _find_literal_positions(text, "Evidence", case_sensitive=True)
    assert pos == [(0, 8)]


def test_literal_positions_case_insensitive() -> None:
    text = "Evidence Verification and evidence verification"
    pos = _find_literal_positions(text, "evidence", case_sensitive=False)
    assert len(pos) == 2


def test_literal_percent_and_underscore_are_literal() -> None:
    text = "rate is 100%_done"
    pos = _find_literal_positions(text, "100%_done", case_sensitive=True)
    assert pos == [(8, 17)]


def test_unicode_offsets() -> None:
    text = "café evidence"
    pos = _find_literal_positions(text, "café", case_sensitive=True)
    assert pos[0] == (0, 4)
    assert _excerpt(text, 0, 4).startswith("café")
