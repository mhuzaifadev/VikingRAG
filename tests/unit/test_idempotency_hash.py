from __future__ import annotations

from vikingrag.ingestion.hashing import hash_bytes, hash_text


def test_hash_stable() -> None:
    assert hash_bytes(b"abc") == hash_bytes(b"abc")
    assert hash_bytes(b"abc") != hash_bytes(b"abd")
    assert hash_text("hello") == hash_bytes(b"hello")
