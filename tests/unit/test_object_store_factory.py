"""Object store backend selection."""

from __future__ import annotations

from pathlib import Path

import pytest

from vikingrag.domain.errors import InfrastructureError
from vikingrag.infrastructure.object_store.factory import build_object_store
from vikingrag.infrastructure.object_store.local import LocalObjectStore
from vikingrag.infrastructure.object_store.s3 import S3ObjectStore
from vikingrag.settings.config import ObjectStoreBackend, ObjectStoreSettings


def test_build_local(tmp_path: Path) -> None:
    store = build_object_store(
        ObjectStoreSettings(backend=ObjectStoreBackend.LOCAL, local_root=str(tmp_path))
    )
    assert isinstance(store, LocalObjectStore)


def test_build_s3() -> None:
    store = build_object_store(
        ObjectStoreSettings(
            backend=ObjectStoreBackend.S3,
            s3_access_key="ak",
            s3_secret_key="sk",
            s3_bucket="b",
            s3_endpoint="http://localhost:9000",
        )
    )
    assert isinstance(store, S3ObjectStore)


def test_build_s3_requires_keys() -> None:
    with pytest.raises(InfrastructureError):
        build_object_store(ObjectStoreSettings(backend=ObjectStoreBackend.S3))
