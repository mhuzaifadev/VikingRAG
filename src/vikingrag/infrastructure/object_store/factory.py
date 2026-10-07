"""Select object store adapter from settings."""

from __future__ import annotations

from vikingrag.domain.errors import InfrastructureError
from vikingrag.infrastructure.object_store.base import ObjectStore
from vikingrag.infrastructure.object_store.local import LocalObjectStore
from vikingrag.infrastructure.object_store.s3 import S3ObjectStore
from vikingrag.settings.config import ObjectStoreBackend, ObjectStoreSettings


def build_object_store(settings: ObjectStoreSettings) -> ObjectStore:
    if settings.backend == ObjectStoreBackend.LOCAL:
        return LocalObjectStore(settings.local_root)
    if settings.backend == ObjectStoreBackend.S3:
        if not settings.s3_access_key or not settings.s3_secret_key:
            raise InfrastructureError(
                "VIKINGRAG_OBJECT_STORE_BACKEND=s3 requires s3_access_key and s3_secret_key"
            )
        return S3ObjectStore(
            bucket=settings.s3_bucket,
            access_key=settings.s3_access_key,
            secret_key=settings.s3_secret_key,
            region=settings.s3_region,
            endpoint=settings.s3_endpoint,
        )
    raise InfrastructureError(f"Unsupported object store backend: {settings.backend!r}")
