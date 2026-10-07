"""Minimal S3-compatible object store via httpx (AWS SigV4).

Works with AWS S3 and MinIO when endpoint/credentials are configured.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import UTC, datetime
from urllib.parse import quote

import httpx

from vikingrag.domain.errors import InfrastructureError, NotFoundError


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac_sha256(key: bytes, msg: str) -> bytes:
    return hmac.new(key, msg.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret: str, datestamp: str, region: str, service: str) -> bytes:
    k_date = _hmac_sha256(("AWS4" + secret).encode("utf-8"), datestamp)
    k_region = hmac.new(k_date, region.encode("utf-8"), hashlib.sha256).digest()
    k_service = hmac.new(k_region, service.encode("utf-8"), hashlib.sha256).digest()
    return hmac.new(k_service, b"aws4_request", hashlib.sha256).digest()


def _uri_encode(path: str, *, encode_slash: bool = True) -> str:
    safe = "-_.~" if encode_slash else "-_.~/"
    return quote(path, safe=safe)


class S3ObjectStore:
    """Async S3/MinIO adapter (Put/Get/Head/Delete)."""

    def __init__(
        self,
        *,
        bucket: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
        endpoint: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not access_key or not secret_key:
            raise InfrastructureError("S3 access_key and secret_key are required")
        if not bucket:
            raise InfrastructureError("S3 bucket is required")
        self._bucket = bucket
        self._access_key = access_key
        self._secret_key = secret_key
        self._region = region
        self._endpoint = (endpoint or "").rstrip("/") or None
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _object_url(self, key: str) -> tuple[str, str, str]:
        """Return (url, host, canonical_uri)."""
        encoded_key = _uri_encode(key, encode_slash=False)
        if self._endpoint:
            # Path-style for MinIO / custom endpoints
            host = self._endpoint.removeprefix("https://").removeprefix("http://")
            scheme = "https" if self._endpoint.startswith("https") else "http"
            # host may include port
            url = f"{scheme}://{host}/{self._bucket}/{encoded_key}"
            canonical_uri = f"/{self._bucket}/{encoded_key}"
            return url, host.split("/")[0], canonical_uri
        host = f"{self._bucket}.s3.{self._region}.amazonaws.com"
        url = f"https://{host}/{encoded_key}"
        return url, host, f"/{encoded_key}"

    def _sign(
        self,
        *,
        method: str,
        host: str,
        canonical_uri: str,
        payload: bytes,
        content_type: str | None,
        amz_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        now = datetime.now(UTC)
        amz_date = now.strftime("%Y%m%dT%H%M%SZ")
        datestamp = now.strftime("%Y%m%d")
        payload_hash = _sha256_hex(payload)
        headers: dict[str, str] = {
            "host": host,
            "x-amz-content-sha256": payload_hash,
            "x-amz-date": amz_date,
        }
        if content_type:
            headers["content-type"] = content_type
        if amz_headers:
            headers.update({k.lower(): v for k, v in amz_headers.items()})

        signed_header_keys = sorted(headers)
        canonical_headers = "".join(f"{k}:{headers[k]}\n" for k in signed_header_keys)
        signed_headers = ";".join(signed_header_keys)
        canonical_request = "\n".join(
            [
                method,
                canonical_uri,
                "",  # query string
                canonical_headers,
                signed_headers,
                payload_hash,
            ]
        )
        credential_scope = f"{datestamp}/{self._region}/s3/aws4_request"
        string_to_sign = "\n".join(
            [
                "AWS4-HMAC-SHA256",
                amz_date,
                credential_scope,
                _sha256_hex(canonical_request.encode("utf-8")),
            ]
        )
        signature = hmac.new(
            _signing_key(self._secret_key, datestamp, self._region, "s3"),
            string_to_sign.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        headers["authorization"] = (
            f"AWS4-HMAC-SHA256 Credential={self._access_key}/{credential_scope}, "
            f"SignedHeaders={signed_headers}, Signature={signature}"
        )
        return headers

    async def _request(
        self,
        method: str,
        key: str,
        *,
        data: bytes = b"",
        content_type: str | None = None,
    ) -> httpx.Response:
        if not key or key.startswith("/") or ".." in key.split("/"):
            raise InfrastructureError(f"Invalid object key: {key!r}")
        url, host, canonical_uri = self._object_url(key)
        headers = self._sign(
            method=method,
            host=host,
            canonical_uri=canonical_uri,
            payload=data,
            content_type=content_type,
        )
        try:
            response = await self._client.request(method, url, content=data, headers=headers)
        except httpx.HTTPError as exc:
            raise InfrastructureError(f"S3 request failed: {exc}") from exc
        return response

    async def put(self, key: str, data: bytes, *, content_type: str | None = None) -> None:
        response = await self._request(
            "PUT", key, data=data, content_type=content_type or "application/octet-stream"
        )
        if response.status_code not in {200, 201}:
            raise InfrastructureError(
                f"S3 put failed status={response.status_code}: {response.text[:200]}"
            )

    async def get(self, key: str) -> bytes:
        response = await self._request("GET", key)
        if response.status_code == 404:
            raise NotFoundError(f"Object not found: {key}")
        if response.status_code != 200:
            raise InfrastructureError(
                f"S3 get failed status={response.status_code}: {response.text[:200]}"
            )
        return response.content

    async def exists(self, key: str) -> bool:
        response = await self._request("HEAD", key)
        if response.status_code == 404:
            return False
        if response.status_code == 200:
            return True
        raise InfrastructureError(
            f"S3 head failed status={response.status_code}: {response.text[:200]}"
        )

    async def delete(self, key: str) -> None:
        response = await self._request("DELETE", key)
        if response.status_code not in {200, 204, 404}:
            raise InfrastructureError(
                f"S3 delete failed status={response.status_code}: {response.text[:200]}"
            )
