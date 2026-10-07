"""Opaque pagination cursors for List/Grep - validated against request scope."""

from __future__ import annotations

import base64
import json
from typing import Any
from uuid import UUID

from vikingrag.domain.errors import ValidationDomainError


def encode_cursor(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii")


def decode_cursor(cursor: str) -> dict[str, Any]:
    try:
        raw = base64.urlsafe_b64decode(cursor.encode("ascii"))
        data = json.loads(raw.decode("utf-8"))
    except (ValueError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValidationDomainError("Invalid pagination cursor") from exc
    if not isinstance(data, dict):
        raise ValidationDomainError("Invalid pagination cursor")
    return data


def require_uuid(value: Any, *, field: str) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError) as exc:
        raise ValidationDomainError(f"Invalid cursor field: {field}") from exc
