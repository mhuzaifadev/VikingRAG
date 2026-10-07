"""Request correlation context for structured logs."""

from __future__ import annotations

from contextvars import ContextVar
from uuid import uuid4

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("trace_id", default=None)


def bind_request_context(
    *, request_id: str | None = None, trace_id: str | None = None
) -> tuple[str, str]:
    rid = request_id or str(uuid4())
    tid = trace_id or rid
    _request_id.set(rid)
    _trace_id.set(tid)
    return rid, tid


def clear_request_context() -> None:
    _request_id.set(None)
    _trace_id.set(None)


def get_request_id() -> str | None:
    return _request_id.get()


def get_trace_id() -> str | None:
    return _trace_id.get()
