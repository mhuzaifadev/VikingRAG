"""Authoritative Read primitive - never returns generated abstracts as source."""

from __future__ import annotations

import time

from vikingrag.application.budget import RetrievalContext
from vikingrag.application.navigation import StructuralNavigationService
from vikingrag.domain.errors import ScopeDeniedError, StaleSourceError, ValidationDomainError
from vikingrag.domain.models.document import DocumentId
from vikingrag.domain.models.primitives import OffsetSystem, ReadRequest, ReadResponse
from vikingrag.domain.uri import VikingURIParser
from vikingrag.infrastructure.database.engine import Database
from vikingrag.infrastructure.database.repositories.document import SqlDocumentRepository
from vikingrag.infrastructure.database.repositories.node import SqlNodeRepository
from vikingrag.ingestion.tokenization import Tokenizer, create_tokenizer
from vikingrag.observability.logging import get_logger

logger = get_logger(__name__)


class ReadService:
    def __init__(
        self,
        *,
        database: Database,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._database = database
        self._tokenizer = tokenizer or create_tokenizer()

    async def read(
        self,
        request: ReadRequest,
        *,
        ctx: RetrievalContext | None = None,
    ) -> ReadResponse:
        context = ctx or RetrievalContext.create()
        async with context.tool_call("read"):
            context.check_deadline()
            started = time.perf_counter()
            parsed = VikingURIParser.parse(request.uri)

            async with self._database.session() as session:
                await context.reserve(db_operations=1)
                nav = StructuralNavigationService(
                    documents=SqlDocumentRepository(session),
                    nodes=SqlNodeRepository(session),
                )
                node = await nav.resolve_uri(request.uri)
                if not context.document_allowed(node.document_id):
                    raise ScopeDeniedError(f"Document {node.document_id} outside permitted scope")
                if parsed.document_id != node.document_id:
                    raise ValidationDomainError("URI document_id mismatch")

            content = node.content
            has_direct = bool(content and content.strip())
            if not has_direct:
                elapsed = (time.perf_counter() - started) * 1000.0
                logger.info(
                    "read_no_direct_content",
                    query_id=str(context.query_id),
                    trace_id=context.trace_id,
                    node_type=node.node_type.value,
                    latency_ms=elapsed,
                )
                return ReadResponse(
                    uri=node.uri,
                    document_id=DocumentId(node.document_id),
                    node_id=node.id,
                    node_type=node.node_type,
                    title=node.title,
                    content_hash=node.content_hash,
                    has_direct_content=False,
                    text="",
                    start_offset=0,
                    end_offset=0,
                    offset_system=OffsetSystem.UNICODE_CODE_POINT,
                    token_count=0,
                    truncated=False,
                    next_offset=None,
                    source_metadata={"note": "no_direct_content"},
                )

            assert content is not None
            if (
                request.expected_content_hash is not None
                and request.expected_content_hash != node.content_hash
            ):
                raise StaleSourceError(
                    f"Content hash mismatch for {node.uri}: expected "
                    f"{request.expected_content_hash}, got {node.content_hash}"
                )

            if request.start_offset > len(content):
                raise ValidationDomainError("start_offset beyond content length")

            # Token-bounded excerpt starting at unicode code-point offset
            text, end_offset, truncated = _slice_by_tokens(
                content,
                start_offset=request.start_offset,
                max_tokens=request.max_tokens,
                tokenizer=self._tokenizer,
            )
            token_count = self._tokenizer.count(text) if text else 0
            if token_count > 0:
                await context.reserve(read_tokens=token_count)

            next_offset = end_offset if truncated else None
            elapsed = (time.perf_counter() - started) * 1000.0
            logger.info(
                "read_completed",
                query_id=str(context.query_id),
                trace_id=context.trace_id,
                token_count=token_count,
                truncated=truncated,
                latency_ms=elapsed,
            )
            return ReadResponse(
                uri=node.uri,
                document_id=DocumentId(node.document_id),
                node_id=node.id,
                node_type=node.node_type,
                title=node.title,
                content_hash=node.content_hash,
                has_direct_content=True,
                text=text,
                start_offset=request.start_offset,
                end_offset=end_offset,
                offset_system=OffsetSystem.UNICODE_CODE_POINT,
                token_count=token_count,
                truncated=truncated,
                next_offset=next_offset,
                source_metadata={
                    "title": node.title,
                    "node_type": node.node_type.value,
                },
            )


def _slice_by_tokens(
    content: str,
    *,
    start_offset: int,
    max_tokens: int,
    tokenizer: Tokenizer,
) -> tuple[str, int, bool]:
    """Return (text, end_offset, truncated) never exceeding max_tokens.

    Uses encode→clip→decode when available, then recounts. Falls back to
    code-point walking with tokenizer.count (no *4 char overshoot).
    """
    if max_tokens < 1:
        raise ValidationDomainError("max_tokens must be >= 1")
    remainder = content[start_offset:]
    if not remainder:
        return "", start_offset, False

    # Single code point that alone exceeds budget → explicit inability
    first_cp = remainder[0]
    if tokenizer.count(first_cp) > max_tokens:
        raise ValidationDomainError(
            "Single code point exceeds max_tokens; cannot return a conforming excerpt"
        )

    # Prefer encode/decode when reversible; always recount and fall back to
    # code-point walk if the approximate tokenizer under-counts via encode.
    try:
        tokens = tokenizer.encode(remainder)
        if len(tokens) <= max_tokens and tokenizer.count(remainder) <= max_tokens:
            return remainder, start_offset + len(remainder), False
        if len(tokens) > max_tokens:
            n = max_tokens
            while n > 0:
                try:
                    clipped = tokenizer.decode(tokens[:n])
                except NotImplementedError:
                    break
                if not clipped:
                    n -= 1
                    continue
                if not remainder.startswith(clipped):
                    clipped = _longest_prefix_within_tokens(remainder, max_tokens, tokenizer)
                if clipped and tokenizer.count(clipped) <= max_tokens:
                    end = start_offset + len(clipped)
                    return clipped, end, end < start_offset + len(remainder)
                n -= 1
    except NotImplementedError:
        pass

    clipped = _longest_prefix_within_tokens(remainder, max_tokens, tokenizer)
    end = start_offset + len(clipped)
    return clipped, end, end < start_offset + len(remainder)


def _longest_prefix_within_tokens(text: str, max_tokens: int, tokenizer: Tokenizer) -> str:
    """Binary-search the longest unicode prefix whose token count <= max_tokens."""
    if not text:
        return ""
    if tokenizer.count(text) <= max_tokens:
        return text
    lo, hi = 1, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid]
        if tokenizer.count(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best
