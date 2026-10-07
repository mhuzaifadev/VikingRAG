"""Dispatch validated tool calls to Search / List / Grep / Read primitives."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from vikingrag.application.budget import RetrievalContext
from vikingrag.application.grep_primitive import GrepService
from vikingrag.application.list_primitive import ListService
from vikingrag.application.read_primitive import ReadService
from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.errors import ValidationDomainError
from vikingrag.domain.models.primitives import GrepRequest, ListRequest, ReadRequest
from vikingrag.domain.models.representation import SearchRequest
from vikingrag.providers.llm.base import ToolCall
from vikingrag.providers.llm.tools import RETRIEVAL_TOOL_NAMES


@dataclass(slots=True)
class ToolExecutionRecord:
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    result_uris: tuple[str, ...] = ()
    ok: bool = True
    error: str | None = None


@dataclass(slots=True)
class ToolExecutorState:
    searched: bool = False
    read_uris: set[str] = field(default_factory=set)
    events: list[ToolExecutionRecord] = field(default_factory=list)
    fingerprints: list[str] = field(default_factory=list)


class RetrievalToolExecutor:
    def __init__(
        self,
        *,
        search: SemanticSearchService | None = None,
        list_service: ListService | None = None,
        grep: GrepService | None = None,
        read: ReadService | None = None,
        search_plus: Any | None = None,
        use_search_plus: bool = False,
    ) -> None:
        self._search = search
        self._list = list_service
        self._grep = grep
        self._read = read
        self._search_plus = search_plus
        self._use_search_plus = use_search_plus
        self.state = ToolExecutorState()

    async def execute(
        self,
        call: ToolCall,
        *,
        ctx: RetrievalContext,
        default_top_k: int = 8,
        default_read_tokens: int = 512,
    ) -> ToolExecutionRecord:
        if call.name not in RETRIEVAL_TOOL_NAMES:
            rec = ToolExecutionRecord(
                tool_call_id=call.id,
                name=call.name,
                arguments=dict(call.arguments),
                result={"error": f"unknown tool {call.name}"},
                ok=False,
                error="unknown_tool",
            )
            self.state.events.append(rec)
            return rec
        if not call.arguments_valid:
            rec = ToolExecutionRecord(
                tool_call_id=call.id,
                name=call.name,
                arguments={},
                result={"error": "malformed_arguments", "raw": call.arguments_raw},
                ok=False,
                error="malformed_arguments",
            )
            self.state.events.append(rec)
            return rec

        try:
            if call.name == "Stop":
                result = {
                    "stopped": True,
                    "reason": str(call.arguments.get("reason", "")),
                    "sufficient": bool(call.arguments.get("sufficient", False)),
                }
                uris: tuple[str, ...] = ()
            elif call.name == "Search":
                result, uris = await self._do_search(call.arguments, ctx=ctx, top_k=default_top_k)
                self.state.searched = True
            elif call.name == "List":
                result, uris = await self._do_list(call.arguments, ctx=ctx)
            elif call.name == "Grep":
                result, uris = await self._do_grep(call.arguments, ctx=ctx)
            elif call.name == "Read":
                result, uris = await self._do_read(
                    call.arguments, ctx=ctx, default_read_tokens=default_read_tokens
                )
                if uris:
                    self.state.read_uris.update(uris)
            else:
                raise ValidationDomainError(f"Unhandled tool {call.name}")
            fp = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, default=str)}"
            self.state.fingerprints.append(fp)
            rec = ToolExecutionRecord(
                tool_call_id=call.id,
                name=call.name,
                arguments=dict(call.arguments),
                result=result,
                result_uris=uris,
                ok=True,
            )
        except Exception as exc:
            rec = ToolExecutionRecord(
                tool_call_id=call.id,
                name=call.name,
                arguments=dict(call.arguments),
                result={"error": type(exc).__name__, "message": str(exc)},
                ok=False,
                error=type(exc).__name__,
            )
        self.state.events.append(rec)
        return rec

    def no_progress(self, *, window: int = 4) -> bool:
        fps = self.state.fingerprints
        if len(fps) < window:
            return False
        recent = fps[-window:]
        return len(set(recent)) <= 1

    async def _do_search(
        self, args: dict[str, Any], *, ctx: RetrievalContext, top_k: int
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        query = str(args.get("query", "")).strip()
        if not query:
            raise ValidationDomainError("Search requires query")
        k = int(args.get("top_k") or top_k)
        scope_raw = args.get("scope_uri") or args.get("uri")
        scope_uri = str(scope_raw).strip() if scope_raw else None
        search_req = SearchRequest(query=query, top_k=k, scope_uri=scope_uri)
        if self._use_search_plus and self._search_plus is not None:
            response = await self._search_plus.search(search_req, ctx=ctx)
            # SearchPlusResponse exposes candidates via `.base`; plain SearchResponse has them top-level
            base = getattr(response, "base", response)
            candidates = base.candidates
            expanded = [
                {"uri": e.target_uri, "similarity": e.similarity, "hop": e.hop}
                for e in getattr(response, "expansions", ())
            ]
        else:
            if self._search is None:
                raise ValidationDomainError("Search service not configured")
            response = await self._search.search(search_req, ctx=ctx)
            candidates = response.candidates
            expanded = []
        hits = [
            {
                "uri": h.uri,
                "title": h.title,
                "score": h.score,
                "preview": h.preview,
                "node_type": h.node_type.value,
            }
            for h in candidates
        ]
        seed_uris = tuple(h.uri for h in candidates)
        expand_uris = tuple(e["uri"] for e in expanded)
        # Deduped seeds + expansions for result_refs / U_edge tracking
        seen: set[str] = set()
        uris_list: list[str] = []
        for u in (*seed_uris, *expand_uris):
            if u not in seen:
                seen.add(u)
                uris_list.append(u)
        return {
            "hits": hits,
            "expansions": expanded,
            "count": len(hits),
            "cold_path": bool(getattr(response, "cold_path", False)),
        }, tuple(uris_list)

    async def _do_list(
        self, args: dict[str, Any], *, ctx: RetrievalContext
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        if self._list is None:
            raise ValidationDomainError("List service not configured")
        uri = str(args.get("uri", "")).strip()
        limit = int(args.get("limit") or 50)
        cursor = args.get("cursor")
        response = await self._list.list(
            ListRequest(uri=uri, limit=limit, cursor=str(cursor) if cursor else None),
            ctx=ctx,
        )
        items = [
            {
                "uri": i.uri,
                "title": i.title,
                "node_type": i.node_type.value,
                "has_content": i.has_content,
            }
            for i in response.items
        ]
        return {
            "items": items,
            "next_cursor": response.next_cursor,
            "truncated": response.truncated,
        }, tuple(i.uri for i in response.items)

    async def _do_grep(
        self, args: dict[str, Any], *, ctx: RetrievalContext
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        if self._grep is None:
            raise ValidationDomainError("Grep service not configured")
        response = await self._grep.grep(
            GrepRequest(
                uri=str(args["uri"]),
                pattern=str(args["pattern"]),
                case_sensitive=bool(args.get("case_sensitive", True)),
                max_matches=int(args.get("max_matches") or 20),
                mode=str(args.get("mode") or "literal"),
            ),
            ctx=ctx,
        )
        matches = [
            {
                "uri": m.uri,
                "excerpt": m.excerpt,
                "start_offset": m.start_offset,
                "end_offset": m.end_offset,
                "content_hash": m.content_hash,
            }
            for m in response.matches
        ]
        return {
            "matches": matches,
            "truncated": response.truncated,
        }, tuple(m.uri for m in response.matches)

    async def _do_read(
        self,
        args: dict[str, Any],
        *,
        ctx: RetrievalContext,
        default_read_tokens: int,
    ) -> tuple[dict[str, Any], tuple[str, ...]]:
        if self._read is None:
            raise ValidationDomainError("Read service not configured")
        uri = str(args["uri"])
        response = await self._read.read(
            ReadRequest(
                uri=uri,
                start_offset=int(args.get("start_offset") or 0),
                max_tokens=int(args.get("max_tokens") or default_read_tokens),
                expected_content_hash=args.get("expected_content_hash"),
            ),
            ctx=ctx,
        )
        return {
            "uri": response.uri,
            "text": response.text,
            "content_hash": response.content_hash,
            "start_offset": response.start_offset,
            "end_offset": response.end_offset,
            "token_count": response.token_count,
            "truncated": response.truncated,
            "has_direct_content": response.has_direct_content,
            "next_offset": response.next_offset,
        }, (response.uri,) if response.has_direct_content else ()
