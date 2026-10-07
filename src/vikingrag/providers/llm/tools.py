"""Canonical VikingRAG retrieval tool definitions for Algorithm 1."""

from __future__ import annotations

from vikingrag.providers.llm.base import ToolDefinition, ToolFunctionSpec

SEARCH_TOOL = ToolDefinition(
    function=ToolFunctionSpec(
        name="Search",
        description=(
            "Semantic discovery over indexed document hierarchy. "
            "Returns compact URI-addressable hits (not authoritative evidence). "
            "Use before Read when the location of evidence is unknown."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural-language search query",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Maximum hits to return",
                    "minimum": 1,
                },
                "uri": {
                    "type": "string",
                    "description": (
                        "Optional directory/subtree scope URI. When set, Search "
                        "only returns hits under that node (path containment)."
                    ),
                },
                "scope_uri": {
                    "type": "string",
                    "description": "Alias for uri — subtree scope for semantic Search",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
)

LIST_TOOL = ToolDefinition(
    function=ToolFunctionSpec(
        name="List",
        description=(
            "List direct children of a structural URI (document/section). "
            "Use for cheap neighborhood inspection."
        ),
        parameters={
            "type": "object",
            "properties": {
                "uri": {
                    "type": "string",
                    "description": "Parent node or document URI",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max children to return",
                    "minimum": 1,
                },
                "cursor": {
                    "type": "string",
                    "description": "Pagination cursor from a prior List response",
                },
            },
            "required": ["uri"],
            "additionalProperties": False,
        },
    )
)

GREP_TOOL = ToolDefinition(
    function=ToolFunctionSpec(
        name="Grep",
        description=(
            "Scoped lexical match under a URI. Prefer when a concrete term or "
            "identifier is missing from current evidence."
        ),
        parameters={
            "type": "object",
            "properties": {
                "uri": {"type": "string", "description": "Subtree root URI"},
                "pattern": {"type": "string", "description": "Literal or pattern to find"},
                "case_sensitive": {"type": "boolean", "default": True},
                "max_matches": {"type": "integer", "minimum": 1},
                "mode": {
                    "type": "string",
                    "enum": ["literal", "pattern"],
                    "default": "literal",
                },
            },
            "required": ["uri", "pattern"],
            "additionalProperties": False,
        },
    )
)

READ_TOOL = ToolDefinition(
    function=ToolFunctionSpec(
        name="Read",
        description=(
            "Authoritative read of original source content at a URI. "
            "Citations and factual claims must come from Read results, not abstracts alone."
        ),
        parameters={
            "type": "object",
            "properties": {
                "uri": {"type": "string", "description": "Node or chunk URI to read"},
                "start_offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Unicode code-point offset",
                },
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Max tokens to return from the source",
                },
                "expected_content_hash": {
                    "type": "string",
                    "description": "Optional content hash for stale-source detection",
                },
            },
            "required": ["uri"],
            "additionalProperties": False,
        },
    )
)

STOP_TOOL = ToolDefinition(
    function=ToolFunctionSpec(
        name="Stop",
        description=(
            "Terminate retrieval when evidence is sufficient to answer or it is clear "
            "the corpus cannot support an answer. Provide a short reason."
        ),
        parameters={
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Why retrieval should stop",
                },
                "sufficient": {
                    "type": "boolean",
                    "description": "True if evidence is believed sufficient to answer",
                },
            },
            "required": ["reason"],
            "additionalProperties": False,
        },
    )
)

RETRIEVAL_TOOLS: tuple[ToolDefinition, ...] = (
    SEARCH_TOOL,
    LIST_TOOL,
    GREP_TOOL,
    READ_TOOL,
    STOP_TOOL,
)

RETRIEVAL_TOOL_NAMES: frozenset[str] = frozenset(t.function.name for t in RETRIEVAL_TOOLS)


def retrieval_tool_definitions() -> list[ToolDefinition]:
    return list(RETRIEVAL_TOOLS)
