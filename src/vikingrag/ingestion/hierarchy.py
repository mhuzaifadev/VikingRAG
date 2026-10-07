"""Build an in-memory hierarchy draft from normalized content blocks."""

from __future__ import annotations

from uuid import uuid4

from vikingrag.domain.errors import HierarchyConstructionError
from vikingrag.domain.models.document import NodeType
from vikingrag.ingestion.types import HierarchyDraft, HierarchyDraftNode, ParsedDocument


def build_hierarchy(parsed: ParsedDocument) -> HierarchyDraft:
    if not parsed.blocks:
        raise HierarchyConstructionError("Parsed document has no content blocks")

    root_id = uuid4()
    nodes: list[HierarchyDraftNode] = [
        HierarchyDraftNode(
            temp_id=root_id,
            parent_temp_id=None,
            node_type=NodeType.DOCUMENT,
            title=parsed.title,
            ordinal=0,
            depth=0,
            text=None,
            metadata={"role": "document_root"},
        )
    ]

    # Stack entries: (heading_level, temp_id, child_ordinal_counter)
    # heading_level 0 = document root
    stack: list[tuple[int, object, list[int]]] = [(0, root_id, [0])]
    pending_body: list[str] = []
    saw_matching_h1 = False

    def flush_body() -> None:
        nonlocal pending_body
        if not pending_body:
            return
        parent_level, parent_id, counter = stack[-1]
        text = "\n\n".join(pending_body).strip()
        pending_body = []
        if not text:
            return
        parent_node = next(n for n in nodes if n.temp_id == parent_id)
        if parent_node.node_type is NodeType.DOCUMENT:
            sec_id = uuid4()
            ordinal = counter[0]
            counter[0] += 1
            nodes.append(
                HierarchyDraftNode(
                    temp_id=sec_id,
                    parent_temp_id=parent_id,  # type: ignore[arg-type]
                    node_type=NodeType.SECTION,
                    title="Introduction",
                    ordinal=ordinal,
                    depth=parent_level + 1,
                    text=text,
                    metadata={"inferred": True},
                )
            )
        elif parent_node.text:
            parent_node.text = f"{parent_node.text}\n\n{text}"
        else:
            parent_node.text = text

    for block in parsed.blocks:
        if block.level > 0 and block.title:
            # First H1 matching document title is the root - do not duplicate it.
            if block.level == 1 and block.title == parsed.title and not saw_matching_h1:
                saw_matching_h1 = True
                flush_body()
                continue

            flush_body()
            while len(stack) > 1 and stack[-1][0] >= block.level:
                stack.pop()
            parent_level, parent_id, counter = stack[-1]
            ordinal = counter[0]
            counter[0] += 1
            node_id = uuid4()
            node_type = _node_type_for_level(block.level)
            depth = parent_level + 1
            meta = dict(block.metadata)
            if block.page is not None:
                meta["page"] = block.page
            if block.offset_start is not None:
                meta["offset_start"] = block.offset_start
            if block.offset_end is not None:
                meta["offset_end"] = block.offset_end
            nodes.append(
                HierarchyDraftNode(
                    temp_id=node_id,
                    parent_temp_id=parent_id,  # type: ignore[arg-type]
                    node_type=node_type,
                    title=block.title,
                    ordinal=ordinal,
                    depth=depth,
                    text=None,
                    metadata=meta,
                )
            )
            stack.append((block.level, node_id, [0]))
        elif block.text.strip():
            pending_body.append(block.text.strip())

    flush_body()

    if len(nodes) == 1:
        body = "\n\n".join(b.text for b in parsed.blocks if b.text.strip()).strip()
        sec_id = uuid4()
        nodes.append(
            HierarchyDraftNode(
                temp_id=sec_id,
                parent_temp_id=root_id,
                node_type=NodeType.SECTION,
                title="Content",
                ordinal=0,
                depth=1,
                text=body or None,
                metadata={"inferred": True},
            )
        )

    return HierarchyDraft(document_title=parsed.title, nodes=nodes)


def _node_type_for_level(level: int) -> NodeType:
    if level <= 1:
        return NodeType.SECTION
    if level == 2:
        return NodeType.SUBSECTION
    return NodeType.SUBSECTION
