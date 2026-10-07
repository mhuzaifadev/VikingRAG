"""Summary generator protocol - no vendor SDK imports."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from vikingrag.domain.models.representation import SummaryRequest, SummaryResult


@runtime_checkable
class SummaryGenerator(Protocol):
    """Produces compact hierarchical abstracts for structural nodes."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    @property
    def version(self) -> str: ...

    async def summarize(self, request: SummaryRequest) -> SummaryResult: ...
