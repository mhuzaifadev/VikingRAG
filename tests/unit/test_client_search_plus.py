"""SDK Search+ ownership: shared embedder, explicit E/E+ errors, idempotent close."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vikingrag.application.answer.generate import AnswerGenerator
from vikingrag.client import VikingRAGClient
from vikingrag.domain.errors import NotImplementedCapabilityError
from vikingrag.domain.models.answer import AnswerRequest, ExecutionMode
from vikingrag.settings.config import Settings


@pytest.mark.asyncio
async def test_answer_generator_raises_when_e_plus_without_search_plus() -> None:
    settings = Settings()
    gen = AnswerGenerator(
        llm=MagicMock(),
        search=MagicMock(),
        list_service=MagicMock(),
        grep_service=MagicMock(),
        read_service=MagicMock(),
        settings=settings,
        search_plus=None,
    )
    with pytest.raises(NotImplementedCapabilityError, match="search_plus"):
        await gen.generate(
            AnswerRequest(
                question="What is the fee?",
                execution_mode=ExecutionMode.VIKINGRAG_E_PLUS,
            )
        )


@pytest.mark.asyncio
async def test_answer_generator_raises_when_e_without_search_plus() -> None:
    settings = Settings()
    gen = AnswerGenerator(
        llm=MagicMock(),
        search=MagicMock(),
        list_service=MagicMock(),
        grep_service=MagicMock(),
        read_service=MagicMock(),
        settings=settings,
        search_plus=None,
    )
    with pytest.raises(NotImplementedCapabilityError, match="search_plus"):
        await gen.generate(
            AnswerRequest(
                question="What is the fee?",
                execution_mode=ExecutionMode.VIKINGRAG_E,
            )
        )


def test_require_search_plus_on_client() -> None:
    client = VikingRAGClient(
        settings=Settings(),
        database=MagicMock(),
        embedding=MagicMock(),
        llm=MagicMock(),
        object_store=MagicMock(),
        search=MagicMock(),
        list_service=MagicMock(),
        grep_service=MagicMock(),
        read_service=MagicMock(),
        search_plus=None,
        search_plus_error="boom",
        _owns_database=False,
    )
    with pytest.raises(NotImplementedCapabilityError, match="boom"):
        client.require_search_plus(ExecutionMode.VIKINGRAG_E_PLUS)
    # Base mode does not require Search+.
    assert client.require_search_plus(ExecutionMode.VIKINGRAG) is None


@pytest.mark.asyncio
async def test_aclose_idempotent() -> None:
    closed: list[str] = []

    class _Emb:
        async def aclose(self) -> None:
            closed.append("emb")

    class _Store:
        async def aclose(self) -> None:
            closed.append("store")

    class _Db:
        async def dispose(self) -> None:
            closed.append("db")

    client = VikingRAGClient(
        settings=Settings(),
        database=_Db(),  # type: ignore[arg-type]
        embedding=_Emb(),  # type: ignore[arg-type]
        llm=None,
        object_store=_Store(),
        search=MagicMock(),
        list_service=MagicMock(),
        grep_service=MagicMock(),
        read_service=MagicMock(),
        search_plus=None,
        _owns_database=True,
    )
    await client.aclose()
    await client.aclose()
    assert closed.count("emb") == 1
    assert closed.count("store") == 1
    assert closed.count("db") == 1


def test_build_search_plus_reuses_shared_embedding() -> None:
    from vikingrag.application.search_plus import build_search_plus

    emb = object()
    search = MagicMock()
    search._database = MagicMock()
    settings = Settings()
    plus = build_search_plus(
        settings,
        MagicMock(),
        search=search,
        embedding_provider=emb,  # type: ignore[arg-type]
    )
    assert plus._embeddings is emb
    assert plus._search is search
