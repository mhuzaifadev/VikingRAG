"""Warm-up reads source docs only; never gold QA JSON; can enqueue learning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vikingrag.evaluation.adapters import get_adapter
from vikingrag.evaluation.base import iter_source_documents_from_root
from vikingrag.evaluation.warmup import (
    _corpus_text_sample,
    corpus_excerpt_from_documents,
    plan_warmup,
    run_warmup,
)
from vikingrag.providers.llm.base import ChatMessage, LLMResponse


class _ScriptedLLM:
    model = "scripted"

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> LLMResponse:
        del model, temperature, max_tokens
        # Ensure poison gold never appears in the prompt corpus excerpt.
        blob = "\n".join((m.content or "") for m in messages)
        assert "POISON_GOLD_STRING_NEVER_IN_CORPUS" not in blob
        return LLMResponse(
            content=json.dumps({"questions": ["What is the fee mentioned in corpus A?"]}),
            model=self.model,
        )


def test_iter_source_documents_skips_gold_json(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("Late fee is twenty-five dollars.\n", encoding="utf-8")
    (tmp_path / "qa.json").write_text(
        json.dumps({"questions": ["POISON_GOLD_STRING_NEVER_IN_CORPUS"]}),
        encoding="utf-8",
    )
    (tmp_path / "qa.jsonl").write_text(
        '{"question":"POISON_GOLD_STRING_NEVER_IN_CORPUS"}\n',
        encoding="utf-8",
    )
    docs = list(iter_source_documents_from_root(tmp_path))
    assert len(docs) == 1
    assert docs[0].document_id == "corpus/a.md"
    excerpt = corpus_excerpt_from_documents(docs)
    assert "twenty-five" in excerpt
    assert "POISON_GOLD_STRING_NEVER_IN_CORPUS" not in excerpt
    assert "POISON" not in _corpus_text_sample(tmp_path)


def test_warmup_poison_json_and_learning_enqueue(tmp_path: Path) -> None:
    root = tmp_path / "syllabusqa"
    (root / "corpus").mkdir(parents=True)
    (root / "corpus" / "a.md").write_text(
        "The course late fee is $25.\n",
        encoding="utf-8",
    )
    (root / "qa.json").write_text(
        json.dumps({"gold": "POISON_GOLD_STRING_NEVER_IN_CORPUS"}),
        encoding="utf-8",
    )

    enqueues: list[str] = []

    async def _answer(question: str) -> dict[str, Any]:
        enqueues.append(question)
        return {"answer": "ok", "citations": ()}

    drain_calls = {"n": 0}

    async def _drain() -> tuple[int, int]:
        drain_calls["n"] += 1
        if drain_calls["n"] == 1:
            return 1, 2
        return 0, 0

    plan = plan_warmup(dataset="syllabusqa", m=1, data_dir=str(tmp_path))
    adapter = get_adapter("syllabusqa")
    assert adapter.is_present(tmp_path)

    result = run_warmup(
        plan,
        llm_configured=True,
        corpus_present=True,
        llm=_ScriptedLLM(),  # type: ignore[arg-type]
        answer_fn=_answer,
        drain_fn=_drain,
        materialize=True,
    )
    assert result.status == "completed"
    assert result.m_generated == 1
    assert len(enqueues) == 1
    assert result.jobs_drained == 1
    assert result.edges_built == 2
    payload = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert payload["measured_scores"] is None
    assert payload["jobs_drained"] == 1
    assert payload["edges_built"] == 2
    assert "POISON_GOLD_STRING_NEVER_IN_CORPUS" not in json.dumps(payload)
