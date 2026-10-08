"""Evaluation runner wires real VikingRAG modes; never invents LLM accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vikingrag.domain.models.answer import (
    AnswerCitation,
    AnswerRequest,
    AnswerResponse,
    AnswerStatus,
    ExecutionMode,
)
from vikingrag.evaluation.runner import (
    DEFAULT_EVAL_METHODS,
    VIKINGRAG_METHOD_MODES,
    EvalExample,
    build_vikingrag_methods,
    citation_validity_metrics,
    load_examples_from_manifest,
    resolve_methods,
    run_evaluation,
)


class _FakeGenerator:
    def __init__(self) -> None:
        self.calls: list[AnswerRequest] = []

    async def generate(self, request: AnswerRequest, **kwargs: Any) -> AnswerResponse:
        del kwargs
        self.calls.append(request)
        return AnswerResponse(
            query_id=request.query_id,
            status=AnswerStatus.ANSWERED,
            answer=f"answer:{request.execution_mode.value}",
            citations=(AnswerCitation(evidence_id="e1", uri="viking://doc", quote="quote"),),
            execution_mode=request.execution_mode,
            usage={"llm_calls": 1},
        )


class _FakeClient:
    def __init__(self) -> None:
        self.generator = _FakeGenerator()

    def answer_generator(self) -> _FakeGenerator:
        return self.generator


def test_build_vikingrag_methods_maps_execution_modes() -> None:
    client = _FakeClient()
    methods = build_vikingrag_methods(client)
    assert set(methods) == set(DEFAULT_EVAL_METHODS)
    assert set(VIKINGRAG_METHOD_MODES) == set(DEFAULT_EVAL_METHODS)


@pytest.mark.asyncio
async def test_resolve_and_run_non_echo_methods_report_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "examples": [
                    {"id": "q1", "question": "What is the late fee?"},
                    {"id": "q2", "question": "When is the exam?"},
                ]
            }
        ),
        encoding="utf-8",
    )
    examples = load_examples_from_manifest(manifest)
    assert len(examples) == 2

    client = _FakeClient()
    methods = resolve_methods(
        ["vikingrag", "vikingrag_e", "vikingrag_e_plus"],
        client=client,
    )
    assert "echo_placeholder" not in methods

    report = await run_evaluation(examples=examples, methods=methods, judge="scripted")
    assert report.status == "completed"
    assert report.measured_scores is not None
    assert set(report.methods) == set(DEFAULT_EVAL_METHODS)
    assert len(report.results) == 6
    for row in report.results:
        assert row.error is None
        assert row.answer and row.answer.startswith("answer:")
        assert row.latency_ms >= 0.0
        assert row.citation_count == 1
        assert row.usage.get("llm_calls") == 1

    modes_called = {c.execution_mode for c in client.generator.calls}
    assert modes_called == {
        ExecutionMode.VIKINGRAG,
        ExecutionMode.VIKINGRAG_E,
        ExecutionMode.VIKINGRAG_E_PLUS,
    }
    scores = report.measured_scores["vikingrag"]
    assert scores["non_empty_answer_rate"] == 1.0
    assert scores["citation_presence_rate"] == 1.0


@pytest.mark.asyncio
async def test_echo_only_without_client() -> None:
    methods = resolve_methods(["echo"], client=None)
    report = await run_evaluation(
        examples=[EvalExample(example_id="q0", question="hi")],
        methods=methods,
    )
    assert report.status == "completed_unjudged"
    assert report.measured_scores is None
    assert report.results[0].answer is None


def test_citation_validity_metrics_handles_errors() -> None:
    from vikingrag.evaluation.runner import MethodResult

    metrics = citation_validity_metrics(
        [
            MethodResult(
                method="vikingrag",
                example_id="a",
                answer="yes",
                latency_ms=1.0,
                citation_count=2,
            ),
            MethodResult(
                method="vikingrag",
                example_id="b",
                answer=None,
                latency_ms=1.0,
                error="boom",
            ),
        ]
    )
    assert metrics["vikingrag"]["n"] == 2
    assert metrics["vikingrag"]["error_rate"] == 0.5
    assert metrics["vikingrag"]["non_empty_answer_rate"] == 1.0
