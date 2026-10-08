"""SDK AnswerRequest learning_policy wiring (no live DB)."""

from __future__ import annotations

from vikingrag.domain.models.answer import AnswerRequest, ExecutionMode
from vikingrag.domain.models.experience import LearningPolicy


def test_answer_request_defaults_learn() -> None:
    req = AnswerRequest(question="What is the fee?")
    assert req.learning_policy is LearningPolicy.LEARN
    assert req.execution_mode is ExecutionMode.VIKINGRAG


def test_answer_request_accepts_frozen() -> None:
    req = AnswerRequest(
        question="Held-out question",
        learning_policy=LearningPolicy.FROZEN,
        execution_mode=ExecutionMode.VIKINGRAG_E_PLUS,
    )
    assert req.learning_policy is LearningPolicy.FROZEN
    assert req.execution_mode is ExecutionMode.VIKINGRAG_E_PLUS
