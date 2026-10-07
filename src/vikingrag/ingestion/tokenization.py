"""Token counting abstraction - prefer tiktoken; fall back to a deterministic estimator."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    def count(self, text: str) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class ApproxWhitespaceTokenizer:
    """Deterministic fallback when tiktoken is unavailable.

    Approximates GPT-style BPE density (~0.75 words per token for English prose).
    """

    def count(self, text: str) -> int:
        if not text:
            return 0
        words = len(text.split())
        # chars/4 is a common heuristic; blend with word count for short strings
        return max(1, max(words, len(text) // 4))

    def encode(self, text: str) -> list[int]:
        # Synthetic token ids - only used for splitting boundaries
        parts = text.split(" ") if text else []
        return list(range(len(parts))) if parts else ([0] if text else [])

    def decode(self, tokens: list[int]) -> str:
        # Not reversible for the approx tokenizer; callers should prefer slice by char
        raise NotImplementedError("ApproxWhitespaceTokenizer cannot decode token ids")


class TikTokenTokenizer:
    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        import tiktoken

        self._enc = tiktoken.get_encoding(encoding_name)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self._enc.encode(text))

    def encode(self, text: str) -> list[int]:
        return list(self._enc.encode(text))

    def decode(self, tokens: list[int]) -> str:
        return self._enc.decode(tokens)


def create_tokenizer() -> Tokenizer:
    try:
        return TikTokenTokenizer()
    except Exception:  # pragma: no cover - environment without tiktoken
        return ApproxWhitespaceTokenizer()
