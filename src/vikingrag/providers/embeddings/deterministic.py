"""Deterministic, offline embedding provider for tests and local eval."""

from __future__ import annotations

import hashlib
import math
import re
import time

from vikingrag.providers.embeddings.base import EmbeddingResult, EmbeddingUsage

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)


class DeterministicEmbeddingProvider:
    """Hash-based bag-of-tokens embeddings - reproducible, no network.

    Similar texts that share tokens land closer in cosine space than unrelated texts.
    """

    def __init__(
        self,
        *,
        dimensions: int = 1536,
        model: str = "deterministic-hash-v1",
        provider_name: str = "deterministic",
    ) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self._dimensions = dimensions
        self._model = model
        self._provider_name = provider_name

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def aclose(self) -> None:
        return None

    async def embed_text(self, text: str, *, model: str | None = None) -> EmbeddingResult:
        return await self.embed([text], model=model)

    async def embed_batch(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        return await self.embed(texts, model=model)

    async def embed(self, texts: list[str], *, model: str | None = None) -> EmbeddingResult:
        started = time.perf_counter()
        vectors = [self._embed_one(text) for text in texts]
        return EmbeddingResult(
            vectors=vectors,
            model=model or self._model,
            dimensions=self._dimensions,
            usage=EmbeddingUsage(prompt_tokens=sum(len(_tokenize(t)) for t in texts)),
            latency_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dimensions
        tokens = _tokenize(text)
        if not tokens:
            # Stable non-zero fallback so empty texts are comparable
            digest = hashlib.sha256(b"").digest()
            for i in range(self._dimensions):
                vec[i] = ((digest[i % len(digest)] / 255.0) * 2.0) - 1.0
            return _l2_normalize(vec)

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            # Two hashed buckets per token for denser coverage
            for offset in (0, 16):
                idx = int.from_bytes(digest[offset : offset + 4], "big") % self._dimensions
                sign = 1.0 if digest[(offset + 4) % len(digest)] % 2 == 0 else -1.0
                weight = 1.0 + (digest[(offset + 5) % len(digest)] / 255.0)
                vec[idx] += sign * weight
        return _l2_normalize(vec)


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-12:
        return vector
    return [v / norm for v in vector]
