"""Multi-granular scoring weight helpers."""

from __future__ import annotations

from vikingrag.application.search import SemanticSearchService
from vikingrag.domain.models.document import NodeType
from vikingrag.providers.embeddings.deterministic import DeterministicEmbeddingProvider
from vikingrag.settings.config import EmbeddingSettings, RetrievalSettings


def test_granularity_weights_from_settings() -> None:
    service = SemanticSearchService(
        database=None,  # type: ignore[arg-type]
        embedding_provider=DeterministicEmbeddingProvider(dimensions=32),
        retrieval_settings=RetrievalSettings(
            weight_document=0.5,
            weight_section=0.9,
            weight_subsection=1.0,
            weight_chunk=1.1,
        ),
        embedding_settings=EmbeddingSettings(dimensions=32),
    )
    assert service.granularity_weight(NodeType.DOCUMENT) == 0.5
    assert service.granularity_weight(NodeType.CHUNK) == 1.1
