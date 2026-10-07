"""Domain models and errors - framework-agnostic."""

from vikingrag.domain.errors import (
    DocumentAlreadyExists,
    DocumentNotFound,
    DocumentParseError,
    DomainError,
    HierarchyConstructionError,
    InvalidVikingURI,
    NodeNotFound,
    NotImplementedCapabilityError,
    UnsupportedDocumentType,
)
from vikingrag.domain.models import (
    ChunkId,
    DocumentId,
    EvidenceBundle,
    NodeId,
    RetrievalBudget,
    RetrievalCandidate,
    RetrievalQuery,
    RetrievalTrace,
    RetrievedEvidence,
)
from vikingrag.domain.models.document import (
    AbstractStatus,
    DocumentStatus,
    IngestionStrategy,
    NodeType,
)
from vikingrag.domain.models.node import DocumentNode, TreeNode

__all__ = [
    "AbstractStatus",
    "ChunkId",
    "DocumentAlreadyExists",
    "DocumentId",
    "DocumentNode",
    "DocumentNotFound",
    "DocumentParseError",
    "DocumentStatus",
    "DomainError",
    "EvidenceBundle",
    "HierarchyConstructionError",
    "IngestionStrategy",
    "InvalidVikingURI",
    "NodeId",
    "NodeNotFound",
    "NodeType",
    "NotImplementedCapabilityError",
    "RetrievalBudget",
    "RetrievalCandidate",
    "RetrievalQuery",
    "RetrievalTrace",
    "RetrievedEvidence",
    "TreeNode",
    "UnsupportedDocumentType",
]
