"""Domain-level errors. Never silently fake unsupported capabilities."""

from __future__ import annotations


class DomainError(Exception):
    """Base error for domain and application failures."""

    def __init__(self, message: str, *, code: str = "domain_error") -> None:
        super().__init__(message)
        self.message = message
        self.code = code


class NotFoundError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="not_found")


class ConflictError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="conflict")


class ValidationDomainError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="validation_error")


class NotImplementedCapabilityError(DomainError):
    """Raised when a Phase-later capability is invoked before implementation."""

    def __init__(self, capability: str) -> None:
        super().__init__(
            f"Capability '{capability}' is not implemented yet",
            code="not_implemented",
        )
        self.capability = capability


class InfrastructureError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="infrastructure_error")


class UnsupportedDocumentType(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsupported_document_type")


class DocumentParseError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="document_parse_error")


class HierarchyConstructionError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="hierarchy_construction_error")


class InvalidVikingURI(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="invalid_viking_uri")


class DocumentAlreadyExists(ConflictError):
    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message, code="document_already_exists")


class NodeNotFound(NotFoundError):
    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message, code="node_not_found")


class DocumentNotFound(NotFoundError):
    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message, code="document_not_found")


class ProviderError(InfrastructureError):
    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message, code="provider_error")


class EmbeddingIdentityMismatch(ValidationDomainError):
    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message, code="embedding_identity_mismatch")


class IndexingError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="indexing_error")


class DocumentNotReadyError(ValidationDomainError):
    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message, code="document_not_ready")


class BudgetExhaustedError(DomainError):
    def __init__(self, resource: str) -> None:
        super().__init__(f"Retrieval budget exhausted: {resource}", code="budget_exhausted")
        self.resource = resource


class StaleSourceError(ConflictError):
    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message, code="stale_source")


class ScopeDeniedError(ValidationDomainError):
    def __init__(self, message: str) -> None:
        DomainError.__init__(self, message, code="scope_denied")


class AssessmentValidationError(DomainError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="assessment_validation_error")
