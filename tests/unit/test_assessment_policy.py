"""Evidence assessment validation and status policy."""

from __future__ import annotations

from uuid import uuid4

import pytest

from vikingrag.application.assessment import (
    EmptyBundleAssessor,
    ScriptedEvidenceAssessor,
    compute_coverage,
    deterministic_aspects_from_query,
    finalize_status,
)
from vikingrag.domain.errors import AssessmentValidationError
from vikingrag.domain.models.assessment import (
    AspectSupport,
    AspectSupportStatus,
    AssessmentStatus,
    EvidenceReference,
)
from vikingrag.domain.models.document import DocumentId, NodeId
from vikingrag.domain.models.evidence import EvidenceBundle, RetrievedEvidence


@pytest.mark.asyncio
async def test_empty_bundle_is_insufficient() -> None:
    result = await EmptyBundleAssessor().assess("How do we verify?", EvidenceBundle.from_items([]))
    assert result.status is AssessmentStatus.INSUFFICIENT
    assert result.coverage == 0.0


@pytest.mark.asyncio
async def test_scripted_sufficient_with_valid_quote() -> None:
    item = RetrievedEvidence(
        evidence_id="e1",
        uri="viking://x",
        text="Evidence Verification maps claims to retrieved evidence.",
        token_count=8,
        document_id=DocumentId(uuid4()),
        node_id=NodeId(uuid4()),
        content_hash="h",
        end_offset=50,
    )
    bundle = EvidenceBundle.from_items([item])
    assessor = ScriptedEvidenceAssessor()
    result = await assessor.assess("How do we verify retrieved evidence?", bundle)
    assert result.status is AssessmentStatus.SUFFICIENT
    assert result.aspect_support[0].references[0].evidence_id == "e1"


@pytest.mark.asyncio
async def test_fabricated_evidence_id_rejected() -> None:
    item = RetrievedEvidence(
        evidence_id="e1",
        uri="viking://x",
        text="hello world",
        token_count=2,
        end_offset=11,
    )
    bundle = EvidenceBundle.from_items([item])
    bad = (
        AspectSupport(
            aspect_id="a1",
            status=AspectSupportStatus.SUPPORTED,
            references=(EvidenceReference(evidence_id="e999", quote="hello"),),
        ),
    )
    assessor = ScriptedEvidenceAssessor(aspect_support=bad)
    with pytest.raises(AssessmentValidationError, match="Fabricated"):
        await assessor.assess("hello?", bundle)


@pytest.mark.asyncio
async def test_fabricated_quote_rejected() -> None:
    item = RetrievedEvidence(
        evidence_id="e1",
        uri="viking://x",
        text="hello world",
        token_count=2,
        end_offset=11,
    )
    bundle = EvidenceBundle.from_items([item])
    bad = (
        AspectSupport(
            aspect_id="a1",
            status=AspectSupportStatus.SUPPORTED,
            references=(EvidenceReference(evidence_id="e1", quote="not in source"),),
        ),
    )
    assessor = ScriptedEvidenceAssessor(aspect_support=bad)
    with pytest.raises(AssessmentValidationError, match="Quote"):
        await assessor.assess("hello?", bundle)


def test_finalize_blocks_on_conflict() -> None:
    support = (
        AspectSupport(
            aspect_id="a1",
            status=AspectSupportStatus.SUPPORTED,
            references=(EvidenceReference(evidence_id="e1"),),
        ),
    )
    status = finalize_status(
        aspect_support=support,
        conflicts=("contradiction",),
        coverage=1.0,
        assessment_ok=True,
    )
    assert status is AssessmentStatus.INSUFFICIENT


def test_coverage_partial() -> None:
    support = (
        AspectSupport(aspect_id="a1", status=AspectSupportStatus.SUPPORTED),
        AspectSupport(aspect_id="a2", status=AspectSupportStatus.PARTIAL),
    )
    assert compute_coverage(support) == 0.75


def test_aspect_split_from_query() -> None:
    aspects = deterministic_aspects_from_query("What is storage? How does retrieval work?")
    assert len(aspects) >= 2
