"""Learning policy helpers — enqueue and worker gates."""

from __future__ import annotations

from vikingrag.domain.models.experience import EdgeBuildStatus, LearningPolicy, QueryRun


def should_persist_run(policy: LearningPolicy) -> bool:
    """Whether a query_run row should be written at all."""
    return policy is not LearningPolicy.OFF


def should_enqueue_edge_build(policy: LearningPolicy) -> bool:
    """Whether edge_build_status may become PENDING."""
    return policy is LearningPolicy.LEARN


def apply_learning_policy(run: QueryRun, policy: LearningPolicy) -> QueryRun:
    """Mutate run edge-build fields according to policy. Returns the same run."""
    run.learning_policy = policy
    if policy is LearningPolicy.OFF:
        run.edge_build_status = EdgeBuildStatus.NONE
        run.edge_build_error = "learning_policy_off"
    elif policy is LearningPolicy.RECORD_ONLY:
        run.edge_build_status = EdgeBuildStatus.SKIPPED
        run.edge_build_error = "learning_policy_record_only"
    elif policy is LearningPolicy.FROZEN:
        run.edge_build_status = EdgeBuildStatus.SKIPPED
        run.edge_build_error = "learning_policy_frozen"
    # LEARN: caller uses mark_run_for_edge_build when embeddings present
    return run


def worker_may_build(run: QueryRun) -> bool:
    """Worker refuses jobs that are not LEARN / learnable."""
    if run.learning_policy is not LearningPolicy.LEARN:
        return False
    return run.is_learnable
