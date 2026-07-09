"""Unit tests for src/triage_iq/models/abstention.py.

Constructs minimal valid TriagePlan instances (pattern mirrors tests/test_grounding.py).
No LLM/cassette I/O.
"""
from __future__ import annotations

from triage_iq.models.abstention import (
    COMPONENT_CONFIDENCE_THRESHOLD,
    RESOLUTION_WIDTH_THRESHOLD_DAYS,
    compute_abstention_status,
)
from triage_iq.models.triage import TriagePlan

K8S = "kubernetes/kubernetes"


def _make_plan(component_confidence: float = 0.9) -> TriagePlan:
    """Return a minimal valid TriagePlan for abstention tests."""
    return TriagePlan(
        predicted_component="storage",
        component_confidence=component_confidence,
        similar_issues=[],
        expected_resolution_summary="Fast fix expected",
        expected_resolution_lower_days=1.0,
        expected_resolution_upper_days=7.0,
        priority_guess="medium",
        priority_rationale="Medium priority",
        suggested_assignee_class="storage-team",
        suggested_next_steps=["Review the PR"],
        triage_summary="Test plan",
    )


def test_high_confidence_grounded_narrow_interval_not_abstained() -> None:
    """Confidence above threshold, grounded, narrow interval -> neither stage abstains."""
    threshold = COMPONENT_CONFIDENCE_THRESHOLD[K8S]
    plan = _make_plan(component_confidence=threshold + 0.1)
    status = compute_abstention_status(
        plan, K8S, component_grounded=True, conformal_width_days=1.0
    )
    assert status.component.abstained is False
    assert status.component.reason == ""
    assert status.resolution.abstained is False
    assert status.resolution.reason == ""


def test_low_confidence_abstains_component_only() -> None:
    """Confidence below threshold -> component abstains with low_confidence; resolution unaffected."""
    threshold = COMPONENT_CONFIDENCE_THRESHOLD[K8S]
    plan = _make_plan(component_confidence=threshold - 0.05)
    status = compute_abstention_status(
        plan, K8S, component_grounded=True, conformal_width_days=1.0
    )
    assert status.component.abstained is True
    assert status.component.reason == "low_confidence"
    assert status.resolution.abstained is False


def test_ungrounded_abstains_regardless_of_confidence() -> None:
    """Ungrounded is a hard trigger -- abstains even at maximum confidence."""
    plan = _make_plan(component_confidence=1.0)
    status = compute_abstention_status(
        plan, K8S, component_grounded=False, conformal_width_days=1.0
    )
    assert status.component.abstained is True
    assert status.component.reason == "ungrounded"


def test_wide_interval_abstains_resolution_only() -> None:
    """Interval width above threshold -> resolution abstains with wide_interval; component unaffected."""
    threshold = RESOLUTION_WIDTH_THRESHOLD_DAYS[K8S]
    plan = _make_plan(component_confidence=1.0)
    status = compute_abstention_status(
        plan, K8S, component_grounded=True, conformal_width_days=threshold + 1.0
    )
    assert status.resolution.abstained is True
    assert status.resolution.reason == "wide_interval"
    assert status.component.abstained is False


def test_width_exactly_at_threshold_not_abstained() -> None:
    """Boundary: width == threshold does not abstain (strictly greater-than triggers it)."""
    threshold = RESOLUTION_WIDTH_THRESHOLD_DAYS[K8S]
    plan = _make_plan(component_confidence=1.0)
    status = compute_abstention_status(
        plan, K8S, component_grounded=True, conformal_width_days=threshold
    )
    assert status.resolution.abstained is False


def test_none_width_never_abstains_resolution() -> None:
    """conformal_width_days=None (no conformal adjustment for this repo) -- fails open."""
    plan = _make_plan(component_confidence=1.0)
    status = compute_abstention_status(
        plan, K8S, component_grounded=True, conformal_width_days=None
    )
    assert status.resolution.abstained is False
    assert status.resolution.reason == ""


def test_unknown_repo_never_abstains_component() -> None:
    """A repo with no configured threshold fails open, same policy as conformal adjustments."""
    plan = _make_plan(component_confidence=0.01)
    status = compute_abstention_status(
        plan, "unknown/repo", component_grounded=True, conformal_width_days=1.0
    )
    assert status.component.abstained is False
    assert status.component.reason == ""
