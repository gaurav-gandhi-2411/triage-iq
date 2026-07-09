"""Unit tests for src/triage_iq/models/plan_verify.py.

Constructs minimal valid TriagePlan instances (pattern mirrors tests/test_grounding.py).
No LLM/cassette I/O.
"""
from __future__ import annotations

from triage_iq.models.plan_verify import verify_plan_consistency
from triage_iq.models.triage import DeclaredAttribution, TriagePlan


def _make_plan(
    priority_guess: str = "medium",
    resolution_bucket: str = "days",
    declared_attribution: DeclaredAttribution | None = None,
) -> TriagePlan:
    """Return a minimal valid TriagePlan for consistency tests."""
    plan = TriagePlan(
        predicted_component="editor",
        component_confidence=0.9,
        similar_issues=[],
        expected_resolution_summary="Fast fix expected",
        expected_resolution_lower_days=1.0,
        expected_resolution_upper_days=7.0,
        priority_guess=priority_guess,
        priority_rationale="Test rationale",
        suggested_assignee_class="editor-team",
        suggested_next_steps=["Review the PR"],
        triage_summary="Test plan",
        declared_attribution=declared_attribution,
    )
    plan.resolution_bucket = resolution_bucket
    return plan


def test_default_plan_is_consistent() -> None:
    """medium priority + days bucket, no declared_attribution -> fully consistent."""
    plan = _make_plan()
    report = verify_plan_consistency(plan)
    assert report.priority_resolution_consistent is True
    assert report.override_reason_consistent is True
    assert report.all_consistent is True


def test_high_priority_months_bucket_is_inconsistent() -> None:
    """high priority + months timeline contradicts itself."""
    plan = _make_plan(priority_guess="high", resolution_bucket="months")
    report = verify_plan_consistency(plan)
    assert report.priority_resolution_consistent is False
    assert report.all_consistent is False


def test_high_priority_long_bucket_is_inconsistent() -> None:
    """high priority + long timeline contradicts itself."""
    plan = _make_plan(priority_guess="high", resolution_bucket="long")
    report = verify_plan_consistency(plan)
    assert report.priority_resolution_consistent is False
    assert report.all_consistent is False


def test_high_priority_weeks_bucket_is_consistent() -> None:
    """high priority + weeks (not months/long) is consistent -- boundary case."""
    plan = _make_plan(priority_guess="high", resolution_bucket="weeks")
    report = verify_plan_consistency(plan)
    assert report.priority_resolution_consistent is True
    assert report.all_consistent is True


def test_medium_priority_months_bucket_is_consistent() -> None:
    """Rule is one-directional: medium/low priority at any timeline is never flagged."""
    plan = _make_plan(priority_guess="medium", resolution_bucket="months")
    report = verify_plan_consistency(plan)
    assert report.priority_resolution_consistent is True
    assert report.all_consistent is True


def test_low_priority_long_bucket_is_consistent() -> None:
    """Low priority with a long timeline is a perfectly normal plan, not a contradiction."""
    plan = _make_plan(priority_guess="low", resolution_bucket="long")
    report = verify_plan_consistency(plan)
    assert report.priority_resolution_consistent is True
    assert report.all_consistent is True


def test_model_override_with_reason_is_consistent() -> None:
    """component_source=model_override WITH a non-empty reason is consistent."""
    da = DeclaredAttribution(
        component_source="model_override",
        component_override_reason="Classifier top-3 clearly wrong for this issue.",
    )
    plan = _make_plan(declared_attribution=da)
    report = verify_plan_consistency(plan)
    assert report.override_reason_consistent is True
    assert report.all_consistent is True


def test_model_override_without_reason_is_inconsistent() -> None:
    """component_source=model_override with a BLANK reason violates the schema's own contract."""
    da = DeclaredAttribution(component_source="model_override", component_override_reason="")
    plan = _make_plan(declared_attribution=da)
    report = verify_plan_consistency(plan)
    assert report.override_reason_consistent is False
    assert report.all_consistent is False


def test_model_override_whitespace_only_reason_is_inconsistent() -> None:
    """A whitespace-only reason is treated the same as blank -- stripped before checking."""
    da = DeclaredAttribution(component_source="model_override", component_override_reason="   ")
    plan = _make_plan(declared_attribution=da)
    report = verify_plan_consistency(plan)
    assert report.override_reason_consistent is False


def test_classifier_top3_source_never_needs_a_reason() -> None:
    """component_source=classifier_top3 with an empty reason is fine -- reason is only
    required for model_override."""
    da = DeclaredAttribution(component_source="classifier_top3", component_override_reason="")
    plan = _make_plan(declared_attribution=da)
    report = verify_plan_consistency(plan)
    assert report.override_reason_consistent is True
    assert report.all_consistent is True


def test_declared_attribution_none_is_vacuously_consistent() -> None:
    """No declared_attribution block (flag off / model omitted it) -> vacuously consistent."""
    plan = _make_plan(declared_attribution=None)
    report = verify_plan_consistency(plan)
    assert report.override_reason_consistent is True


def test_both_rules_can_fail_independently() -> None:
    """A plan can violate both rules at once -- all_consistent reflects the conjunction."""
    da = DeclaredAttribution(component_source="model_override", component_override_reason="")
    plan = _make_plan(priority_guess="high", resolution_bucket="months", declared_attribution=da)
    report = verify_plan_consistency(plan)
    assert report.priority_resolution_consistent is False
    assert report.override_reason_consistent is False
    assert report.all_consistent is False
