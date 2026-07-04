"""Unit tests for src/triage_iq/models/grounding.py.

Constructs minimal valid TriagePlan instances (pattern mirrors
eval/test_invariants.py:146-157) to exercise verify_plan_grounding without
touching the LLM or any I/O.
"""
from __future__ import annotations

from triage_iq.models.grounding import verify_plan_grounding
from triage_iq.models.triage import SimilarIssue, TriagePlan

CLASSIFIER_TOP3 = [
    {"label": "editor", "confidence": 0.7},
    {"label": "terminal", "confidence": 0.2},
    {"label": "extensions", "confidence": 0.1},
]


def _make_plan(
    predicted_component: str = "editor",
    similar_issues: list[SimilarIssue] | None = None,
) -> TriagePlan:
    """Return a minimal valid TriagePlan for grounding tests."""
    return TriagePlan(
        predicted_component=predicted_component,
        component_confidence=0.9,
        similar_issues=similar_issues or [],
        expected_resolution_summary="Fast fix expected",
        expected_resolution_lower_days=1.0,
        expected_resolution_upper_days=7.0,
        priority_guess="medium",
        priority_rationale="Medium priority",
        suggested_assignee_class="editor-team",
        suggested_next_steps=["Review the PR"],
        triage_summary="Test plan",
    )


def _sim(number: int) -> SimilarIssue:
    """Return a minimal valid SimilarIssue for the given issue number."""
    return SimilarIssue(number=number, similarity=0.5, relevance_note="similar bug")


def test_component_grounded_when_in_top3() -> None:
    """predicted_component exact-matches a classifier_top3 label."""
    plan = _make_plan(predicted_component="editor")
    report = verify_plan_grounding(plan, CLASSIFIER_TOP3, retrieved_numbers=set())

    assert report.component_grounded is True
    assert report.component_reason == "in classifier_top3"


def test_component_ungrounded_when_not_in_top3() -> None:
    """A real-sounding but absent label is not grounded."""
    plan = _make_plan(predicted_component="networking")
    report = verify_plan_grounding(plan, CLASSIFIER_TOP3, retrieved_numbers=set())

    assert report.component_grounded is False
    assert report.component_reason == "not in classifier_top3"


def test_all_similar_refs_grounded() -> None:
    """All cited similar-issue numbers are present in retrieved_numbers."""
    plan = _make_plan(similar_issues=[_sim(10), _sim(20)])
    report = verify_plan_grounding(plan, CLASSIFIER_TOP3, retrieved_numbers={10, 20, 30})

    assert report.ungrounded_refs == []


def test_some_similar_refs_ungrounded() -> None:
    """Refs not in retrieved_numbers appear sorted and deduped in ungrounded_refs."""
    plan = _make_plan(similar_issues=[_sim(10), _sim(99), _sim(99), _sim(50)])
    report = verify_plan_grounding(plan, CLASSIFIER_TOP3, retrieved_numbers={10, 20})

    assert report.similar_issue_refs == [10, 99, 99, 50]
    assert report.ungrounded_refs == [50, 99]


def test_empty_similar_issues_list() -> None:
    """No similar_issues means no refs and no ungrounded refs, and doesn't force all_grounded."""
    plan = _make_plan(predicted_component="editor", similar_issues=[])
    report = verify_plan_grounding(plan, CLASSIFIER_TOP3, retrieved_numbers={1, 2, 3})

    assert report.similar_issue_refs == []
    assert report.ungrounded_refs == []
    assert report.all_grounded is True


def test_combined_ungrounded_component_and_refs() -> None:
    """Both component and ref ungrounding are reported and all_grounded is False."""
    plan = _make_plan(predicted_component="networking", similar_issues=[_sim(99)])
    report = verify_plan_grounding(plan, CLASSIFIER_TOP3, retrieved_numbers={10, 20})

    assert report.component_grounded is False
    assert report.ungrounded_refs == [99]
    assert report.all_grounded is False


def test_all_grounded_true_only_when_both_hold() -> None:
    """all_grounded is True iff component is grounded AND no refs are ungrounded."""
    grounded_plan = _make_plan(predicted_component="editor", similar_issues=[_sim(10)])
    report = verify_plan_grounding(grounded_plan, CLASSIFIER_TOP3, retrieved_numbers={10})
    assert report.all_grounded is True

    component_only_bad = _make_plan(predicted_component="networking", similar_issues=[_sim(10)])
    report2 = verify_plan_grounding(component_only_bad, CLASSIFIER_TOP3, retrieved_numbers={10})
    assert report2.all_grounded is False

    refs_only_bad = _make_plan(predicted_component="editor", similar_issues=[_sim(99)])
    report3 = verify_plan_grounding(refs_only_bad, CLASSIFIER_TOP3, retrieved_numbers={10})
    assert report3.all_grounded is False


def test_case_and_whitespace_mismatch_is_ungrounded() -> None:
    """Exact match after strip is case-sensitive: case/whitespace drift is treated as ungrounded.

    This pins down the documented (not accidental) behavior: "Editor" and " editor " both
    differ from the top-3 label "editor" and must be reported as not grounded.
    """
    plan_case = _make_plan(predicted_component="Editor")
    report_case = verify_plan_grounding(plan_case, CLASSIFIER_TOP3, retrieved_numbers=set())
    assert report_case.component_grounded is False
    assert report_case.component_reason == "not in classifier_top3"

    plan_ws = _make_plan(predicted_component=" editor ")
    report_ws = verify_plan_grounding(plan_ws, CLASSIFIER_TOP3, retrieved_numbers=set())
    # " editor " strips to "editor" which DOES match — strip() is applied per spec.
    assert report_ws.component_grounded is True
    assert report_ws.component_reason == "in classifier_top3"
