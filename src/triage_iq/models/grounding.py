"""Deterministic verification that a TriagePlan is traceable to this pipeline's own
classifier_top3 / retrieval outputs for this request — not world-truth verification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GroundingReport:
    """Result of checking a TriagePlan's claims against upstream signals.

    component_grounded: whether predicted_component exact-matches a classifier_top3 label.
    component_reason: "in classifier_top3" or "not in classifier_top3".
    similar_issue_refs: all issue numbers the plan's similar_issues cited (raw, not deduped).
    ungrounded_refs: distinct similar_issue_refs not present in retrieved_numbers, sorted.
    all_grounded: component_grounded and no ungrounded_refs.
    """

    component_grounded: bool
    component_reason: str
    similar_issue_refs: list[int]
    ungrounded_refs: list[int]
    all_grounded: bool


def verify_plan_grounding(
    plan: Any, classifier_top3: list[dict], retrieved_numbers: set[int]
) -> GroundingReport:
    """Check whether `plan`'s component and similar-issue citations are grounded.

    Component grounding is strict: `plan.predicted_component` (after `.strip()`) must
    exact-match (case-sensitive) one of the `label` values in `classifier_top3` — not
    label-space membership, just top-3.

    Similar-issue grounding: each `s.number` in `plan.similar_issues` is checked against
    `retrieved_numbers` (the pipeline's own retrieval output for this request). Distinct
    ungrounded numbers are returned sorted for deterministic, readable output.

    Args:
        plan: object with `.predicted_component: str` and `.similar_issues: list[Any]`
            where each item has a `.number: int` attribute (e.g. a TriagePlan).
        classifier_top3: list of `{"label": str, "confidence": float}` dicts.
        retrieved_numbers: set of issue numbers actually retrieved for this request.

    Returns:
        GroundingReport summarizing component and similar-issue grounding.
    """
    top3_labels = {str(entry["label"]) for entry in classifier_top3}
    predicted = str(plan.predicted_component).strip()
    component_grounded = predicted in top3_labels
    component_reason = "in classifier_top3" if component_grounded else "not in classifier_top3"

    similar_issue_refs: list[int] = [s.number for s in plan.similar_issues]
    ungrounded_refs = sorted({n for n in similar_issue_refs if n not in retrieved_numbers})

    all_grounded = component_grounded and len(ungrounded_refs) == 0

    return GroundingReport(
        component_grounded=component_grounded,
        component_reason=component_reason,
        similar_issue_refs=similar_issue_refs,
        ungrounded_refs=ungrounded_refs,
        all_grounded=all_grounded,
    )
