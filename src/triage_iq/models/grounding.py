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


@dataclass(frozen=True)
class DeclaredAttributionReport:
    """Fidelity of the LLM's *declared* attribution vs the pipeline's actual upstream outputs.

    compliant: declared_attribution block present and well-formed.
    component_declaration: grounded_declared | honest_override | misattributed | missing.
        misattributed = declared component_source 'classifier_top3' but predicted_component
        is NOT in top-3 — a fabricated attribution, the worst class.
    cited_refs: distinct issue numbers cited across summary+next_steps, sorted.
    fabricated_citations: distinct cited numbers NOT in retrieved_numbers, sorted.
    grounded_citation_count / total_citation_count: over distinct cited_refs.
    blanket_citation: cited_refs covers every retrieved number (and retrieval was non-empty).
    """

    compliant: bool
    component_declaration: str
    cited_refs: list[int]
    fabricated_citations: list[int]
    grounded_citation_count: int
    total_citation_count: int
    blanket_citation: bool
    summary_cited: list[int]
    next_steps_cited: list[int]


def verify_declared_attribution(
    plan: Any, classifier_top3: list[dict], retrieved_numbers: set[int]
) -> DeclaredAttributionReport:
    """Check whether `plan.declared_attribution`'s claims match this pipeline's own outputs.

    Pure function, deterministic given a fixed plan — mirrors `verify_plan_grounding`'s style
    and strictness, but scores the LLM's *declared* attribution (elicited by the prompt) rather
    than reconstructing attribution post-hoc from the plan's other fields.

    Component-source declaration uses the same strict semantics as `verify_plan_grounding`:
    `plan.predicted_component` (after `.strip()`) must exact-match one of the `classifier_top3`
    labels for a "classifier_top3" declaration to be considered grounded.

    Args:
        plan: object with `.declared_attribution: DeclaredAttribution | None` and
            `.predicted_component: str` (e.g. a TriagePlan).
        classifier_top3: list of `{"label": str, "confidence": float}` dicts.
        retrieved_numbers: set of issue numbers actually retrieved for this request.

    Returns:
        DeclaredAttributionReport summarizing compliance, component-declaration class, and
        citation fidelity.
    """
    da = getattr(plan, "declared_attribution", None)
    if da is None:
        return DeclaredAttributionReport(
            compliant=False,
            component_declaration="missing",
            cited_refs=[],
            fabricated_citations=[],
            grounded_citation_count=0,
            total_citation_count=0,
            blanket_citation=False,
            summary_cited=[],
            next_steps_cited=[],
        )

    top3_labels = {str(entry["label"]) for entry in classifier_top3}
    predicted = str(plan.predicted_component).strip()
    in_top3 = predicted in top3_labels

    if da.component_source == "classifier_top3":
        component_declaration = "grounded_declared" if in_top3 else "misattributed"
    else:
        component_declaration = "honest_override"

    summary_cited = list(da.summary_cited_issues)
    next_steps_cited = list(da.next_steps_cited_issues)
    cited_refs = sorted(set(summary_cited) | set(next_steps_cited))
    fabricated_citations = sorted(n for n in cited_refs if n not in retrieved_numbers)
    grounded_citation_count = len(cited_refs) - len(fabricated_citations)
    total_citation_count = len(cited_refs)
    blanket_citation = bool(retrieved_numbers) and set(cited_refs) >= retrieved_numbers

    return DeclaredAttributionReport(
        compliant=True,
        component_declaration=component_declaration,
        cited_refs=cited_refs,
        fabricated_citations=fabricated_citations,
        grounded_citation_count=grounded_citation_count,
        total_citation_count=total_citation_count,
        blanket_citation=blanket_citation,
        summary_cited=summary_cited,
        next_steps_cited=next_steps_cited,
    )
