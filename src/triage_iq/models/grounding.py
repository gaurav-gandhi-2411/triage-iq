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


# ---------------------------------------------------------------------------
# Validated override reason (2026-08-28, Part C/E of the diagnostic session)
# ---------------------------------------------------------------------------
#
# The version of this check that shipped on PR #106's branch (never merged) accepted
# ANY non-empty component_override_reason string as sufficient to mark a
# classifier-top3-deviating component as "grounded" -- self-certification, checked
# against nothing. Confirmed directly: all 3 of the openai/gpt-oss-20b cassette's
# "ungrounded" issues were exactly the 3 issues where the model used this declaration,
# and applying the unsound check would have handed the model a pass on precisely the
# disputed cases. This replacement requires the reason to be checkably tied to real
# signal -- a cited retrieved issue number, or a verbatim entity from the issue's own
# title/body -- not merely present. Still not immune to a model fabricating a
# plausible, evidence-citing lie, but a real improvement over "any string passes."

_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "this", "that", "these", "those",
    "and", "or", "but", "not", "with", "from", "into", "onto", "for", "to", "of", "in",
    "on", "at", "by", "as", "it", "its", "be", "been", "being", "has", "have", "had",
    "does", "did", "doing", "when", "where", "which", "who", "what", "why", "how",
})


def _significant_words(text: str, min_length: int = 4) -> set[str]:
    """Lowercased words of at least `min_length` chars, minus common stopwords.

    Deliberately simple (no NLP dependency) -- this only needs to catch a reason that
    verbatim echoes something specific from the issue, not paraphrase detection.
    """
    import re

    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_./-]*", text.lower())
    return {w for w in words if len(w) >= min_length and w not in _STOPWORDS}


@dataclass(frozen=True)
class ResolvedGroundingStatus:
    """The single, final grounding verdict for a plan -- component_grounded here already
    reflects the (optional, disabled-by-default) validated-override rescue, so any caller
    reading `all_grounded` gets the actual answer, not a partial one they still need to
    combine with a separate override check themselves.

    This is what both triage_with_metadata() (production) and measure_grounding.py (eval)
    must call -- not two independent implementations of "compute grounding," which is
    exactly the class of bug this replaces (measure_grounding.py, unmerged PR #106 found,
    silently never exercised the override-rescue branch triage_with_metadata() had grown).
    """

    component_grounded: bool
    component_reason: str
    similar_issue_refs: list[int]
    ungrounded_refs: list[int]
    all_grounded: bool
    override_applied: bool
    override_reason_validated: bool | None  # None when no override was declared at all


def compute_grounding_status(
    plan: Any,
    classifier_top3: list[dict],
    retrieved_numbers: set[int],
    *,
    enable_validated_override_rescue: bool = False,
    issue_title: str = "",
    issue_body: str = "",
) -> ResolvedGroundingStatus:
    """Single source of truth for "is this plan grounded" -- shared by production and eval.

    `enable_validated_override_rescue` defaults to False: a component that deviates from
    classifier_top3 is flagged ungrounded unless the caller explicitly opts into rescuing
    an honestly-declared, evidence-tied override via `verify_override_reason_grounded`.
    Disabled by default so nothing self-certifies until a caller makes a deliberate choice
    to enable it -- see grounding.py's validated-override-reason section for why the prior
    (self-certifying) version was unsound and is not present here at all.
    """
    report = verify_plan_grounding(plan, classifier_top3, retrieved_numbers)
    component_grounded = report.component_grounded
    component_reason = report.component_reason
    override_applied = False
    override_reason_validated: bool | None = None

    declared = getattr(plan, "declared_attribution", None)
    if not component_grounded and enable_validated_override_rescue and declared is not None:
        da_report = verify_declared_attribution(plan, classifier_top3, retrieved_numbers)
        if da_report.component_declaration == "honest_override":
            override_reason_validated = verify_override_reason_grounded(
                declared.component_override_reason, issue_title, issue_body, retrieved_numbers,
            )
            if override_reason_validated:
                component_grounded = True
                component_reason = "model_override (validated)"
                override_applied = True

    return ResolvedGroundingStatus(
        component_grounded=component_grounded,
        component_reason=component_reason,
        similar_issue_refs=report.similar_issue_refs,
        ungrounded_refs=report.ungrounded_refs,
        all_grounded=component_grounded and len(report.ungrounded_refs) == 0,
        override_applied=override_applied,
        override_reason_validated=override_reason_validated,
    )


def verify_override_reason_grounded(
    reason: str,
    issue_title: str,
    issue_body: str,
    retrieved_numbers: set[int],
) -> bool:
    """True if a declared component-override reason is checkably tied to real signal.

    Two independent sufficient conditions (either passes):
    1. The reason text names a retrieved issue number (e.g. "#12345" or "12345") that
       is actually in `retrieved_numbers` for this request.
    2. The reason text contains at least one "significant" word (>=4 chars, not a
       stopword) that also appears verbatim in the issue's own title or body.

    Does not (cannot) catch a model fabricating a plausible-sounding reason that also
    happens to reuse issue vocabulary -- it raises the bar from "any non-empty string"
    to "checkably tied to something real," not to "provably true."
    """
    import re

    cited_numbers = {int(n) for n in re.findall(r"\d+", reason)}
    if cited_numbers & retrieved_numbers:
        return True

    issue_words = _significant_words(issue_title) | _significant_words(issue_body)
    reason_words = _significant_words(reason)
    return bool(issue_words & reason_words)
