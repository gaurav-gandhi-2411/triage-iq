"""Deterministic verification that a TriagePlan is internally self-consistent.

Contrast with grounding.py: that module checks the plan's claims against this pipeline's own
upstream signals (classifier_top3, retrieval output) — "is this traceable to what we showed
the model." This module checks the plan against ITSELF — "does one field contradict another,"
independent of any external signal. Pure Python, no LLM, deterministic. FLAG-not-strip: never
raises, never mutates the plan.

ADR-0022: kept deliberately narrow (2 rules), using only structured (enum/boolean) fields —
never free text. The spec's illustrative examples ("a next-step references a component not in
the plan", "no next-step references a nonexistent field") were NOT implemented as rules: reliably
extracting a component reference from free-form next-step text without an LLM is a fragile
substring/NLP problem with real false-positive risk, which violates the "flag clear
contradictions, not stylistic judgments" bar this project holds every heuristic to. Documented
here rather than silently skipped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Buckets treated as "not urgent" for the priority/timeline contradiction rule. Must match
# BUCKET_LABELS in resolution.py (hours, days, weeks, months, long).
_SLOW_BUCKETS = frozenset({"months", "long"})


@dataclass(frozen=True)
class ConsistencyReport:
    """Result of checking a TriagePlan's internal self-consistency.

    priority_resolution_consistent: False iff priority_guess == "high" AND resolution_bucket
        is "months" or "long" — a plan claiming both maximum urgency and a multi-month timeline
        contradicts itself. One-directional by design: low/medium priority at any timeline, or
        high priority at hours/days/weeks, are all consistent — only the single clearest
        contradiction is flagged, not a stylistic preference about pairing.
    override_reason_consistent: False iff declared_attribution.component_source ==
        "model_override" but component_override_reason is blank — the schema documents this
        field as required in that case (ADR-0020). Vacuously True when declared_attribution is
        absent (the field is optional/flag-gated — see ADR-0020).
    all_consistent: conjunction of the above.
    """

    priority_resolution_consistent: bool
    override_reason_consistent: bool
    all_consistent: bool


def verify_plan_consistency(plan: Any) -> ConsistencyReport:
    """Check `plan` for internal contradictions, using only structured fields.

    Args:
        plan: object with `.priority_guess: str`, `.resolution_bucket: str`, and optionally
            `.declared_attribution: DeclaredAttribution | None` (e.g. a TriagePlan).

    Returns:
        ConsistencyReport. Never raises on well-formed input; never mutates `plan`.
    """
    priority_resolution_consistent = not (
        str(plan.priority_guess) == "high" and str(plan.resolution_bucket) in _SLOW_BUCKETS
    )

    declared_attribution = getattr(plan, "declared_attribution", None)
    if declared_attribution is None:
        override_reason_consistent = True
    else:
        override_reason_consistent = not (
            declared_attribution.component_source == "model_override"
            and not str(declared_attribution.component_override_reason).strip()
        )

    return ConsistencyReport(
        priority_resolution_consistent=priority_resolution_consistent,
        override_reason_consistent=override_reason_consistent,
        all_consistent=priority_resolution_consistent and override_reason_consistent,
    )
