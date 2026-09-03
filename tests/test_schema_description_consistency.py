from __future__ import annotations

"""Regression test for the class of bug found 2026-08-29 (Part A schema audit, session
2): resolution_interval_conformal's field description read "k8s 76.6% [74.0%, 79.1%]"
(percent language) while its own nested model constrained those same values to a [0,1]
fraction (Field(ge=0.0, le=1.0)) -- a genuine units mismatch that only surfaced live
because Groq's strict structured-output mode turns a value the model can never actually
derive into a hard 400 (json_validate_failed) instead of a silent validation warning.

Walks every Pydantic model feeding TriagePlan's native structured-output schema and
checks each field's description text for numeric-range language against its own ge/le
constraints, so this class of contract violation cannot recur silently.
"""

import re

from triage_iq.models.triage import (
    AbstentionStatus,
    ConformalIntervalResult,
    DeclaredAttribution,
    GroundingAttribution,
    GroundingStatus,
    SimilarIssue,
    StageAbstention,
    TriagePlan,
)

_ALL_MODELS = [
    TriagePlan,
    SimilarIssue,
    ConformalIntervalResult,
    GroundingAttribution,
    GroundingStatus,
    DeclaredAttribution,
    StageAbstention,
    AbstentionStatus,
]

_PERCENT_PATTERN = re.compile(r"\d+(?:\.\d+)?\s*%")
_RANGE_PATTERN = re.compile(r"(-?\d+(?:\.\d+)?)\s*[-–]\s*(-?\d+(?:\.\d+)?)")


def _iter_fields():
    for model in _ALL_MODELS:
        for name, field in model.model_fields.items():
            yield model, name, field


def _ge_le(field) -> tuple[float | None, float | None]:
    ge = le = None
    for meta in field.metadata:
        if hasattr(meta, "ge") and not hasattr(meta, "le"):
            ge = meta.ge
        if hasattr(meta, "le") and not hasattr(meta, "ge"):
            le = meta.le
    return ge, le


def test_no_percent_language_on_a_fraction_constrained_field():
    """A description mentioning '%' on a field whose upper bound is <=1.0 is exactly the
    units-mismatch shape that broke resolution_interval_conformal: the description told
    the model "percent" (0-100 scale) while the schema enforced a 0-1 fraction."""
    violations = []
    for model, name, field in _iter_fields():
        description = field.description or ""
        if not _PERCENT_PATTERN.search(description):
            continue
        _, le = _ge_le(field)
        if le is not None and le <= 1.0:
            violations.append(
                f"{model.__name__}.{name}: description mentions '%' but le={le} "
                "(percent language on a fraction-constrained field)"
            )
    assert not violations, "\n".join(violations)


def test_described_numeric_range_matches_field_constraints():
    """A description containing an explicit 'X-Y' range must match the field's own
    ge/le when both are present -- a hand-authored range in prose is exactly what can
    silently drift from the Pydantic constraint sitting right next to it."""
    violations = []
    for model, name, field in _iter_fields():
        description = field.description or ""
        # Percent-scale mismatches are already caught by the dedicated test above;
        # skip here to avoid double-reporting the same root cause.
        if _PERCENT_PATTERN.search(description):
            continue
        matches = _RANGE_PATTERN.findall(description)
        if not matches:
            continue
        ge, le = _ge_le(field)
        if ge is None and le is None:
            continue
        for lo_s, hi_s in matches:
            lo, hi = float(lo_s), float(hi_s)
            if ge is not None and abs(lo - ge) > 1e-6:
                violations.append(
                    f"{model.__name__}.{name}: description range starts at {lo}, but ge={ge}"
                )
            if le is not None and abs(hi - le) > 1e-6:
                violations.append(
                    f"{model.__name__}.{name}: description range ends at {hi}, but le={le}"
                )
    assert not violations, "\n".join(violations)


def test_fields_overwritten_post_synthesis_instruct_the_model_not_to_guess():
    """Fields that TriageAssistant.triage_with_metadata / app.py always overwrite after
    generation must tell the model not to fabricate a value for them -- see the
    resolution_interval_conformal incident this test suite is named after. A field the
    model can never actually derive, wearing a description that reads like a real
    instruction to compute something, is exactly what produces a fabricated value under
    strict-mode's "every field is required" constraint.
    """
    # (model, field_name): substrings that must all appear in the field's description,
    # case-insensitive -- these fields are unconditionally (grounding/grounding_status/
    # resolution_bucket/resolution_confidence_pct) or conditionally
    # (resolution_interval_conformal, abstention_status) overwritten post-synthesis; see
    # triage.py's triage_with_metadata and app.py's post-processing block.
    must_instruct_no_fabrication = {
        (TriagePlan, "resolution_interval_conformal"): ["null", "cannot derive"],
        (TriagePlan, "resolution_bucket"): ["cannot derive", "overwritten"],
        (TriagePlan, "resolution_confidence_pct"): ["cannot derive", "overwritten"],
        (TriagePlan, "grounding"): ["cannot derive", "overwritten"],
        (TriagePlan, "grounding_status"): ["cannot derive", "overwritten"],
        (TriagePlan, "abstention_status"): ["cannot derive", "overwritten"],
    }
    violations = []
    for (model, name), required_substrings in must_instruct_no_fabrication.items():
        field = model.model_fields[name]
        description = (field.description or "").lower()
        missing = [s for s in required_substrings if s not in description]
        if missing:
            violations.append(
                f"{model.__name__}.{name}: description missing {missing} -- "
                f"current description: {field.description!r}"
            )
    assert not violations, "\n".join(violations)
