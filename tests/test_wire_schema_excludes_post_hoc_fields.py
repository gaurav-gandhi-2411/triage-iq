from __future__ import annotations

"""Regression test for ADR-0054/0055's schema-reduction fix (2026-09-03).

All 9 early-termination failures examined across the openai/gpt-oss-120b re-record and
the model bake-off screen were missing a subset of exactly 7 TriagePlan fields:
resolution_bucket, resolution_confidence_pct, resolution_interval_conformal, grounding,
grounding_status, declared_attribution, abstention_status. Every one of these is either
overwritten post-hoc by TriageAssistant.triage_with_metadata / app.py regardless of what
the model emits, or already null-tolerant by design (declared_attribution). Forcing them
into Groq's strict `required` list asked the model to spend generation budget on values
nothing downstream consumes, and the model reliably dropped a subset of them under
constrained decoding, producing a 400 (json_validate_failed).

Fix: _strip_post_hoc_fields (src/triage_iq/models/triage.py) removes any TriagePlan
field with an explicit Pydantic `default` (not `default_factory`) from the wire schema
before it's sent to Groq. This test pins that behavior at the schema-content level so a
future field addition can't silently regress it -- the failure mode this guards against
is a NEW field being added to TriagePlan with a `default=...` (i.e. genuinely optional,
overwritten-post-hoc, or null-tolerant) that nonetheless ends up forced into `required`
again because the exclusion logic didn't fire, or a currently-excluded field being
removed from the excluded set by accident.
"""

from triage_iq.models.triage import (
    TriagePlan,
    _build_triage_plan_response_format,
)
from pydantic_core import PydanticUndefined

# The exact 7 fields this fix targets -- see the module docstring and
# _strip_post_hoc_fields's own docstring for the full mechanism/evidence.
_EXPECTED_STRIPPED_FIELDS = frozenset({
    "resolution_bucket",
    "resolution_confidence_pct",
    "resolution_interval_conformal",
    "grounding",
    "grounding_status",
    "declared_attribution",
    "abstention_status",
})


def test_post_hoc_fields_absent_from_wire_schema():
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    properties = set(schema["properties"].keys())
    overlap = properties & _EXPECTED_STRIPPED_FIELDS
    assert not overlap, (
        f"Field(s) {overlap} still present in the wire schema sent to Groq -- these are "
        "overwritten post-hoc or null-tolerant by design and must not be forced into "
        "required. See ADR-0054/0055 and _strip_post_hoc_fields's docstring."
    )


def test_wire_schema_required_equals_its_own_properties():
    """Groq's strict:true mode requires `required` to list exactly every remaining
    property (no notion of an optional property) -- confirmed by trial, documented in
    _force_strict_schema_requirements. A schema violating this gets rejected by Groq
    outright, not silently degraded, so this must hold exactly."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    assert set(schema["required"]) == set(schema["properties"].keys())


def test_every_default_bearing_triage_plan_field_is_excluded():
    """Root-cause check (not just the 7 named fields): ANY TriagePlan field with an
    explicit Pydantic `default` (not `default_factory`) must be absent from the wire
    schema -- this is the general mechanism _strip_post_hoc_fields implements, so a
    newly-added defaulted field is excluded automatically without needing this test
    updated. default_factory fields (e.g. similar_issues) are a different semantic
    ("try, empty is an acceptable fallback") and are deliberately NOT excluded."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    properties = set(schema["properties"].keys())
    for name, field_info in TriagePlan.model_fields.items():
        has_explicit_default = field_info.default is not PydanticUndefined
        if has_explicit_default:
            assert name not in properties, (
                f"{name!r} has an explicit Pydantic default but is still present in the "
                "wire schema -- _strip_post_hoc_fields should have excluded it."
            )


def test_similar_issues_still_required():
    """similar_issues (default_factory=list) carries real model-derived signal and was
    present in all 9 examined early-termination failures -- confirm it's untouched by
    this fix, not accidentally swept up by a broader exclusion."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    assert "similar_issues" in schema["properties"]
    assert "similar_issues" in schema["required"]


def test_required_field_count_is_eleven():
    """18 -> 11 required fields (7 stripped). A change to this number should be a
    deliberate schema decision, not a silent side effect -- if this test needs updating,
    check ADR-0054/0055 first."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    assert len(schema["required"]) == 11


def test_orphaned_defs_are_pruned():
    """Once grounding/grounding_status/declared_attribution/abstention_status/
    resolution_interval_conformal are stripped, their nested $defs
    (ConformalIntervalResult, GroundingAttribution, GroundingStatus, DeclaredAttribution,
    AbstentionStatus, StageAbstention) become unreachable and must not still be paid for
    in the wire schema (rule 15b/quota accounting)."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    defs = schema.get("$defs", {})
    assert set(defs.keys()) == {"SimilarIssue"}
