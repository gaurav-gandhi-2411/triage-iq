from __future__ import annotations

"""Regression test for ADR-0054/0055's schema-reduction fix (2026-09-03), updated for
ADR-0057 Phase 3 (2026-09-04)'s declared_attribution restoration.

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

ADR-0057 Phase 3: declared_attribution is carved back OUT of the strip
(_NEVER_STRIP_DESPITE_DEFAULT) -- unlike the other 6, it is real LLM-elicited signal
(ADR-0020), not a fixed/derived value. It is restored to the wire schema's `properties`
as a nullable object (`type: [object, "null"]`, via _inline_nullable_object_refs) --
"optional" in the only sense Groq's strict:true mode allows: present, but satisfiable
with `null`, not literally absent from `required` (strict mode forces required to equal
every remaining property, with no exception).
"""

from triage_iq.models.triage import (
    TriagePlan,
    _build_triage_plan_response_format,
)
from pydantic_core import PydanticUndefined

# The 6 fields still stripped -- declared_attribution moved to
# _EXPECTED_RESTORED_FIELDS below (ADR-0057 Phase 3).
_EXPECTED_STRIPPED_FIELDS = frozenset({
    "resolution_bucket",
    "resolution_confidence_pct",
    "resolution_interval_conformal",
    "grounding",
    "grounding_status",
    "abstention_status",
})

# Carved out of the generic default-based strip (ADR-0057 Phase 3) -- present in the
# wire schema despite having an explicit Pydantic default, because it's real
# LLM-elicited signal, not a fixed/derived value the app overwrites post-hoc.
_EXPECTED_RESTORED_FIELDS = frozenset({"declared_attribution"})


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
    """Root-cause check (not just the 6 named fields): ANY TriagePlan field with an
    explicit Pydantic `default` (not `default_factory`) must be absent from the wire
    schema, EXCEPT the explicit _NEVER_STRIP_DESPITE_DEFAULT carve-out
    (declared_attribution, ADR-0057 Phase 3) -- this is the general mechanism
    _strip_post_hoc_fields implements, so a newly-added defaulted field is excluded
    automatically without needing this test updated. default_factory fields (e.g.
    similar_issues) are a different semantic ("try, empty is an acceptable fallback")
    and are deliberately NOT excluded."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    properties = set(schema["properties"].keys())
    for name, field_info in TriagePlan.model_fields.items():
        has_explicit_default = field_info.default is not PydanticUndefined
        if has_explicit_default and name not in _EXPECTED_RESTORED_FIELDS:
            assert name not in properties, (
                f"{name!r} has an explicit Pydantic default but is still present in the "
                "wire schema -- _strip_post_hoc_fields should have excluded it."
            )


def test_declared_attribution_restored_as_nullable_object():
    """ADR-0057 Phase 3: declared_attribution is present in the wire schema (unlike the
    other 6 default-bearing fields), nullable rather than a bare object -- the model can
    satisfy Groq's strict `required` by emitting null, exactly like the pre-ADR-0055
    treatment of grounding/grounding_status/abstention_status."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    assert "declared_attribution" in schema["properties"]
    assert "declared_attribution" in schema["required"]
    prop = schema["properties"]["declared_attribution"]
    assert set(prop["type"]) == {"object", "null"}
    assert {"component_source", "component_override_reason",
            "summary_cited_issues", "next_steps_cited_issues"} == set(prop["properties"].keys())


def test_similar_issues_still_required():
    """similar_issues (default_factory=list) carries real model-derived signal and was
    present in all 9 examined early-termination failures -- confirm it's untouched by
    this fix, not accidentally swept up by a broader exclusion."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    assert "similar_issues" in schema["properties"]
    assert "similar_issues" in schema["required"]


def test_required_field_count_is_twelve():
    """18 -> 11 (ADR-0055) -> 12 (ADR-0057 Phase 3 restores declared_attribution). A
    change to this number should be a deliberate schema decision, not a silent side
    effect -- if this test needs updating, check ADR-0054/0055/0057 first."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    assert len(schema["required"]) == 12


def test_orphaned_defs_are_pruned():
    """Once grounding/grounding_status/abstention_status/resolution_interval_conformal
    are stripped, their nested $defs (ConformalIntervalResult, GroundingAttribution,
    GroundingStatus, AbstentionStatus, StageAbstention) become unreachable and must not
    still be paid for in the wire schema (rule 15b/quota accounting). DeclaredAttribution
    is restored (ADR-0057 Phase 3) but its $def is STILL unreachable: it has no nested
    BaseModel refs of its own, so _inline_nullable_object_refs copies its schema
    directly into the `declared_attribution` property rather than leaving a $ref behind
    -- confirmed empirically, not assumed (test_declared_attribution_restored_as_nullable_object
    checks the inlined content directly)."""
    schema = _build_triage_plan_response_format()["json_schema"]["schema"]
    defs = schema.get("$defs", {})
    assert set(defs.keys()) == {"SimilarIssue"}
