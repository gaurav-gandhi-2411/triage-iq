"""Unit tests for W6 attribution-fidelity: DeclaredAttribution parsing, verify_declared_attribution,
and the SYSTEM_PROMPT / few-shot additions.

No LLM/cassette I/O — constructs minimal valid TriagePlan instances (pattern mirrors
tests/test_grounding.py).
"""
from __future__ import annotations

import json

from triage_iq.models.grounding import verify_declared_attribution
from triage_iq.models.triage import DeclaredAttribution, TriagePlan

CLASSIFIER_TOP3 = [
    {"label": "editor", "confidence": 0.7},
    {"label": "terminal", "confidence": 0.2},
    {"label": "extensions", "confidence": 0.1},
]


def _make_plan(
    predicted_component: str = "editor",
    declared_attribution: dict | DeclaredAttribution | None = None,
) -> TriagePlan:
    """Return a minimal valid TriagePlan for attribution tests."""
    return TriagePlan(
        predicted_component=predicted_component,
        component_confidence=0.9,
        similar_issues=[],
        expected_resolution_summary="Fast fix expected",
        expected_resolution_lower_days=1.0,
        expected_resolution_upper_days=7.0,
        priority_guess="medium",
        priority_rationale="Medium priority",
        suggested_assignee_class="editor-team",
        suggested_next_steps=["Review the PR"],
        triage_summary="Test plan",
        declared_attribution=declared_attribution,
    )


# ---------------------------------------------------------------------------
# verify_declared_attribution
# ---------------------------------------------------------------------------


def test_grounded_declared_when_component_source_and_top3_agree() -> None:
    """component_source=classifier_top3 and predicted_component IS in top-3 -> grounded_declared."""
    plan = _make_plan(
        predicted_component="editor",
        declared_attribution={
            "component_source": "classifier_top3",
            "summary_cited_issues": [],
            "next_steps_cited_issues": [],
        },
    )
    report = verify_declared_attribution(plan, CLASSIFIER_TOP3, retrieved_numbers=set())

    assert report.compliant is True
    assert report.component_declaration == "grounded_declared"


def test_honest_override_when_component_source_is_model_override() -> None:
    """component_source=model_override is always honest_override regardless of top-3 membership."""
    plan = _make_plan(
        predicted_component="networking",
        declared_attribution={
            "component_source": "model_override",
            "component_override_reason": "Title strongly implies networking despite low TF-IDF signal.",
            "summary_cited_issues": [],
            "next_steps_cited_issues": [],
        },
    )
    report = verify_declared_attribution(plan, CLASSIFIER_TOP3, retrieved_numbers=set())

    assert report.compliant is True
    assert report.component_declaration == "honest_override"


def test_misattributed_when_declared_top3_but_not_actually_in_top3() -> None:
    """component_source=classifier_top3 but predicted_component is NOT in top-3 -> misattributed
    (a fabricated attribution — the worst class)."""
    plan = _make_plan(
        predicted_component="networking",
        declared_attribution={
            "component_source": "classifier_top3",
            "summary_cited_issues": [],
            "next_steps_cited_issues": [],
        },
    )
    report = verify_declared_attribution(plan, CLASSIFIER_TOP3, retrieved_numbers=set())

    assert report.compliant is True
    assert report.component_declaration == "misattributed"


def test_missing_when_declared_attribution_is_none() -> None:
    """No declared_attribution block -> compliant=False, component_declaration='missing'."""
    plan = _make_plan(predicted_component="editor", declared_attribution=None)
    report = verify_declared_attribution(plan, CLASSIFIER_TOP3, retrieved_numbers={1, 2})

    assert report.compliant is False
    assert report.component_declaration == "missing"
    assert report.cited_refs == []
    assert report.fabricated_citations == []
    assert report.grounded_citation_count == 0
    assert report.total_citation_count == 0
    assert report.blanket_citation is False


def test_fabricated_citations_detected_and_sorted() -> None:
    """Citations not present in retrieved_numbers are reported as fabricated, sorted and deduped."""
    plan = _make_plan(
        predicted_component="editor",
        declared_attribution={
            "component_source": "classifier_top3",
            "summary_cited_issues": [99, 10],
            "next_steps_cited_issues": [50, 99],
        },
    )
    report = verify_declared_attribution(plan, CLASSIFIER_TOP3, retrieved_numbers={10, 20})

    assert report.cited_refs == [10, 50, 99]
    assert report.fabricated_citations == [50, 99]


def test_grounded_citation_count_math() -> None:
    """grounded_citation_count is total distinct citations minus fabricated ones."""
    plan = _make_plan(
        predicted_component="editor",
        declared_attribution={
            "component_source": "classifier_top3",
            "summary_cited_issues": [10, 20],
            "next_steps_cited_issues": [30],
        },
    )
    report = verify_declared_attribution(plan, CLASSIFIER_TOP3, retrieved_numbers={10, 20})

    assert report.total_citation_count == 3
    assert report.fabricated_citations == [30]
    assert report.grounded_citation_count == 2


def test_blanket_citation_true_when_citing_every_retrieved_number() -> None:
    """blanket_citation is True only when cited_refs covers every retrieved number."""
    plan = _make_plan(
        predicted_component="editor",
        declared_attribution={
            "component_source": "classifier_top3",
            "summary_cited_issues": [10, 20],
            "next_steps_cited_issues": [],
        },
    )
    report = verify_declared_attribution(plan, CLASSIFIER_TOP3, retrieved_numbers={10, 20})

    assert report.blanket_citation is True


def test_blanket_citation_false_when_citing_a_subset() -> None:
    """Citing only a subset of retrieved numbers is not a blanket citation."""
    plan = _make_plan(
        predicted_component="editor",
        declared_attribution={
            "component_source": "classifier_top3",
            "summary_cited_issues": [10],
            "next_steps_cited_issues": [],
        },
    )
    report = verify_declared_attribution(plan, CLASSIFIER_TOP3, retrieved_numbers={10, 20})

    assert report.blanket_citation is False


def test_blanket_citation_false_when_retrieval_is_empty() -> None:
    """Empty retrieval never counts as a blanket citation, even with no cited refs."""
    plan = _make_plan(
        predicted_component="editor",
        declared_attribution={
            "component_source": "classifier_top3",
            "summary_cited_issues": [],
            "next_steps_cited_issues": [],
        },
    )
    report = verify_declared_attribution(plan, CLASSIFIER_TOP3, retrieved_numbers=set())

    assert report.blanket_citation is False


# ---------------------------------------------------------------------------
# Tolerant parsing on TriagePlan.declared_attribution
# ---------------------------------------------------------------------------


def test_missing_key_parses_to_none() -> None:
    """No declared_attribution key in the input dict -> field defaults to None."""
    plan = _make_plan(declared_attribution=None)
    assert plan.declared_attribution is None


def test_valid_block_parses_to_declared_attribution() -> None:
    """A well-formed declared_attribution dict parses into a DeclaredAttribution instance."""
    plan = _make_plan(
        declared_attribution={
            "component_source": "model_override",
            "component_override_reason": "deviated on purpose",
            "summary_cited_issues": [1, 2],
            "next_steps_cited_issues": [2],
        },
    )
    assert isinstance(plan.declared_attribution, DeclaredAttribution)
    assert plan.declared_attribution.component_source == "model_override"
    assert plan.declared_attribution.component_override_reason == "deviated on purpose"
    assert plan.declared_attribution.summary_cited_issues == [1, 2]
    assert plan.declared_attribution.next_steps_cited_issues == [2]


def test_malformed_component_source_parses_to_none_without_raising() -> None:
    """An invalid component_source value is tolerated: field becomes None, plan still valid."""
    plan = _make_plan(declared_attribution={"component_source": "banana"})
    assert plan.declared_attribution is None


def test_malformed_string_value_parses_to_none_without_raising() -> None:
    """A completely wrong-shaped value (a string instead of an object) is tolerated to None."""
    plan = _make_plan(declared_attribution="not-an-object")
    assert plan.declared_attribution is None


def test_existing_fields_unaffected_by_malformed_attribution() -> None:
    """Malformed declared_attribution never blocks validation of the rest of the plan."""
    plan = _make_plan(predicted_component="editor", declared_attribution={"component_source": "banana"})
    assert plan.predicted_component == "editor"
    assert plan.priority_guess == "medium"
    assert plan.triage_summary == "Test plan"


# ---------------------------------------------------------------------------
# Prompt content
# ---------------------------------------------------------------------------


def test_system_prompt_contains_attribution_rules() -> None:
    """SYSTEM_PROMPT includes the new ATTRIBUTION RULES block and schema key."""
    from triage_iq.prompts.triage_prompt import SYSTEM_PROMPT

    assert "ATTRIBUTION RULES" in SYSTEM_PROMPT
    assert "declared_attribution" in SYSTEM_PROMPT


def test_system_prompt_priority_guidelines_still_pinned() -> None:
    """The pinned priority-calibration lines (tests/test_api.py:383-390) must be unaffected."""
    from triage_iq.prompts.triage_prompt import SYSTEM_PROMPT

    assert "PRIORITY GUIDELINES" in SYSTEM_PROMPT
    assert "low — cosmetic or non-blocking" in SYSTEM_PROMPT
    assert "medium — reproducible regression with a workaround" in SYSTEM_PROMPT
    assert "high — crash, data loss, auth failure" in SYSTEM_PROMPT
    assert "default to medium" in SYSTEM_PROMPT


def test_few_shot_examples_have_valid_declared_attribution() -> None:
    """Every few-shot assistant turn is valid JSON with a classifier_top3 declared_attribution."""
    from triage_iq.prompts.triage_prompt import build_few_shot_examples

    examples = build_few_shot_examples()
    assistant_turns = [ex for ex in examples if ex["role"] == "assistant"]
    assert len(assistant_turns) == 3

    for turn in assistant_turns:
        data = json.loads(turn["content"])
        assert "declared_attribution" in data
        assert data["declared_attribution"]["component_source"] == "classifier_top3"
