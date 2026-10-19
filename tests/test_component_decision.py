"""Phase 3a: classifier decides the component (spec-quality-push.md sec. 5). Network-free."""

from __future__ import annotations

import pytest
from tests.test_vscode_naive_point import K8S, _assistant, _issue

from triage_iq.models.component_decision import (
    DECIDED_BY_CLASSIFIER,
    DECIDED_BY_LLM_NO_CLASSIFIER,
    DECIDED_BY_OVERRIDE,
    decide_component,
    repair_component_text,
)
from triage_iq.models.triage import DeclaredAttribution, SimilarIssue, TriagePlan

TOP3 = [
    {"label": "area/kubectl", "confidence": 0.6},
    {"label": "area/kubelet", "confidence": 0.3},
    {"label": "area/test", "confidence": 0.1},
]
TITLE, BODY = "Pod stuck", "kubelet never schedules the pod"


def _override(reason: str) -> DeclaredAttribution:
    return DeclaredAttribution(component_source="model_override", component_override_reason=reason)


def _decide(llm: str, da: DeclaredAttribution | None = None, retrieved: set[int] | None = None):
    return decide_component(llm, TOP3, da, TITLE, BODY, retrieved or {7})


def test_llm_equal_to_top1_is_classifier_and_uses_classifier_spelling() -> None:
    d = _decide(" AREA/Kubectl ")
    assert (d.component, d.decided_by, d.changed) == ("area/kubectl", DECIDED_BY_CLASSIFIER, False)
    assert d.llm_proposal == " AREA/Kubectl "


def test_llm_in_top3_without_declared_attribution_loses() -> None:
    d = _decide("area/kubelet")
    assert (d.component, d.decided_by, d.changed) == ("area/kubectl", DECIDED_BY_CLASSIFIER, True)


def test_llm_in_top3_declaring_classifier_source_loses() -> None:
    da = DeclaredAttribution(component_source="classifier_top3")
    assert _decide("area/kubelet", da).component == "area/kubectl"


def test_llm_outside_top3_loses_even_with_a_grounded_reason() -> None:
    d = _decide("area/network", _override("see #7"))
    assert (d.component, d.decided_by) == ("area/kubectl", DECIDED_BY_CLASSIFIER)


def test_override_citing_a_retrieved_issue_is_kept() -> None:
    d = _decide("AREA/KUBELET", _override("duplicate of #7, which was kubelet"))
    assert (d.component, d.decided_by, d.changed) == (
        "area/kubelet",
        DECIDED_BY_OVERRIDE,
        False,
    )  # case-only difference
    assert d.llm_proposal == "AREA/KUBELET"


def test_override_citing_issue_vocabulary_is_kept() -> None:
    assert _decide("area/kubelet", _override("the kubelet scheduling is the cause")).decided_by == (
        DECIDED_BY_OVERRIDE
    )


def test_override_not_tied_to_signal_loses() -> None:
    d = _decide("area/kubelet", _override("seems right to me"), retrieved={99})
    assert (d.component, d.decided_by) == ("area/kubectl", DECIDED_BY_CLASSIFIER)


def test_no_usable_classifier_serves_the_llm_component() -> None:
    d = decide_component("area/x", [{"label": "unknown", "confidence": 0.0}], None, "", "", set())
    assert (d.component, d.decided_by, d.changed) == ("area/x", DECIDED_BY_LLM_NO_CLASSIFIER, False)


def test_repair_is_whole_word_and_case_insensitive() -> None:
    out = repair_component_text("Git and GIT, not github or digit.", "git", "core")
    assert out == "core and core, not github or digit."


def test_repair_handles_slashes_and_noop_cases() -> None:
    assert repair_component_text("Route to area/kubelet now", "area/kubelet", "area/kubectl") == (
        "Route to area/kubectl now"
    )
    assert repair_component_text("same", "SAME", "same") == "same"
    assert repair_component_text("anything", "", "x") == "anything"


# --- TriageAssistant wiring ---------------------------------------------------------------------


def _plan(component: str, da: DeclaredAttribution | None = None) -> TriagePlan:
    return TriagePlan(
        predicted_component=component,
        component_confidence=0.9,
        similar_issues=[SimilarIssue(number=7, similarity=0.9, relevance_note="n")],
        expected_resolution_summary="s",
        expected_resolution_lower_days=1.0,
        expected_resolution_upper_days=2.0,
        priority_guess="low",
        priority_rationale="r",
        suggested_assignee_class=f"{component} owners",
        suggested_next_steps=[f"Check {component} logs", "Escalate"],
        triage_summary=f"Looks like {component}.",
        declared_attribution=da,
    )


def _run(monkeypatch: pytest.MonkeyPatch, flag: bool | None, plan: TriagePlan):
    asst = _assistant(K8S)
    if flag is not None:
        asst.classifier_decides_component = flag
    monkeypatch.setattr(asst, "_call_llm_verbose", lambda signals: (plan, "", {}, "ok", False))
    return asst.triage_with_metadata(_issue())


def test_flag_defaults_off_and_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _assistant(K8S).classifier_decides_component is False
    plan, meta = _run(monkeypatch, None, _plan("b"))  # classifier top-1 is "a"
    assert plan.predicted_component == "b"
    assert plan.triage_summary == "Looks like b."
    assert plan.grounding.component_source == "b"
    assert "component_decided_by" not in meta
    assert "component_llm_proposal" not in meta


def test_flag_on_serves_classifier_repairs_text_and_adds_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, meta = _run(monkeypatch, True, _plan("b"))
    assert plan.predicted_component == "a"
    assert plan.triage_summary == "Looks like a."
    assert plan.suggested_assignee_class == "a owners"
    assert plan.suggested_next_steps == ["Check a logs", "Escalate"]
    assert plan.component_confidence == pytest.approx(0.7)
    assert plan.grounding.component_source == "a"
    assert plan.grounding_status.component_grounded is True
    assert meta["component_decided_by"] == DECIDED_BY_CLASSIFIER
    assert meta["component_llm_proposal"] == "b"


def test_flag_on_keeps_a_validated_override(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, meta = _run(monkeypatch, True, _plan("b", _override("same root cause as #7")))
    assert plan.predicted_component == "b"
    assert plan.triage_summary == "Looks like b."
    assert meta["component_decided_by"] == DECIDED_BY_OVERRIDE


def test_api_exposes_decision_keys_only_when_present() -> None:
    from tests.test_vscode_naive_point import _post

    off = _post({})
    assert "_component_decided_by" not in off
    on = _post({"component_decided_by": "classifier", "component_llm_proposal": "area/x"})
    assert (on["_component_decided_by"], on["_component_llm_proposal"]) == ("classifier", "area/x")
