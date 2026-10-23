"""Phase 3b flag 1: TRIAGE_PRIORITY_MODEL serves the text-only priority model. Network-free."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
from tests.test_component_decision import _plan
from tests.test_vscode_naive_point import K8S, VSCODE, _assistant, _issue

from triage_iq.api import loader
from triage_iq.models.priority import PriorityPrediction

PROBA = {"low": 0.1, "medium": 0.2, "high": 0.7}
TIMING = {
    "system1_latency_ms", "system2_latency_ms", "system3_latency_ms", "system4_latency_ms",
    "total_latency_ms",
}  # fmt: skip


class _FakePriority:
    """Stands in for PriorityModel; records the serve-time inputs it was given."""

    def __init__(self, label: str = "high", fail: bool = False) -> None:
        self.label, self.fail, self.seen = label, fail, []

    def predict(self, title: str, body: str) -> PriorityPrediction:
        self.seen.append((title, body))
        if self.fail:
            raise RuntimeError("boom")
        return PriorityPrediction(self.label, dict(PROBA))


def _run(monkeypatch, plan, model=None, repo: str = K8S, status: str = "ok"):  # noqa: ANN001, ANN202
    asst = _assistant(repo)
    if model is not None:
        asst.priority_model = model
    monkeypatch.setattr(asst, "_call_llm_verbose", lambda signals: (plan, "", {}, status, False))
    return asst.triage_with_metadata(_issue())


def _with_priority(guess: str, rationale: str):  # noqa: ANN202
    plan = _plan("a")
    plan.priority_guess, plan.priority_rationale = guess, rationale
    return plan


def test_flag_default_off_changes_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _assistant(K8S).priority_model is None
    plan, meta = _run(monkeypatch, _with_priority("low", "minor cosmetic issue"))
    assert (plan.priority_guess, plan.priority_rationale) == ("low", "minor cosmetic issue")
    assert not [k for k in meta if "priority" in k]


# sha256 of (plan JSON, sorted metadata keys, metadata minus timing) with the flag OFF, computed on
# origin/main (f515e7b) BEFORE this change by running this exact projection against
# `git archive origin/main`. Any drift means the flag-off path changed (QP3B check C1).
OFF_GOLDEN_SHA256 = "b46f4e97d0abd2679f2ce99c8f4468819703b039f0d5673e3f41a544982a4644"


def test_flag_off_output_identical_to_main(monkeypatch: pytest.MonkeyPatch) -> None:
    blobs = []
    for plan_in in (
        _with_priority("low", "minor"),
        _with_priority("high", "crash on start"),
        _with_priority("medium", "medium impact"),
    ):
        plan, meta = _run(monkeypatch, plan_in)
        blobs.append(
            {
                "plan": json.loads(plan.model_dump_json()),
                "meta": {k: v for k, v in meta.items() if k not in TIMING},
            }
        )
    fb = _assistant(K8S)
    sig = fb._collect_signals(_issue())
    blobs.append(json.loads(fb._make_fallback_plan(sig, reason="x").model_dump_json()))
    digest = hashlib.sha256(json.dumps(blobs, sort_keys=True, default=str).encode()).hexdigest()
    assert digest == OFF_GOLDEN_SHA256


def test_flag_on_serves_model_priority_and_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakePriority("high")
    plan, meta = _run(monkeypatch, _with_priority("low", "minor cosmetic issue"), fake)
    assert plan.priority_guess == "high"
    assert "minor cosmetic issue" not in plan.priority_rationale  # level differs: replaced
    assert meta["priority_decided_by"] == "model"
    assert meta["priority_llm_proposal"] == "low"
    assert meta["priority_model_proba"] == PROBA
    assert fake.seen == [("Pod 99 stuck", "kubelet never schedules")]  # serve-time inputs


def test_contradicting_rationale_is_replaced_by_model_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan, _ = _run(monkeypatch, _with_priority("low", "Low priority, cosmetic."), _FakePriority())
    assert plan.priority_guess == "high"
    assert "low priority" not in plan.priority_rationale.lower().replace("llm proposed low", "")
    assert "high (p=0.70" in plan.priority_rationale
    assert "the LLM proposed low" in plan.priority_rationale


def test_differing_level_replaces_rationale_even_without_a_level_word(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    llm_text = "Documentation question, no crash or data loss."
    plan, meta = _run(monkeypatch, _with_priority("low", llm_text), _FakePriority())
    assert plan.priority_rationale.startswith("Text-only priority model: high (p=0.70")
    assert meta["priority_llm_proposal"] == "low"


def test_agreeing_rationale_is_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, meta = _run(monkeypatch, _with_priority("high", "High: data loss."), _FakePriority())
    assert plan.priority_rationale == "High: data loss."
    assert meta["priority_llm_proposal"] == "high"


def test_model_error_falls_back_to_llm_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, meta = _run(monkeypatch, _with_priority("low", "minor"), _FakePriority(fail=True))
    assert (plan.priority_guess, plan.priority_rationale) == ("low", "minor")
    assert meta["priority_decided_by"] == "llm"
    assert meta["priority_llm_proposal"] == "low"
    assert meta["priority_model_proba"] is None


def test_model_error_on_llm_fallback_plan_has_no_llm_proposal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fb = _with_priority("medium", "x - priority defaulting to medium.")
    plan, meta = _run(monkeypatch, fb, _FakePriority(fail=True), status="degraded_truncated")
    assert plan.priority_guess == "medium"
    assert (meta["priority_decided_by"], meta["priority_llm_proposal"]) == ("llm", None)


@pytest.mark.parametrize("status", ["degraded_provider_error", "parse_failure"])
def test_fallback_plan_uses_model_priority(monkeypatch: pytest.MonkeyPatch, status: str) -> None:
    asst = _assistant(K8S)
    asst.priority_model = _FakePriority("high")
    sig = asst._collect_signals(_issue())
    fb = asst._make_fallback_plan(sig, reason="LLM provider unavailable (x)")
    assert fb.priority_guess == "high"
    assert "high (p=0.70" in fb.priority_rationale
    assert "defaulting to medium" not in fb.priority_rationale
    assert asst._make_fallback_plan(sig).priority_rationale.startswith("LLM parse failure")

    monkeypatch.setattr(asst, "_call_llm_verbose", lambda s: (fb, "", {}, status, False))
    plan, meta = asst.triage_with_metadata(_issue())
    assert plan.priority_guess == "high"
    assert (meta["priority_decided_by"], meta["priority_llm_proposal"]) == ("model", None)


def test_fallback_plan_without_model_still_defaults_to_medium() -> None:
    asst = _assistant(K8S)
    fb = asst._make_fallback_plan(asst._collect_signals(_issue()), reason="x")
    assert fb.priority_guess == "medium"
    assert fb.priority_rationale == "x — priority defaulting to medium."


# --- loader ---------------------------------------------------------------------------------


def test_loader_missing_artifact_warns_once_and_returns_none(tmp_path: Path) -> None:
    records: list[logging.LogRecord] = []

    class _Grab(logging.Handler):  # direct handler: other tests may cut propagation to root
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    h = _Grab(logging.WARNING)
    loader.logger.addHandler(h)
    try:
        assert loader._load_priority_model(tmp_path, "kubernetes_kubernetes") is None
    finally:
        loader.logger.removeHandler(h)
    assert len([r for r in records if "TRIAGE_PRIORITY_MODEL" in r.getMessage()]) == 1


def test_loader_unreadable_artifact_returns_none(tmp_path: Path) -> None:
    (tmp_path / "priority_model_kubernetes_kubernetes.joblib").write_bytes(b"not joblib")
    assert loader._load_priority_model(tmp_path, "kubernetes_kubernetes") is None


def test_loader_is_k8s_only() -> None:
    slugs = set(loader._PRIORITY_MODEL_SLUGS)  # lower-case name: ruff SIM300 reads UPPER as const
    assert slugs == {"kubernetes_kubernetes"}


def test_vscode_unaffected_without_model(monkeypatch: pytest.MonkeyPatch) -> None:
    plan, meta = _run(monkeypatch, _with_priority("low", "minor"), repo=VSCODE)
    assert plan.priority_guess == "low"
    assert not [k for k in meta if "priority" in k]


def test_api_exposes_priority_keys_only_when_present() -> None:
    from tests.test_vscode_naive_point import _post

    assert "_priority_decided_by" not in _post({})
    on = _post(
        {
            "priority_decided_by": "model",
            "priority_llm_proposal": "low",
            "priority_model_proba": PROBA,
        }
    )
    assert (on["_priority_decided_by"], on["_priority_llm_proposal"]) == ("model", "low")
    assert on["_priority_model_proba"] == PROBA
