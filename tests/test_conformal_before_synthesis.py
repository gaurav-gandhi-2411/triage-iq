"""The conformal interval is computed BEFORE synthesis (ADR-0059 addendum 2026-10-09).

Until then the prompt carried the raw model interval (vscode served-path coverage 45.4 pct) while
the API returned the conformal one (82.2 pct): prose and fields described different numbers.
Now one interval is built once and shared by the prompt, the plan fields and the API's
resolution_interval_conformal.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_vscode_naive_point import K8S, VSCODE, _assistant, _issue, _Predictor, _pca

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import prose_interval_check as pic  # noqa: E402

Q_HOURS = 24.0  # Q = 1 day, large enough to see


def _adj(q_hours: float = Q_HOURS) -> dict:
    return {"q_adjustment_hours": q_hours, "target_coverage": 0.8, "empirical_coverage": 0.8,
            "coverage_ci95_lower": 0.78, "coverage_ci95_upper": 0.82}


def _with_adj(repo: str, adj: dict | None, **kw):
    a = _assistant(repo, **kw)
    a.conformal_adjustment = adj
    return a


def test_signals_carry_the_conformal_interval_the_prompt_and_the_plan_use() -> None:
    sig = _with_adj(K8S, _adj())._collect_signals(_issue())
    assert sig["resolution_interval_conformal_applied"] is True
    assert sig["lo_days"] == pytest.approx(max(0.0, sig["raw_lo_days"] - 1.0))
    assert sig["hi_days"] == pytest.approx(sig["raw_hi_days"] + 1.0)
    assert f"[{sig['lo_days']:.1f}d, {sig['hi_days']:.1f}d]" in sig["prompt"]
    assert "80% prediction interval (coverage-calibrated):" in sig["prompt"]


def test_without_an_adjustment_the_prompt_is_the_historical_wording() -> None:
    sig = _with_adj(K8S, None)._collect_signals(_issue())
    assert sig["resolution_interval_conformal_applied"] is False
    assert (sig["lo_days"], sig["hi_days"]) == (sig["raw_lo_days"], sig["raw_hi_days"])
    assert "80% prediction interval: [" in sig["prompt"]
    assert "coverage-calibrated" not in sig["prompt"]


def test_vscode_conformal_applies_on_top_of_the_recentred_interval() -> None:
    sig = _with_adj(VSCODE, _adj())._collect_signals(_issue())
    assert sig["resolution_interval_basis"] == "naive_scaled"
    assert sig["hi_days"] == pytest.approx(sig["raw_hi_days"] + 1.0)


def test_negative_q_shrinks_and_reclamps_the_point_into_the_served_interval() -> None:
    class _Tight(_Predictor):
        def predict_intervals(self, X):  # raw interval [1.2 d, 3.0 d]; median 1.25 d is inside
            return np.array([1.2 * 24.0]), np.array([3.0 * 24.0])

    sig = _with_adj(K8S, _adj(q_hours=-9.6), predictor=_Tight(_pca()))._collect_signals(_issue())
    assert (sig["lo_days"], sig["hi_days"]) == pytest.approx((1.6, 2.6))  # raw [1.2, 3.0] -/+ 0.4
    assert sig["pred_days"] == pytest.approx(1.6)  # median 1.25 d sat inside raw, outside served
    assert sig["resolution_point_clamped"] is True


def test_predictor_failure_skips_the_adjustment() -> None:
    sig = _with_adj(K8S, _adj(), predictor=_Predictor(_pca(), fail=True))._collect_signals(_issue())
    assert sig["resolution_interval_conformal_applied"] is False
    assert (sig["lo_days"], sig["hi_days"]) == (1.0, 30.0)


def test_api_does_not_apply_q_twice() -> None:
    from tests.test_vscode_naive_point import _post

    body = _post({"resolution_interval_conformal_applied": True})
    plan_lo, plan_hi = body["expected_resolution_lower_days"], body["expected_resolution_upper_days"]
    ci = body["resolution_interval_conformal"]
    assert (ci["lower_days"], ci["upper_days"]) == pytest.approx((plan_lo, plan_hi))


def test_api_adds_q_when_the_assistant_did_not() -> None:
    from tests.test_vscode_naive_point import _ADJ, _post

    body = _post({})
    plan_lo, plan_hi = body["expected_resolution_lower_days"], body["expected_resolution_upper_days"]
    q = _ADJ["q_adjustment_hours"] / 24.0
    ci = body["resolution_interval_conformal"]
    assert ci["upper_days"] == pytest.approx(plan_hi + q)
    assert ci["lower_days"] == pytest.approx(max(0.0, plan_lo - q))


def test_loader_hands_the_store_entry_to_each_assistant() -> None:
    src = (Path(__file__).resolve().parents[1] / "src/triage_iq/api/loader.py").read_text("utf-8")
    assert "conformal_adjustment=conformal.get(repo)" in src
    assert src.index("_load_conformal_adjustments(models_dir)") < src.index("TriageAssistant(")


# --- the cassette-level check (eval/prose_interval_check.py) ---------------------------------


def _entry(calibrated: bool, summary: str, lo: float = 1.0, hi: float = 20.0) -> dict:
    label = "80% prediction interval" + (" (coverage-calibrated)" if calibrated else "")
    user = (
        "Repository: kubernetes/kubernetes\n\n--- ISSUE ---\nTitle: t\n\n--- SYSTEM 3: X ---\n"
        f"Point estimate: 3.0 days\n{label}: [{lo:.1f}d, {hi:.1f}d]\nNote\n"
    )
    plan = {"expected_resolution_summary": summary}
    return {"request_messages": [{"role": "user", "content": "Repository: microsoft/vscode"},
                                 {"role": "user", "content": user}],
            "response": {"content": json.dumps(plan)}}


def test_check_fails_a_pre_change_entry_and_passes_a_calibrated_one() -> None:
    assert pic.violations(_entry(False, "typical 1-3 days"))  # raw interval in the prompt
    assert pic.violations(_entry(True, "typical 1-3 days")) == []


def test_check_flags_prose_outside_the_served_interval() -> None:
    v = pic.violations(_entry(True, "typically 1 hour or less", lo=2.8, hi=21.6))
    assert any("no overlap" in x for x in v)
