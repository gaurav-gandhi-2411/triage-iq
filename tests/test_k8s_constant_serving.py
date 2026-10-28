"""G3 (docs/DECISION_LOG_2026-10.md D88): k8s serves the 2023-2025 closed-issue median (35.0954 d) with
a fixed, CQR-calibrated interval. Network-free; drives the real TriageAssistant._collect_signals()."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from tests.test_vscode_naive_point import K8S, VSCODE, _assistant, _issue, _pca, _Predictor

from triage_iq.models import triage as triage_mod
from triage_iq.models.resolution import (
    CONSTANT_INTERVAL_COVERAGE,
    INTERVAL_BASIS_CONSTANT_CQR,
    SERVED_CONSTANT_RESOLUTION,
)

ROOT = Path(__file__).resolve().parents[1]
REPORT = json.loads((ROOT / "reports/k8s_constant_35d_calibration.json").read_text("utf-8"))
POINT, LO, HI = 35.0954, 0.2755, 372.5905
# A huge Q: if v2's adjustment were added on top of the constant interval, the bounds would move.
BIG_Q = {"q_adjustment_hours": 240.0, "target_coverage": 0.8, "empirical_coverage": 0.8,
         "coverage_ci95_lower": 0.78, "coverage_ci95_upper": 0.82}  # fmt: skip


def _calibration_module():
    spec = importlib.util.spec_from_file_location(
        "k8s_constant_calibration", ROOT / "scripts" / "k8s_constant_calibration.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _signals(repo: str, **kw) -> dict:
    a = _assistant(repo, **kw)
    a.conformal_adjustment = BIG_Q
    return a._collect_signals(_issue())


def test_pinned_constants_equal_the_committed_calibration_report() -> None:
    assert SERVED_CONSTANT_RESOLUTION == {"kubernetes_kubernetes": (POINT, LO, HI)}
    assert REPORT["served"] == {"point_days": POINT, "lo_days": LO, "hi_days": HI}
    v = REPORT["validation"]
    assert CONSTANT_INTERVAL_COVERAGE["kubernetes_kubernetes"] == {
        "empirical_coverage": round(v["closed_coverage"], 4),
        "coverage_ci95_lower": round(v["closed_coverage_ci95"][0], 4),
        "coverage_ci95_upper": round(v["closed_coverage_ci95"][1], 4),
    }


def test_k8s_serves_the_constant_point_and_interval_without_double_cqr(monkeypatch) -> None:
    # tests in this module run on the real table (no pre-G3 override)
    sig = _signals(K8S)
    assert (sig["resolution_point_days"], sig["lo_days"], sig["hi_days"]) == (POINT, LO, HI)
    assert sig["resolution_point_source"] == "train_median"
    assert sig["resolution_interval_basis"] == INTERVAL_BASIS_CONSTANT_CQR
    assert sig["resolution_interval_conformal_applied"] is True
    assert sig["resolution_point_clamped"] is False
    assert f"[{LO:.1f}d, {HI:.1f}d]" in sig["prompt"]
    assert sig["prompt"].count("35.1") >= 1, "the prompt must state the constant point"


def test_k8s_constant_is_served_even_when_the_predictor_fails() -> None:
    sig = _signals(K8S, predictor=_Predictor(_pca(), fail=True))
    assert (sig["resolution_point_days"], sig["lo_days"], sig["hi_days"]) == (POINT, LO, HI)
    assert sig["resolution_interval_basis"] == INTERVAL_BASIS_CONSTANT_CQR


def test_k8s_bucket_and_model_diagnostics_are_unchanged_by_the_constant(monkeypatch) -> None:
    new = _signals(K8S)
    monkeypatch.setattr(triage_mod, "SERVED_CONSTANT_RESOLUTION", {})
    old = _signals(K8S)
    for key in ("resolution_bucket", "resolution_conf_pct", "model_point_days", "model_lo_days",
                "model_hi_days", "classifier_top3"):  # fmt: skip
        assert new[key] == old[key], key
    assert new["resolution_point_days"] != old["resolution_point_days"]


def test_vscode_signals_are_identical_with_and_without_the_constant_table(monkeypatch) -> None:
    new = _signals(VSCODE)
    monkeypatch.setattr(triage_mod, "SERVED_CONSTANT_RESOLUTION", {})
    old = _signals(VSCODE)
    stable = lambda d: json.dumps(  # noqa: E731
        {k: v for k, v in d.items() if not k.startswith("_t_")}, sort_keys=True, default=str
    )
    assert stable(new) == stable(old)
    assert new["resolution_interval_basis"] != INTERVAL_BASIS_CONSTANT_CQR


def test_api_reports_the_validation_coverage_of_the_constant_interval() -> None:
    from unittest.mock import patch

    from fastapi.testclient import TestClient
    from tests.test_api import _fake_meta, _make_store

    from triage_iq.api.app import app

    store = _make_store()
    store.conformal_adjustments = {K8S: BIG_Q}
    plan = store.get.return_value.assistant.triage_with_metadata.return_value[0]
    meta = {**_fake_meta(), "resolution_interval_basis": INTERVAL_BASIS_CONSTANT_CQR,
            "resolution_interval_conformal_applied": True}  # fmt: skip
    store.get.return_value.assistant.triage_with_metadata.return_value = (plan, meta)
    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "k", "RATE_LIMIT_ENABLED": "false"}),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        r = c.post("/triage", json={"repo": K8S, "title": "t", "body": "b", "issue_number": 5})
    assert r.status_code == 200
    ci = r.json()["resolution_interval_conformal"]
    assert ci["empirical_coverage"] == pytest.approx(0.9132)
    assert (ci["coverage_ci95_lower"], ci["coverage_ci95_upper"]) == (0.8919, 0.9319)
    # Q is not added again: the interval is the plan's own
    assert ci["lower_days"] == pytest.approx(max(0.0, plan.expected_resolution_lower_days))
    assert ci["upper_days"] == pytest.approx(plan.expected_resolution_upper_days)


def test_calibrate_follows_the_cqr_recipe_on_a_deterministic_series() -> None:
    mod = _calibration_module()
    rng = np.random.default_rng(42)
    hours = np.exp(rng.normal(5.0, 1.5, 4000))  # chronological order = array order
    cal = mod.calibrate(hours)
    k = int(np.floor(len(hours) * 0.6))
    a, b = hours[:k], hours[k:]
    assert (cal["n_a"], cal["n_b"]) == (k, len(hours) - k)
    assert cal["q10_h"] == pytest.approx(np.quantile(a, 0.1))
    lo = np.log1p(cal["q10_h"]) - cal["q_log"]
    hi = np.log1p(cal["q90_h"]) + cal["q_log"]
    assert cal["lo_h"] == pytest.approx(np.expm1(lo)) and cal["hi_h"] == pytest.approx(np.expm1(hi))
    cov = np.mean((np.log1p(b) >= lo) & (np.log1p(b) <= hi))
    assert cov >= 0.8  # split-conformal: calibration-set coverage is at least 1 - alpha
    assert cal["lo_h"] < cal["hi_h"]
