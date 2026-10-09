"""ADR-0064 (owner decision D7): both repos serve the training-window median as the point estimate.

Network-free and deterministic. Drives the real TriageAssistant._collect_signals(), the real
engineer_features() and the real prompt builder with stub classifier / retriever / predictor.
"""

from __future__ import annotations

import hashlib
import json
import logging

import numpy as np
import pandas as pd
import pytest

from triage_iq.models.resolution import (
    BUCKET_CLASSIFIER_TRUSTED,
    INTERVAL_RECENTRED,
    NAIVE_INTERVAL_COVERAGE,
    POINT_ESTIMATE_TRUSTED,
    naive_median_days,
    naive_scaled_interval,
    repo_slug,
)
from triage_iq.models.triage import TriageAssistant

K8S = "kubernetes/kubernetes"
VSCODE = "microsoft/vscode"
BASE = ["title_len_chars", "body_len_chars", "day_of_week", "days_since_repo_start"]
EMB = [f"emb_{i}" for i in range(64)]

# sha256 of the k8s signals for the fixed sample below, computed on origin/main (30e6fd8, #150+#153 merged) with the platform-exact projection below, BEFORE this change. Was on origin/fix/k8s-resolution-
# embeddings-reuse (PR #150, e79cc70) BEFORE this change. Any drift means the k8s serving path
# changed; regenerate only after deliberately changing k8s behaviour (and say so in the PR).
K8S_GOLDEN_SHA256 = "d2c0f259d6fac78c276e474e9ecff4f7de8fd06f92dec4a357355efbf64ea1e4"


class _Classifier:
    def predict_proba_calibrated(self, texts):
        return np.array([[0.7, 0.3]])

    def classes_(self):
        return ["a", "b"]


class _Predictor:
    """Deterministic stand-in whose outputs depend on the feature frame (incl. emb_*)."""

    def __init__(self, pca, fail: bool = False) -> None:
        self.feature_names = BASE + EMB
        self.pca = pca
        self.fail = fail

    def _s(self, X) -> float:
        return float(np.abs(X.to_numpy(dtype=float)).sum() % 50.0) + 2.0

    def predict(self, X):
        if self.fail:
            raise RuntimeError("predictor down")
        return np.array([self._s(X) * 24.0])

    def predict_intervals(self, X):
        s = self._s(X)
        return np.array([s * 0.02 * 24.0]), np.array([s * 25.0 * 24.0])

    def predict_bucket(self, X):
        return ["days"], np.array([0.5])


class _Retriever:
    def retrieve(self, text, k=5, exclude_number=None):
        return [{"number": 7, "score": 0.9, "text": "similar"}]

    def retrieve_with_embedding(self, text, k=5, exclude_number=None):
        v = np.random.default_rng(int(exclude_number or 0)).normal(size=768).astype(np.float32)
        return self.retrieve(text, k, exclude_number), v / np.linalg.norm(v)


class _FixedProjection:
    """Stand-in for the fitted PCA: first 64 dims rounded to 2 dp.

    A real ``PCA(...).fit`` on random data is BLAS/platform dependent at ~1e-12, enough to flip the
    k8s golden digest between a laptop and the CI runner (PR #155 first CI run). Slicing + rounding
    is exact on every platform, so the digest only moves when serving behaviour does.
    """

    n_features_in_ = 768

    def transform(self, X):
        return np.round(np.asarray(X, dtype=float)[:, :64], 2)


def _pca() -> _FixedProjection:
    return _FixedProjection()


def _train(resolution_hours=(10.0, 20.0, 30.0, 40.0, 100.0)) -> pd.DataFrame:
    n = len(resolution_hours)
    return pd.DataFrame({
        "number": list(range(1, n + 1)),
        "author": ["a", "b", "a", "c", "b"][:n],
        "created_at": pd.to_datetime([f"2020-01-0{i + 1}" for i in range(n)], utc=True),
        "resolution_hours": list(resolution_hours),
        "state": ["closed"] * n,
    })


def _issue(number: int = 99) -> pd.Series:
    return pd.Series({
        "number": number, "title": f"Pod {number} stuck", "body_clean": "kubelet never schedules",
        "created_at": pd.Timestamp("2020-02-01", tz="UTC"),
    })


def _assistant(repo: str, predictor=None, train=None) -> TriageAssistant:
    return TriageAssistant(
        repo=repo, classifier=_Classifier(), detector=_Retriever(),
        predictor=predictor or _Predictor(_pca()), train_df=_train() if train is None else train,
        groq_api_key="test-key",
    )


# --- flag ------------------------------------------------------------------------------------


def test_flag_defaults_and_alignment_with_bucket_trust() -> None:
    # D7 (2026-10-08): both repos serve the training-window median as the point estimate.
    assert POINT_ESTIMATE_TRUSTED == {"kubernetes_kubernetes": False, "microsoft_vscode": False}
    # k8s keeps the model interval bit-identical (D7); vscode re-centres it on the median.
    assert INTERVAL_RECENTRED == {"kubernetes_kubernetes": False, "microsoft_vscode": True}
    # Same repos, same keys as the bucket gate; an unlisted repo defaults to trusted.
    assert set(POINT_ESTIMATE_TRUSTED) == set(BUCKET_CLASSIFIER_TRUSTED)
    assert POINT_ESTIMATE_TRUSTED.get(repo_slug("some/future-repo"), True) is True
    assert repo_slug(VSCODE) == "microsoft_vscode"


def test_naive_median_days_matches_study_definition() -> None:
    df = _train((10.0, 20.0, 30.0, 40.0, 100.0))
    assert naive_median_days(df) == pytest.approx(30.0 / 24.0)  # median(hours) / 24
    # Open issues (NaN resolution) do not shift the median.
    df2 = pd.concat([df, pd.DataFrame({"resolution_hours": [np.nan, np.nan]})], ignore_index=True)
    assert naive_median_days(df2) == pytest.approx(30.0 / 24.0)


@pytest.mark.parametrize(
    "df",
    [None, pd.DataFrame({"x": [1]}), pd.DataFrame({"resolution_hours": []}),
     pd.DataFrame({"resolution_hours": [np.nan]}), pd.DataFrame({"resolution_hours": [0.0, 0.0]})],
)
def test_naive_median_days_unavailable_returns_none(df) -> None:
    assert naive_median_days(df) is None


# --- vscode serves the naive median ----------------------------------------------------------


def test_vscode_serves_naive_median_point_and_scaled_interval() -> None:
    a = _assistant(VSCODE)
    sig = a._collect_signals(_issue())
    naive = 30.0 / 24.0
    assert sig["resolution_point_source"] == "train_median"
    assert sig["resolution_interval_basis"] == "naive_scaled"
    assert sig["pred_days"] == pytest.approx(naive) == pytest.approx(sig["resolution_point_days"])
    # Model outputs are kept (diagnostics) and the interval is the model's relative width.
    p, lo, hi = sig["model_point_days"], sig["model_lo_days"], sig["model_hi_days"]
    assert sig["lo_days"] == pytest.approx(naive * lo / p)
    assert sig["hi_days"] == pytest.approx(naive * hi / p)
    assert sig["lo_days"] <= sig["pred_days"] <= sig["hi_days"]
    assert f"Point estimate: {naive:.1f} days" in sig["prompt"]
    # Bucket (already the naive prior for vscode) is untouched by the point gate.
    assert sig["resolution_bucket"] == "days"
    assert sig["resolution_conf_pct"] == 50.0


def test_vscode_non_resolution_fields_unchanged_vs_model_path(monkeypatch) -> None:
    served = _assistant(VSCODE)._collect_signals(_issue())
    monkeypatch.setitem(POINT_ESTIMATE_TRUSTED, "microsoft_vscode", True)
    model = _assistant(VSCODE)._collect_signals(_issue())
    assert model["resolution_point_source"] == "model"
    for key in ("classifier_top3", "similar_raw", "resolution_bucket", "resolution_conf_pct"):
        assert served[key] == model[key]
    # Only the System 3 block differs: numbers, plus (ADR-0064) a header and note that say the point
    # is a historical median instead of telling the synthesis model that LightGBM predicted it.
    drop = ("Point estimate:", "80% prediction", "--- SYSTEM 3", "Note: ")
    strip = lambda p: "\n".join(  # noqa: E731
        ln for ln in p.splitlines() if not ln.startswith(drop)
    )
    assert strip(served["prompt"]) == strip(model["prompt"])
    assert served["prompt"] != model["prompt"]
    assert "RESOLUTION TIME ESTIMATE (historical median" in served["prompt"]
    assert "RESOLUTION TIME PREDICTOR (LightGBM)" not in served["prompt"]
    assert "RESOLUTION TIME PREDICTOR (LightGBM)" in model["prompt"]


def test_naive_scaled_interval_guards() -> None:
    assert naive_scaled_interval(4.0, 8.0, 0.8, 80.0) == pytest.approx((0.4, 40.0))
    assert naive_scaled_interval(4.0, 0.0, 0.8, 80.0) is None
    assert naive_scaled_interval(4.0, float("nan"), 0.8, 80.0) is None
    assert naive_scaled_interval(float("inf"), 8.0, 0.8, 80.0) is None


def test_interval_coverage_constants_are_the_measured_c2_values() -> None:
    # reports/vscode_naive_serving_eval.json, split_40_60 / C2_naive_scaled_model_width_stored_Q.
    c = NAIVE_INTERVAL_COVERAGE["microsoft_vscode"]
    assert c == {"empirical_coverage": 0.827, "coverage_ci95_lower": 0.7852,
                 "coverage_ci95_upper": 0.8622}
    assert c["coverage_ci95_lower"] < c["empirical_coverage"] < c["coverage_ci95_upper"]


# --- fallbacks: a request must never fail ----------------------------------------------------


@pytest.mark.parametrize(
    "train", [pd.DataFrame({"number": [1], "author": ["a"], "created_at": pd.to_datetime(["2020-01-01"], utc=True)}),
              _train((np.nan, np.nan, np.nan, np.nan, np.nan))],
)
def test_no_train_median_falls_back_to_model_with_warning(train, caplog) -> None:
    a = _assistant(VSCODE, train=train)
    with caplog.at_level(logging.WARNING, logger="triage_iq.models.triage"):
        sig = a._collect_signals(_issue())
    assert sig["resolution_point_source"] == "model"
    assert sig["resolution_interval_basis"] == "model"
    assert sig["pred_days"] == pytest.approx(sig["model_point_days"])
    assert any("Naive median unavailable" in r.getMessage() for r in caplog.records)


def test_predictor_failure_still_serves_naive_with_fixed_interval() -> None:
    a = _assistant(VSCODE, predictor=_Predictor(_pca(), fail=True))
    sig = a._collect_signals(_issue())
    assert sig["resolution_point_source"] == "train_median"
    assert sig["resolution_interval_basis"] == "model"  # fixed fallback interval, not re-centred
    assert (sig["lo_days"], sig["hi_days"]) == (1.0, 30.0)
    assert sig["pred_days"] == pytest.approx(30.0 / 24.0)


def test_k8s_predictor_failure_serves_median_with_fixed_interval() -> None:
    sig = _assistant(K8S, predictor=_Predictor(_pca(), fail=True))._collect_signals(_issue())
    assert sig["resolution_point_source"] == "train_median"
    assert sig["resolution_interval_basis"] == "model"
    assert (sig["lo_days"], sig["hi_days"]) == (1.0, 30.0)
    assert sig["pred_days"] == pytest.approx(30.0 / 24.0)


# --- k8s bit-identical -----------------------------------------------------------------------


def _k8s_digest() -> str:
    a = _assistant(K8S)
    rows = []
    for n in (11, 12, 13, 14, 15, 16):
        s = a._collect_signals(_issue(n))
        # D7 changes the served POINT (and so the prompt text); everything else the model
        # produces must stay bit-identical, so prompt and pred_days are not in the digest.
        rows.append({k: s[k] for k in (
            "lo_days", "hi_days", "resolution_bucket",
            "resolution_conf_pct", "classifier_top3", "similar_raw")})
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=repr).encode()).hexdigest()


def test_k8s_bucket_interval_and_retrieval_identical_to_pre_change_main() -> None:
    assert _k8s_digest() == K8S_GOLDEN_SHA256


def test_k8s_serves_train_median_with_unchanged_model_interval() -> None:
    sig = _assistant(K8S)._collect_signals(_issue())
    assert sig["resolution_point_source"] == "train_median"
    assert sig["resolution_interval_basis"] == "model"
    assert sig["pred_days"] == pytest.approx(30.0 / 24.0)  # train median(hours) / 24
    assert sig["model_point_days"] != sig["pred_days"]  # the model point is still computed
    assert (sig["lo_days"], sig["hi_days"]) == (sig["model_lo_days"], sig["model_hi_days"])


# --- API: expand-only response ---------------------------------------------------------------

_ADJ = {
    "q_adjustment_hours": 1.2655,
    "target_coverage": 0.80,
    "empirical_coverage": 0.7459,
    "coverage_ci95_lower": 0.6992,
    "coverage_ci95_upper": 0.7876,
}
# Every key the deployed UI / earlier clients read; none may be removed or renamed.
_PRE_EXISTING_KEYS = {
    "predicted_component", "component_confidence", "similar_issues", "expected_resolution_summary",
    "expected_resolution_lower_days", "expected_resolution_upper_days", "resolution_bucket",
    "resolution_confidence_pct", "resolution_interval_conformal", "priority_guess",
    "priority_rationale", "suggested_assignee_class", "suggested_next_steps", "triage_summary",
    "grounding", "grounding_status", "declared_attribution", "abstention_status",
    "classifier_top3", "resolution_model_beats_naive", "_request_id", "_llm_status", "_degraded",
    "_llm_cache_hit", "_model",
}


def _post(meta_extra: dict) -> dict:
    from unittest.mock import patch

    from fastapi.testclient import TestClient
    from tests.test_api import _fake_meta, _make_store

    from triage_iq.api.app import app

    store = _make_store()
    store.conformal_adjustments = {"microsoft/vscode": _ADJ}
    plan = store.get.return_value.assistant.triage_with_metadata.return_value[0]
    store.get.return_value.assistant.triage_with_metadata.return_value = (
        plan,
        {**_fake_meta(), **meta_extra},
    )
    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "k", "RATE_LIMIT_ENABLED": "false"}),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        r = c.post("/triage", json={"repo": "microsoft/vscode", "title": "t", "body": "b",
                                    "issue_number": 5})
    assert r.status_code == 200
    return r.json()


def test_api_response_is_expand_only_and_reports_naive_source() -> None:
    body = _post({"resolution_point_days": 3.8374, "resolution_point_source": "train_median",
                  "resolution_interval_basis": "naive_scaled"})
    assert set(body) >= _PRE_EXISTING_KEYS
    assert body["resolution_point_source"] == "train_median"
    assert body["resolution_point_days"] == pytest.approx(3.8374)
    assert body["resolution_interval_basis"] == "naive_scaled"
    assert body["resolution_model_beats_naive"] is False  # meaning unchanged (measured fact)
    ci = body["resolution_interval_conformal"]
    # Coverage reported is the one measured for the interval actually served, not the artifact one.
    assert ci["empirical_coverage"] == pytest.approx(0.827)
    assert (ci["coverage_ci95_lower"], ci["coverage_ci95_upper"]) == (0.7852, 0.8622)
    assert ci["target_coverage"] == 0.80


def test_api_model_basis_keeps_artifact_coverage_and_defaults_source_to_model() -> None:
    body = _post({})  # older meta without the new keys (e.g. a mocked assistant)
    assert body["resolution_point_source"] == "model"
    assert body["resolution_interval_basis"] == "model"
    assert body["resolution_interval_conformal"]["empirical_coverage"] == pytest.approx(0.7459)


# --- clamp: the served point never sits outside its own interval (GG 2026-10-08, P1b) ----------


def test_k8s_point_below_interval_is_clamped_to_lower_edge_and_flagged() -> None:
    a = _assistant(K8S)  # train median 30 h = 1.25 d
    point, lo, hi, source, basis, clamped = a._apply_point_trust(9.0, 2.0, 40.0)
    assert (point, lo, hi) == (2.0, 2.0, 40.0)  # median 1.25 d < lo 2.0 d -> lo
    assert (source, basis, clamped) == ("train_median", "model", True)


def test_k8s_point_above_interval_is_clamped_to_upper_edge() -> None:
    a = _assistant(K8S)
    point, _, hi, _, _, clamped = a._apply_point_trust(0.5, 0.1, 1.0)
    assert (point, hi, clamped) == (1.0, 1.0, True)  # median 1.25 d > hi 1.0 d -> hi


def test_k8s_point_inside_interval_is_not_clamped() -> None:
    a = _assistant(K8S)
    point, *_, clamped = a._apply_point_trust(9.0, 0.5, 40.0)
    assert (point, clamped) == (pytest.approx(30.0 / 24.0), False)


def test_clamped_point_is_what_the_prompt_shows_and_the_flag_is_in_signals() -> None:
    class _NarrowHigh(_Predictor):
        def predict_intervals(self, X):  # interval entirely above the 1.25 d median
            return np.array([48.0]), np.array([240.0])

    sig = _assistant(K8S, predictor=_NarrowHigh(_pca()))._collect_signals(_issue())
    assert sig["resolution_point_clamped"] is True
    assert sig["pred_days"] == sig["lo_days"] == 2.0
    assert "2.0 days" in sig["prompt"] or "2.0 d" in sig["prompt"]
    assert "1.2 days" not in sig["prompt"]


def test_trusted_path_and_vscode_never_clamp() -> None:
    assert _assistant(VSCODE)._collect_signals(_issue())["resolution_point_clamped"] is False
    a = _assistant(K8S)
    assert a._apply_point_trust(9.0, 0.5, 40.0, model_ok=False)[-1] is False  # fixed 1-30 d fallback


def test_api_exposes_the_clamp_flag_defaulting_false() -> None:
    assert _post({})["resolution_point_clamped"] is False
    assert _post({"resolution_point_clamped": True})["resolution_point_clamped"] is True


def test_vscode_served_coverage_is_one_figure_across_api_eval_and_evidence() -> None:
    """Canonical: 82.7% [78.5, 86.2], n=370, chronological 40/60 split by created_at. The API
    constant, the /eval summary block and the evidence JSON must agree, so README/ADRs that quote
    it have a single source (owner request 2026-10-09)."""
    import json
    from pathlib import Path

    from triage_iq.models.resolution import NAIVE_INTERVAL_COVERAGE

    root = Path(__file__).resolve().parents[1]
    ev = json.loads((root / "reports/vscode_naive_serving_eval.json").read_text("utf-8"))
    c2 = ev["interval"]["split_40_60"]["candidates"]["C2_naive_scaled_model_width_stored_Q"]
    summ = json.loads((root / "reports/eval_summary.json").read_text("utf-8"))
    iv = summ["resolution_served"]["repos"]["microsoft/vscode"]["interval"]
    api = NAIVE_INTERVAL_COVERAGE["microsoft_vscode"]
    assert c2["n"] == iv["n_heldout"] == 370
    assert round(c2["coverage_wilson95"][0], 3) == iv["coverage_heldout"] == api["empirical_coverage"]
    assert [round(x, 4) for x in c2["coverage_wilson95"][1:]] == iv["coverage_heldout_ci95"] == [
        api["coverage_ci95_lower"], api["coverage_ci95_upper"]]
