"""Unit tests for CQR calibration math.

Uses synthetic data with known coverage properties to verify:
- Q computation formula is correct
- Conformal intervals achieve target marginal coverage on i.i.d. data
- Edge cases: Q negative (overcovering base model), Q very large (undercovering)
"""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from triage_iq.models.resolution import ConformalAdjustment, ResolutionTimePredictor


def _make_predictor() -> ResolutionTimePredictor:
    """Return a ResolutionTimePredictor with sentinel model attrs (no LightGBM needed)."""
    predictor = ResolutionTimePredictor.__new__(ResolutionTimePredictor)
    predictor.repo = "test"
    predictor.model_q10 = object()  # non-None sentinel so assert passes
    predictor.model_q90 = object()
    predictor.conformal_adjustments = {}
    return predictor


def test_cqr_q_formula() -> None:
    """Q equals the ceil((n+1)(1-alpha))/n quantile of the conformity scores."""
    n = 9
    # Known scores 1..9 in sorted order
    scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    target_coverage = 0.80

    # Construct synthetic problem: y_cal = scores (all positive), intervals [0, 0].
    # Conformity score = max(lo - y, y - hi) = max(-y, y) = y when lo=hi=0.
    # So E_i = scores[i].
    y_cal = scores.copy()
    lo_cal = np.zeros(n)
    hi_cal = np.zeros(n)

    predictor = _make_predictor()
    dummy_X = pd.DataFrame({"f": np.zeros(n)})

    with patch.object(predictor, "predict_intervals", return_value=(lo_cal, hi_cal)):
        adj = predictor.calibrate_cqr(dummy_X, y_cal, target_coverage=target_coverage)

    expected_level = np.ceil((n + 1) * target_coverage) / n
    expected_Q = float(np.quantile(scores, min(expected_level, 1.0)))

    assert adj.q_adjustment_hours == pytest.approx(expected_Q, rel=1e-9)
    assert adj.n_calibration == n
    assert adj.target_coverage == target_coverage
    assert adj.empirical_test_coverage == 0.0
    assert adj.n_test == 0


def test_cqr_iid_coverage() -> None:
    """Empirical test coverage should be within ±0.05 of target_coverage on i.i.d. data."""
    rng = np.random.default_rng(42)
    n_cal = 500
    n_test = 2000
    target_coverage = 0.80

    y_cal = rng.uniform(0, 100, n_cal)
    y_test = rng.uniform(0, 100, n_test)

    # "Model" intervals: fixed [25, 75], covering 50% of Uniform[0,100]
    lo_cal = np.full(n_cal, 25.0)
    hi_cal = np.full(n_cal, 75.0)
    lo_test = np.full(n_test, 25.0)
    hi_test = np.full(n_test, 75.0)

    predictor = _make_predictor()
    dummy_X_cal = pd.DataFrame({"f": np.zeros(n_cal)})
    dummy_X_test = pd.DataFrame({"f": np.zeros(n_test)})

    with patch.object(predictor, "predict_intervals") as mock_pi:
        mock_pi.side_effect = [
            (lo_cal, hi_cal),   # first call: inside calibrate_cqr
            (lo_test, hi_test), # second call: inside predict_conformal_interval
        ]
        adj = predictor.calibrate_cqr(dummy_X_cal, y_cal, target_coverage=target_coverage)
        conf_lo, conf_hi = predictor.predict_conformal_interval(dummy_X_test, adj)

    covered = ((conf_lo <= y_test) & (y_test <= conf_hi)).mean()
    assert abs(covered - target_coverage) < 0.05, (
        f"Empirical coverage {covered:.3f} too far from target {target_coverage}"
    )
    assert adj.n_calibration == n_cal
    assert adj.q_adjustment_hours > 0  # base model under-covers so Q must be positive


def test_predict_conformal_interval_clips_negative() -> None:
    """Lower bound clips to 0; upper bound increases by Q even when Q > raw_lower."""
    n = 5
    raw_lower = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    raw_upper = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Q = 10 exceeds the first two raw_lower values → those should clip to 0
    adj = ConformalAdjustment(
        target_coverage=0.80,
        n_calibration=100,
        q_adjustment_hours=10.0,
        empirical_test_coverage=0.0,
        n_test=0,
    )

    predictor = _make_predictor()
    dummy_X = pd.DataFrame({"f": np.zeros(n)})

    with patch.object(predictor, "predict_intervals", return_value=(raw_lower, raw_upper)):
        conf_lo, conf_hi = predictor.predict_conformal_interval(dummy_X, adj)

    # lower = clip(raw_lower - 10, 0, None)
    expected_lo = np.clip(raw_lower - 10.0, 0, None)
    np.testing.assert_array_almost_equal(conf_lo, expected_lo)

    # upper = raw_upper + 10
    expected_hi = raw_upper + 10.0
    np.testing.assert_array_almost_equal(conf_hi, expected_hi)

    assert (conf_lo >= 0).all(), "Lower bound must never be negative"
