from __future__ import annotations

import importlib.util
import inspect
import math
import sys
from pathlib import Path

import numpy as np
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp1c", _S / "qp1_conformal.py")
cf = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1c"] = cf
_SPEC.loader.exec_module(cf)

P = np.array([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3]])
SPEC = {s["name"]: s for s in cf.specs()}


def test_score_families_known_answers() -> None:
    u = np.array([0.5, 0.5])
    aps = cf.score_matrix(P, u, SPEC["aps_nonrandomised"])
    np.testing.assert_allclose(aps, [[0.6, 0.9, 1.0], [1.0, 0.5, 0.8]])
    rnd = cf.score_matrix(P, u, SPEC["aps_randomised"])  # mass above + u * p
    np.testing.assert_allclose(rnd, [[0.3, 0.75, 0.95], [0.9, 0.25, 0.65]])
    np.testing.assert_allclose(
        cf.score_matrix(P, np.zeros(2), SPEC["aps_randomised"]), [[0.0, 0.6, 0.9], [0.8, 0.0, 0.5]]
    )  # u = 0: mass strictly above
    raps = cf.score_matrix(P, u, {"name": "raps", "lam": 0.1, "k_reg": 1})
    assert (raps - rnd)[0].tolist() == pytest.approx(
        [0.0, 0.1, 0.2]
    )  # penalty lam * (rank - k_reg)+
    np.testing.assert_allclose(
        cf.score_matrix(P, u, SPEC["lac"]), [[0.4, 0.7, 0.9], [0.8, 0.5, 0.7]]
    )


def test_sets_calibration_and_coverage_known_answer() -> None:
    y = np.array([0, 2])
    s = cf.score_matrix(P, np.zeros(2), SPEC["lac"])[np.arange(2), y]  # [0.4, 0.7]
    assert cf.calibrate(P, y, SPEC["lac"], np.zeros(2), alpha=0.5) == pytest.approx(
        0.7
    )  # k = ceil(1.5) = 2
    assert (
        cf.calibrate(P, y, SPEC["lac"], np.zeros(2)) == math.inf
    )  # alpha .05, n = 2: quantile undefined
    m = cf.predict_mask(P, 0.5, SPEC["lac"], np.zeros(2))
    assert m.tolist() == [[True, False, False], [False, True, False]]
    rep = cf.mask_report(m, y, np.array([[1, 0, 0], [0, 1, 0]], dtype=bool))
    assert rep["coverage_primary"]["k"] == 1 and rep["coverage_any_valid"]["k"] == 2
    assert rep["set_size_mean"] == 1.0 and rep["empty_set_rate"] == 0.0
    assert s.tolist() == pytest.approx([0.4, 0.7])
    empty = cf.predict_mask(P, 0.1, SPEC["lac"], np.zeros(2))
    assert empty.sum() == 0  # LAC may abstain
    aps = cf.predict_mask(P, 0.0, SPEC["aps_nonrandomised"], np.zeros(2))
    assert aps.sum(axis=1).tolist() == [1, 1]  # the pre-registered APS never returns an empty set


def test_choose_rule() -> None:
    def row(n, c, s):  # noqa: ANN001, ANN202
        return {"spec": {"name": n}, "coverage": {"mean": c}, "set_size_mean": {"mean": s}}

    t = [row("a", 0.96, 5.0), row("b", 0.951, 3.0), row("c", 0.94, 1.0)]
    assert cf.choose(t)["spec"]["name"] == "b"  # smallest set among coverage >= .95
    t2 = [row("a", 0.90, 5.0), row("b", 0.93, 9.0), row("c", 0.93, 7.0)]
    assert (
        cf.choose(t2)["spec"]["name"] == "c"
    )  # none reaches .95: highest coverage, then smaller set
    t3 = [row("b", 0.95, 2.0), row("a", 0.95, 2.0)]
    assert cf.choose(t3)["spec"]["name"] == "a"  # name breaks exact ties deterministically


def _synthetic(n: int = 400):
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(n, 5)) * 2
    p = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(5, p=row) for row in p])
    return p, y


def test_repeat_eval_valid_coverage_and_determinism() -> None:
    p, y = _synthetic()
    a = cf.repeat_eval(p, y, SPEC["lac"], reps=20)
    assert a == cf.repeat_eval(p, y, SPEC["lac"], reps=20)
    assert a["coverage"]["mean"] >= 0.93  # exchangeable synthetic data: near-nominal coverage
    sel = cf.select_score(p, y, reps=5)
    assert len(sel["table"]) == len(cf.specs()) == 12
    assert sel["selected"] == cf.select_score(p, y, reps=5)["selected"]


def test_selection_sees_validation_arrays_only_and_final_needs_flag() -> None:
    assert list(inspect.signature(cf.select_score).parameters) == [
        "proba_val",
        "y_val",
        "reps",
        "seed",
    ]
    p, y = _synthetic(100)
    sets = np.eye(5, dtype=bool)[y]
    with pytest.raises(cf.ev.TestGuardError):
        cf.final_sets(p, y, p, y, sets, SPEC["lac"], final=False)
    out = cf.final_sets(p, y, p, y, sets, SPEC["lac"], final=True)
    assert set(out) == {"selected", "pre_registered_aps_nonrandomised", "fixed_top3", "fixed_top4"}
    assert out["fixed_top3"]["set_size_mean"] == 3.0
    assert out["selected"]["coverage_primary"]["rate"] >= 0.9
