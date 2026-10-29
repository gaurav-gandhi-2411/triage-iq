from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp1_eval as ev  # noqa: E402
import qp4_resolution as q4  # noqa: E402
import r10_vscode_survival as r10  # noqa: E402

DAY = np.timedelta64(1, "D")


def test_make_targets_censors_open_issues_at_cutoff_and_floors_zero() -> None:
    cutoff = pd.Timestamp("2026-01-11")
    created = pd.Series(pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-10"]))
    closed = pd.Series(pd.to_datetime(["2026-01-02", None, "2026-01-10"]))
    res = pd.Series([24.0, np.nan, 0.0])
    t, e, follow = r10.make_targets(created, closed, res, cutoff)
    assert list(e) == [True, False, True]
    assert t[0] == 24.0 and t[1] == 240.0  # open: hours from creation to the cutoff
    assert t[2] == r10.FLOOR_H  # zero-hour resolution floored, never 0
    assert follow == pytest.approx([240.0, 240.0, 24.0])


def test_load_features_use_only_the_past() -> None:
    created = np.array(["2026-01-10", "2026-01-12", "2026-02-01"], dtype="datetime64[ns]")
    closed = np.array(["2026-01-11", "NaT", "2026-02-05"], dtype="datetime64[ns]")
    res = np.array([24.0, np.nan, 96.0])
    f = r10.load_features(created, closed, res)
    # issue 1 (t=Jan 12): issue 0 created before and already closed (Jan 11) -> 0 open
    assert f["load_all"].tolist()[:2] == [0, 0]
    assert f["rec_n"][1] == 1 and f["rec_med_res_h"][1] == 24.0
    assert f["arr7"][1] == 1 and f["clo7"][1] == 1
    assert np.isnan(f["rec_med_res_h"][0]) and f["rec_n"][0] == 0
    # issue 2 (Feb 1): issue 1 is still open, 90-day window holds only issue 0's closure
    assert f["load_all"][2] == 1 and f["rec_n"][2] == 1
    # planted future: an extra issue created and closed after t_q changes nothing for earlier rows
    c2 = np.append(created, np.datetime64("2026-03-01", "ns"))
    k2 = np.append(closed, np.datetime64("2026-03-02", "ns"))
    f2 = r10.load_features(c2, k2, np.append(res, 5.0))
    pd.testing.assert_frame_equal(f2.iloc[:3].reset_index(drop=True), f)


def test_creation_labels_window_and_train_only_top_k() -> None:
    created = pd.Series(pd.to_datetime(["2026-01-01 00:00:00", "2026-01-02 00:00:00"]))
    nums = pd.Series([1, 2])
    ev_df = pd.DataFrame({
        "number": [1, 1, 2, 2],
        "ts": pd.to_datetime(["2026-01-01 00:00:30", "2026-01-01 02:00:00",
                              "2026-01-02 00:00:10", "2026-01-02 00:00:20"]),
        "action": ["labeled"] * 4, "label": ["bug", "late", "x", "bug"],
    })  # fmt: skip
    n_lab, flags = r10.creation_labels(ev_df, created, nums, np.array([True, False]), k=5)
    assert n_lab.tolist() == [1, 2]  # "late" (2 h after creation) is not a creation-time label
    assert list(flags.columns) == ["lab_bug"]  # top labels come from the pool row only
    assert flags["lab_bug"].tolist() == [1, 1]


def test_c_indexes_match_qp4_harrell_and_constant_is_half() -> None:
    rng = np.random.default_rng(0)
    t, e, risk = rng.exponential(10, 60), rng.random(60) < 0.7, rng.normal(size=60)
    harrell, uno_unit = r10.c_indexes(t, e, risk, np.ones(60), tau=np.inf)
    assert harrell == pytest.approx(q4.harrell_c(t, e, risk))
    assert uno_unit == pytest.approx(harrell)  # unit weights, no truncation -> Harrell
    h0, u0 = r10.c_indexes(t, e, np.zeros(60), np.ones(60))
    assert h0 == 0.5 and u0 == 0.5  # all ties


def test_uno_truncation_ignores_late_events() -> None:
    t, e = np.array([1.0, 2, 100, 200]), np.array([True, True, True, True])
    risk = np.array([4.0, 3, 1, 2])  # pair (100, 200) is discordant, but T_i = 100 > tau
    harrell, uno = r10.c_indexes(t, e, risk, np.ones(4), tau=50)
    assert harrell == pytest.approx(5 / 6) and uno == 1.0


def test_ipcw_mae_weights_closed_issues_only() -> None:
    err = np.array([1.0, 3.0, 100.0])
    e = np.array([True, True, False])
    assert r10.ipcw_mae(err, e, np.array([1.0, 3.0, 9.0])) == pytest.approx((1 + 9) / 4)
    s = r10.Split(pd.DataFrame({"T": [1.0, 2, 3, 4], "event": [True, False, True, False]}))
    assert s.w[:3] == pytest.approx([1.0, 1.5, 1.5])  # same toy as qp4 IPCW test; G of this split


def test_breslow_and_grid_quantile_hand_computed() -> None:
    t, e, lp = np.array([10.0, 20.0, 30.0]), np.array([True, True, False]), np.zeros(3)
    h0, c = r10.breslow_h0(lp, t, e)
    assert c == 0.0
    at = lambda hr: h0[np.searchsorted(r10.GRID_H, hr, "right") - 1]  # noqa: E731
    assert at(5.0) == 0.0 and at(11.0) == pytest.approx(1 / 3)
    assert at(22.0) == pytest.approx(1 / 3 + 1 / 2)
    s = np.exp(-h0)[None, :]
    assert r10.grid_quantile(s, 0.9)[0] == r10.CAP_H  # S never drops to 0.1 -> capped
    q = r10.grid_quantile(s, 0.5)[0]
    assert 11.0 < q <= 22.0  # S = 0.72 after the first event, 0.43 after the second


def test_pop_rows_test_guard_and_population() -> None:
    tab = pd.DataFrame({
        "split": ["train", "validation", "test", "test"],
        "eligible": [True, True, True, False], "leak": [False, False, False, False],
    })  # fmt: skip
    with pytest.raises(ev.TestGuardError):
        r10.pop_rows(tab, "test", "eligible", False)
    assert len(r10.pop_rows(tab, "test", "eligible", True)) == 1
    assert len(r10.pop_rows(tab, "test", "nonleaky", True)) == 2


def test_frozen_split_boundaries_are_reused() -> None:
    manifest = Path(__file__).resolve().parents[1] / "reports" / f"qp0_split_manifest_{r10.SLUG}_2026-10.json"  # fmt: skip
    b = json.loads(manifest.read_text(encoding="utf-8"))["boundaries"]
    assert b == {"test_start": "2026-01-01 00:00:00+00:00", "validation_start": "2025-07-01 00:00:00+00:00"}  # fmt: skip


def test_final_test_refused_when_already_read_or_unfrozen(tmp_path: Path) -> None:
    tab = pd.DataFrame({"split": ["train"], "eligible": [True], "leak": [False]})
    rep = tmp_path / "r.json"
    for body in (
        {"freeze": {"test_read": True}},  # test already read
        {"freeze": {"test_read": False}, "test": {}},  # a test entry exists
        {},  # no freeze record at all: fail closed
    ):
        rep.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(r10.FinalTestError):
            r10.run("final", tab, rep, tmp_path)


def test_criteria_requires_all_three_against_both_baselines() -> None:
    def gain(lo: float) -> dict:
        ci = {"point": 1.0, "ci95": [lo, 2.0], "n": 5}
        return {"median_ae_gain_days": ci, "ipcw_mae_gain_days": ci}

    champ = {"harrell_c": {"ci95": [0.55, 0.6]}}
    ok = r10.criteria({"served": gain(0.1), "naive": gain(0.2)}, champ)
    assert ok["all_met"] and ok["C1"]
    assert not r10.criteria({"served": gain(0.1), "naive": gain(-0.1)}, champ)["all_met"]
    assert not r10.criteria({"served": gain(0.1)}, {"harrell_c": {"ci95": [0.49, 0.6]}})["all_met"]


def _synthetic(n: int = 400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    x = rng.normal(size=(n, 4))
    t_true = np.exp(3 - 1.0 * x[:, 0] + rng.normal(0, 0.5, n))  # x0 high -> fast
    cens = rng.exponential(60, n)
    return x, np.minimum(t_true, cens), t_true <= cens


@pytest.mark.parametrize("fam", ["F4_hazard_lgbm", "F1_coxnet", "F3_gb_cox", "F2_aft"])
def test_each_family_ranks_fast_issues_higher_and_survival_is_valid(fam: str) -> None:
    if fam in ("F1_coxnet", "F3_gb_cox"):
        pytest.importorskip("sksurv")
    if fam == "F2_aft":
        pytest.importorskip("lifelines")
    x, t, e = _synthetic()
    hp = {"F1_coxnet": r10.GRIDS[fam][0], "F2_aft": r10.GRIDS[fam][0]}.get(fam, r10.GRIDS[fam][0])
    if fam == "F3_gb_cox":
        hp = {"n_estimators": 20, "max_depth": 2}
    if fam == "F4_hazard_lgbm":
        hp = {"num_leaves": 7, "learning_rate": 0.1, "n_estimators": 20, "min_child_samples": 20}
    p = r10.fit_family(fam, hp, x, t, e)(x)
    assert p["surv"].shape == (len(x), 9) and (np.diff(p["surv"], axis=1) <= 1e-9).all()
    assert (p["q10"] <= p["point"] + 1e-6).all() and (p["point"] <= p["q90"] + 1e-6).all()
    assert np.corrcoef(p["risk"], x[:, 0])[0, 1] > 0.5  # higher risk = faster = larger x0
    assert r10.c_indexes(t, e, p["risk"], np.ones(len(t)))[0] > 0.6
