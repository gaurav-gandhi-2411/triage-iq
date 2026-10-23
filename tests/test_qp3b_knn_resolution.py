from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp1_eval as ev  # noqa: E402
import qp3b_knn_resolution as q3  # noqa: E402


def test_neighbour_closed_after_query_is_censored_at_query_age() -> None:
    created, dur = np.array([0.0, 10.0, 20.0]), np.array([5.0, 100.0, 4.0])
    event = np.array([True, True, False])
    t, e = q3.neighbour_pairs(50.0, created, dur, event)
    # n0 closed at 5 <= 50 -> observed 5; n1 closes at 110 > 50 -> censored at 40; n2 open -> 30
    assert list(t) == [5.0, 40.0, 30.0]
    assert list(e) == [True, False, False]
    t2, e2 = q3.neighbour_pairs(110.0, created, dur, event)  # closing exactly at t_q counts
    assert t2[1] == 100.0 and e2[1]


def test_km_median_defined_and_undefined_falls_back() -> None:
    t, e = np.array([1.0, 2.0, 3.0, 4.0]), np.array([True, True, False, True])
    assert q3.km_q(t, e, 0.5) == 2.0  # S: 0.75, 0.5 -> first S <= 0.5 at t=2
    assert np.isnan(q3.km_q(np.array([1.0, 5.0, 6.0]), np.array([True, False, False]), 0.5))
    assert np.isnan(q3.km_q(np.array([3.0]), np.array([False]), 0.5))  # no events at all
    assert (q3.km_surv(np.array([3.0]), np.array([False])) == 1).all()


def _tab() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": [1, 2, 3, 4],
            "created_at": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"]),
            "T": [24.0, 1000.0, 5.0, 2.0],
            "event": [True, True, True, False],
        }
    )


def test_query_excluded_and_future_issues_not_neighbours() -> None:
    ref = _tab()
    cr, nums = q3._hours(ref["created_at"]), ref["number"].to_numpy()
    sims = np.array([0.1, 0.9, 0.8, 0.7])
    idx = q3.neighbour_ids(sims, cr[1], 2, cr, nums, 10)  # query = issue 2 at its created time
    assert list(nums[idx]) == [1]  # itself excluded; issues 3, 4 are created later
    idx = q3.neighbour_ids(sims, cr[3] + 1, 99, cr, nums, 2)  # unknown query: top-2 by sim
    assert list(nums[idx]) == [2, 3]


def test_knn_k1_uses_only_reference_rows_and_reports_fallback() -> None:
    ref = _tab()
    emb = np.eye(4, dtype=np.float32)
    q = pd.DataFrame({"number": [10], "created_at": pd.to_datetime(["2026-01-10"])})
    naive = {"median_h": 777.0, "q": (1.0, 2.0)}
    out = q3.knn_k1(q, ref, np.eye(4, dtype=np.float32)[[0]], emb, 3, naive)
    # neighbours by sim: only issue 1 has sim 1; the others tie at 0 (stable order 2, 3)
    # pairs: n1 observed 24, n2 closes 1000h after Jan 2 > Jan 10 -> censored, n3 observed 5
    # KM: events at 5 and 24 of 3 -> S=2/3, 1/3 -> median 24 (first S <= 0.5)
    assert out["point"][0] == 24.0 and not out["fell_back"][0]
    out = q3.knn_k1(q, ref, np.eye(4, dtype=np.float32)[[3]], emb, 1, naive)  # only issue 4 (open)
    assert out["fell_back"][0] and out["point"][0] == 777.0


def test_combine_is_geometric_mean() -> None:
    k1 = dict(point=np.array([4.0]), q10=np.array([1.0]), q90=np.array([9.0]), surv=np.ones((1, 9)))
    c = q3.combine_c1(k1, np.array([16.0]), np.array([4.0]), np.array([1.0]))
    assert c["point"][0] == pytest.approx(8.0) and c["q10"][0] == pytest.approx(2.0)
    assert c["q90"][0] == pytest.approx(3.0) and c["risk"][0] == pytest.approx(-8.0)


def test_test_split_guard_and_second_test_refused(tmp_path: Path) -> None:
    with pytest.raises(ev.TestGuardError):
        ev.guard("test", False)
    man = tmp_path / "m.json"
    man.write_text("{}")
    w = ev.ResultsWriter(tmp_path / "r.json", man)
    pred = tmp_path / "p.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(pred)
    w.add(q3.IDS["K1"], "test", {"x": 1}, pred, final=True)
    with pytest.raises(ev.TestGuardError):
        w.add(q3.IDS["K1"], "test", {"x": 1}, pred, final=True)
