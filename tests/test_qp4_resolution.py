from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp1_eval as ev  # noqa: E402
import qp4_features as feat  # noqa: E402
import qp4_resolution as q4  # noqa: E402


def ts(*days: int | None) -> np.ndarray:
    base = np.datetime64("2026-01-01")
    return np.array([base + np.timedelta64(d, "D") if d is not None else np.datetime64("NaT") for d in days], dtype="datetime64[ns]")  # fmt: skip


def test_harrell_c_hand_computed_with_censoring_and_ties() -> None:
    t, e = np.array([1.0, 2, 3, 4]), np.array([True, True, False, True])
    # comparable pairs: (0,1) (0,2) (0,3) (1,2) (1,3); censored 2 and last-event 3 start none
    assert q4.harrell_c(t, e, np.array([4.0, 3, 2, 1])) == 1.0
    assert q4.harrell_c(t, e, np.array([4.0, 2, 3, 1])) == pytest.approx(4 / 5)  # (1,2) discordant
    assert q4.harrell_c(t, e, np.array([4.0, 3, 3, 1])) == pytest.approx(4.5 / 5)  # (1,2) tied


def test_km_and_ipcw_weights_toy() -> None:
    t, e = np.array([1.0, 2, 3, 4, 5]), np.array([True, False, True, True, False])
    x, s = q4.km(t, e)
    assert list(x) == [1, 3, 4]
    assert s == pytest.approx([0.8, 0.8 * 2 / 3, 0.8 * 2 / 3 * 0.5])
    assert q4.km_quantile(x, s, 0.5) == 4
    # censoring KM of T=[1,2,3,4], E=[1,0,1,0]: censorings at 2 (3 at risk) and 4 (1 at risk)
    tt, ee = np.array([1.0, 2, 3, 4]), np.array([True, False, True, False])
    w = q4.ipcw_weights(tt, ee, np.array([1.0, 2, 3, 4]))
    assert w[:3] == pytest.approx([1.0, 1.5, 1.5])
    assert w[3] == pytest.approx(1 / q4.GFLOOR)  # G(4) = 0 is floored, never inf


def test_person_period_expansion_with_censoring() -> None:
    x = np.array([[10.0], [20.0], [30.0]])
    t, e = np.array([0.5, 30.0, 5000.0]), np.array([True, False, True])
    xp, j, y = q4.expand(x, t, e)
    assert len(xp) == 1 + 2 + 9  # event in bin 0; censored in bin 2 -> bins 0,1; event in bin 8
    assert list(j[:3]) == [0, 0, 1] and list(y[:3]) == [1, 0, 0]  # censored bin itself excluded
    assert y.sum() == 2 and y[-1] == 1 and list(xp[1:3, 0]) == [20.0, 20.0]
    assert list(q4.bin_index(np.array([0.0, 1.0, 23.9, 24.0, 1e9]))) == [0, 1, 1, 2, 9]


def test_hazard_to_median_and_quantile() -> None:
    h = np.zeros((1, q4.NB))
    h[0, :3] = [0.2, 0.3, 0.4]  # S = 0.8, 0.56, 0.336 -> first S <= 0.5 in bin 2 = [1d, 3d)
    s = q4.surv_from_hazard(h)
    assert q4.median_point_h(s)[0] == pytest.approx(np.sqrt(24 * 72))
    q = q4.quantile_h(s, 0.5)[0]
    assert 24 < q < 72 and q == pytest.approx(24 + 48 * np.log(0.56 / 0.5) / np.log(0.56 / 0.336))
    assert q4.median_point_h(np.ones((1, q4.NB)))[0] == q4.MID[-1]  # never reaches 0.5 -> last bin
    assert q4.quantile_h(s, 0.1)[0] < q


def test_cqr_and_bucket() -> None:
    lo, hi, y = np.zeros(4), np.ones(4), np.array([0.5, 0.5, 0.5, 3.0])
    assert q4.cqr_q(lo, hi, y, alpha=0.2) == 2.0  # ceil(5 * 0.8) = 4th smallest score
    assert list(q4.bucket(np.array([1.0, 24.0, 200.0, 800.0, 9000.0]))) == [0, 1, 2, 3, 4]


def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "author": ["a", "a", "a", "b"],
        "created_at": ts(0, 10, 20, 15),
        "closed_at": ts(5, 40, None, None),  # issue 1 closes (day 40) after issue 2 is created
        "resolution_hours": [120.0, 720.0, np.nan, np.nan],
    })  # fmt: skip


def test_author_history_uses_only_information_available_at_creation() -> None:
    df = frame()
    h = feat.author_history(df)
    assert list(h["a_n_prior"]) == [0, 1, 2, 0]
    assert list(h["a_n_done"]) == [0, 1, 1, 0]  # row 2: issue 0 closed day 5 <= 20, issue 1 not yet
    assert np.allclose(h["a_med_res_h"], [np.nan, 120.0, 120.0, np.nan], equal_nan=True)
    future = pd.concat([df, pd.DataFrame({"author": ["a"], "created_at": ts(100), "closed_at": ts(101), "resolution_hours": [24.0]})], ignore_index=True)  # fmt: skip
    h2 = feat.author_history(future)
    pd.testing.assert_frame_equal(h2.iloc[:4], h)  # a planted future issue changes no earlier row


def test_open_load_ignores_the_future_and_counts_still_open() -> None:
    cr, cl = ts(0, 10, 20, 30), ts(5, 40, None, None)
    comp = np.array([0, 0, 0, 1])
    load = feat.open_load(cr, cl, comp)
    assert list(load) == [0, 0, 1, 0]  # row2: issue0 closed day5, issue1 open until day 40
    cr2, cl2 = ts(0, 10, 20, 30, 50), ts(5, 40, None, None, None)
    assert list(feat.open_load(cr2, cl2, np.array([0, 0, 0, 1, 0]))[:4]) == list(load)


def test_knn_features_exclude_self_unclosed_and_future() -> None:
    emb = np.array([[1, 0], [0.99, 0.14], [0.98, 0.2], [0, 1]], dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    cr, cl = ts(0, 5, 20, 30), ts(3, 50, 25, None)
    rh = np.array([72.0, 1100.0, 300.0, np.nan])
    ref = np.array([True, True, True, False])
    f = q4.feat.knn_features(emb, cr, cl, rh, ref, k=2)
    assert np.isnan(f[0]).all()  # nothing closed before t_q=0, and not itself
    assert f[1, 2] == 1 and f[1, 0] == pytest.approx(np.log1p(72.0))  # only issue 0 closed by day 5
    assert f[2, 2] == 1  # issue 1 closes at day 50 > 20: unknown at t_q, excluded
    assert f[3, 2] == 2 and f[3, 1] == pytest.approx(0.5)  # day 30: issues 0 and 2 closed
    emb5 = np.vstack([emb, [0, 1]]).astype(np.float32)  # planted: a near-identical FUTURE issue
    f5 = q4.feat.knn_features(emb5, ts(0, 5, 20, 30, 60), ts(3, 50, 25, None, 61), np.append(rh, 24.0), np.array([True, True, True, False, True]), k=2)  # fmt: skip
    np.testing.assert_array_equal(f5[:4], f)


def test_test_split_guard_and_test_once(tmp_path: Path) -> None:
    tab = pd.DataFrame({"split": ["train", "test"], "classifier_eligible": [True, True]})
    assert len(q4.rows(tab, "test", True)) == 1
    with pytest.raises(ev.TestGuardError):
        q4.rows(tab, "test", False)
    root, slug = tmp_path, "r__r"
    out = root / "models" / slug / "qp4"
    out.mkdir(parents=True)
    tab.to_parquet(out / "features.parquet")
    manifest, report = tmp_path / "m.json", tmp_path / "r.json"
    manifest.write_text("{}", encoding="utf-8")
    report.write_text(json.dumps({"models": {q4.IDS["M1"]: {"test": {}}}, "selection": {}}), encoding="utf-8")  # fmt: skip
    with pytest.raises(ev.TestGuardError, match="already evaluated"):
        q4.run("final-test", root, slug, report, manifest, n_boot=2)
