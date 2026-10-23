from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp4_features as feat  # noqa: E402


def ts(*days: int | None) -> np.ndarray:
    base = np.datetime64("2026-01-01")
    return np.array([base + np.timedelta64(d, "D") if d is not None else np.datetime64("NaT") for d in days], dtype="datetime64[ns]")  # fmt: skip


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
    f = feat.knn_features(emb, cr, cl, rh, ref, k=2)
    assert np.isnan(f[0]).all()  # nothing closed before t_q=0, and not itself
    assert f[1, 2] == 1 and f[1, 0] == pytest.approx(np.log1p(72.0))  # only issue 0 closed by day 5
    assert f[2, 2] == 1  # issue 1 closes at day 50 > 20: unknown at t_q, excluded
    assert f[3, 2] == 2 and f[3, 1] == pytest.approx(0.5)  # day 30: issues 0 and 2 closed
    emb5 = np.vstack([emb, [0, 1]]).astype(np.float32)  # planted: a near-identical FUTURE issue
    f5 = feat.knn_features(emb5, ts(0, 5, 20, 30, 60), ts(3, 50, 25, None, 61), np.append(rh, 24.0), np.array([True, True, True, False, True]), k=2)  # fmt: skip
    np.testing.assert_array_equal(f5[:4], f)
