"""Tests for the CQR calibration/eval-set disjointness fix (scripts/10_calibrate_cqr.py).

scripts/10_calibrate_cqr.py previously had zero disjointness checks: the calibration
slice was just "first cal_frac% of temporal_test by created_at", with no guard against
an eval-set issue landing in it. An eval issue whose resolution outcome contributes to
the CQR Q adjustment, then gets re-scored on interval containment as part of the same
eval, violates conformal exchangeability (docs/investigations/gold-set-leakage.md;
confirmed leak: vscode #311836/#311878, k8s #13508/#13784). This file covers the fix:
`split_cal_true_test`'s eval-set exclusion from cal candidacy, and
`load_eval_numbers_by_repo`.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

_spec = importlib.util.spec_from_file_location(
    "calibrate_cqr", Path(__file__).parent.parent / "scripts" / "10_calibrate_cqr.py"
)
calibrate_cqr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(calibrate_cqr)


def _sorted_test_df(numbers: list[int]) -> pd.DataFrame:
    """A temporal_test-shaped frame, already sorted/reset like main() produces it."""
    return pd.DataFrame({"number": numbers}).reset_index(drop=True)


class TestSplitCalTrueTest:

    def test_no_eval_overlap_matches_plain_slice(self):
        df = _sorted_test_df(list(range(10)))
        is_eval = pd.Series([False] * 10)
        cal_df, true_test_df, excluded = calibrate_cqr.split_cal_true_test(df, is_eval, 0.3)
        assert list(cal_df["number"]) == [0, 1, 2]
        assert list(true_test_df["number"]) == [3, 4, 5, 6, 7, 8, 9]
        assert excluded == []

    def test_eval_row_in_cal_window_is_excluded_and_backfilled(self):
        # n=10, cal_frac=0.3 -> n_cal=3. Row at position 1 (number=1) is an eval issue.
        df = _sorted_test_df(list(range(10)))
        is_eval = pd.Series([False, True, False, False, False, False, False, False, False, False])
        cal_df, true_test_df, excluded = calibrate_cqr.split_cal_true_test(df, is_eval, 0.3)
        # Backfilled from the next eligible (non-eval) row: 0, 2, 3 instead of 0, 1, 2.
        assert list(cal_df["number"]) == [0, 2, 3]
        assert excluded == [1]
        # The excluded eval row lands in true_test, not dropped entirely.
        assert 1 in set(true_test_df["number"])
        assert len(cal_df) + len(true_test_df) == len(df)

    def test_eval_row_outside_cal_window_is_untouched(self):
        # Eval issue at position 7 is already outside the n_cal=3 window — no effect.
        df = _sorted_test_df(list(range(10)))
        is_eval = pd.Series([False] * 7 + [True] + [False] * 2)
        cal_df, true_test_df, excluded = calibrate_cqr.split_cal_true_test(df, is_eval, 0.3)
        assert list(cal_df["number"]) == [0, 1, 2]
        assert excluded == []

    def test_multiple_eval_rows_in_cal_window(self):
        df = _sorted_test_df(list(range(10)))
        is_eval = pd.Series([True, True, False, False, False, False, False, False, False, False])
        cal_df, true_test_df, excluded = calibrate_cqr.split_cal_true_test(df, is_eval, 0.3)
        assert list(cal_df["number"]) == [2, 3, 4]
        assert excluded == [0, 1]

    def test_cal_true_test_partition_is_exhaustive(self):
        df = _sorted_test_df(list(range(20)))
        is_eval = pd.Series([i % 6 == 0 for i in range(20)])
        cal_df, true_test_df, _ = calibrate_cqr.split_cal_true_test(df, is_eval, 0.3)
        assert set(cal_df["number"]) | set(true_test_df["number"]) == set(range(20))
        assert set(cal_df["number"]) & set(true_test_df["number"]) == set()


class TestLoadEvalNumbersByRepo:

    def test_reads_repo_scoped_numbers(self, tmp_path, monkeypatch):
        eval_path = tmp_path / "eval_set.jsonl"
        eval_path.write_text(
            '{"repo": "kubernetes/kubernetes", "number": 13508}\n'
            '{"repo": "kubernetes/kubernetes", "number": 13784}\n'
            '{"repo": "microsoft/vscode", "number": 311836}\n',
            encoding="utf-8",
        )
        monkeypatch.setattr(calibrate_cqr, "EVAL_SET_PATH", eval_path)
        result = calibrate_cqr.load_eval_numbers_by_repo()
        assert result == {
            "kubernetes_kubernetes": {13508, 13784},
            "microsoft_vscode": {311836},
        }

    def test_missing_file_returns_empty_dict(self, tmp_path, monkeypatch):
        monkeypatch.setattr(calibrate_cqr, "EVAL_SET_PATH", tmp_path / "does_not_exist.jsonl")
        assert calibrate_cqr.load_eval_numbers_by_repo() == {}
