"""Tests for the gold-curation script's cross-split disjointness fix (ADR-0018).

`load_eval_splits` previously unioned temporal_val/temporal_test/classifier_val/
classifier_test without checking either split's train set. Because the temporal split
and the classifier split are computed independently over the same corpus, an issue held
out by one split could simultaneously be a training example in the OTHER split — this
went undetected from the gold set's original curation (2026-04-29) until W5 built the
first disjointness guard for a different script. This test file covers the fix:
`load_train_numbers` and `load_eval_splits`'s new train-set exclusion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import importlib.util

_spec = importlib.util.spec_from_file_location(
    "curate_triage_gold", Path(__file__).parent.parent / "scripts" / "10_curate_triage_gold.py"
)
curate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(curate)


class TestLoadTrainNumbers:

    def test_reads_number_column(self, tmp_path, monkeypatch):
        monkeypatch.setattr(curate, "ROOT", tmp_path)
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)
        pd.DataFrame({"number": [10, 20, 30]}).to_parquet(
            processed / "kubernetes_kubernetes_classifier_train.parquet"
        )
        result = curate.load_train_numbers("kubernetes_kubernetes", "classifier_train")
        assert result == {10, 20, 30}

    def test_missing_file_returns_empty_set(self, tmp_path, monkeypatch):
        monkeypatch.setattr(curate, "ROOT", tmp_path)
        result = curate.load_train_numbers("nonexistent_repo", "temporal_train")
        assert result == set()


class TestLoadEvalSplitsCrossSplitDisjointness:

    def _write_splits(self, tmp_path, repo, val_test_numbers, classifier_train_numbers,
                       temporal_train_numbers):
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True, exist_ok=True)
        base_cols = {
            "number": val_test_numbers,
            "title": [f"t{n}" for n in val_test_numbers],
            "body_clean": [f"b{n}" for n in val_test_numbers],
            "component": ["debug"] * len(val_test_numbers),
            "resolution_hours": [24.0] * len(val_test_numbers),
        }
        empty = pd.DataFrame(base_cols).iloc[0:0]
        pd.DataFrame(base_cols).to_parquet(processed / f"{repo}_temporal_val.parquet")
        empty.to_parquet(processed / f"{repo}_temporal_test.parquet")
        empty.to_parquet(processed / f"{repo}_classifier_val.parquet")
        empty.to_parquet(processed / f"{repo}_classifier_test.parquet")
        pd.DataFrame({"number": classifier_train_numbers}).to_parquet(
            processed / f"{repo}_classifier_train.parquet"
        )
        pd.DataFrame({"number": temporal_train_numbers}).to_parquet(
            processed / f"{repo}_temporal_train.parquet"
        )

    def test_excludes_issue_held_out_by_temporal_but_in_classifier_train(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(curate, "ROOT", tmp_path)
        # Issue 5 is held out by the temporal split (in temporal_val) but is a
        # classifier-train example — this is exactly the ADR-0018 cross-split gap.
        self._write_splits(
            tmp_path, "kubernetes_kubernetes",
            val_test_numbers=[5, 6],
            classifier_train_numbers=[5],
            temporal_train_numbers=[],
        )
        result = curate.load_eval_splits("kubernetes_kubernetes")
        assert list(result["number"]) == [6]

    def test_excludes_issue_held_out_by_classifier_but_in_temporal_train(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(curate, "ROOT", tmp_path)
        self._write_splits(
            tmp_path, "microsoft_vscode",
            val_test_numbers=[1, 2],
            classifier_train_numbers=[],
            temporal_train_numbers=[2],
        )
        result = curate.load_eval_splits("microsoft_vscode")
        assert list(result["number"]) == [1]

    def test_no_overlap_keeps_all_held_out_rows(self, tmp_path, monkeypatch):
        monkeypatch.setattr(curate, "ROOT", tmp_path)
        self._write_splits(
            tmp_path, "kubernetes_kubernetes",
            val_test_numbers=[1, 2, 3],
            classifier_train_numbers=[100],
            temporal_train_numbers=[200],
        )
        result = curate.load_eval_splits("kubernetes_kubernetes")
        assert sorted(result["number"]) == [1, 2, 3]
