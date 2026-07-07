"""Tests for the W5 gold-expansion candidate generator (scripts/w5_t3_generate_candidates.py).

Covers the training-disjointness fix: filter_pool's three-way exclusion (retrieval-train,
classifier-train, temporal-train) plus the two new loader functions
(load_classifier_train_numbers, load_temporal_train_numbers). This closes the correctness
bug where the generator excluded gold and retrieval-train overlap but never checked
classifier_train/temporal_train, which let 77/90 previously-accepted candidates leak into
the ingestion script's disjointness assertion.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from w5_t3_generate_candidates import (  # noqa: E402
    filter_pool,
    load_classifier_train_numbers,
    load_temporal_train_numbers,
)

# ---------------------------------------------------------------------------
# filter_pool
# ---------------------------------------------------------------------------


def _make_pool(numbers: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "number": numbers,
        "component": ["debug"] * len(numbers),
        "resolution_hours": [24.0] * len(numbers),
    })


class TestFilterPool:

    def test_drops_invalid_rows_before_any_disjointness_check(self):
        df = pd.DataFrame({
            "number": [1, 2, 3],
            "component": [None, "debug", "debug"],
            "resolution_hours": [24.0, 0.0, 24.0],
        })
        result, stats = filter_pool(df, set(), set(), set(), set())
        assert list(result["number"]) == [3]
        assert stats["dropped_already_in_current_gold"] == 0

    def test_gold_exclusion_reported_separately_from_training_exclusions(self):
        df = _make_pool([1, 2, 3, 4])
        result, stats = filter_pool(df, gold_numbers={1}, retrieval_train_numbers={2},
                                     classifier_train_numbers=set(), temporal_train_numbers=set())
        assert stats["dropped_already_in_current_gold"] == 1
        assert stats["dropped_retrieval_train_overlap"] == 1
        assert list(result["number"]) == [3, 4]

    def test_each_training_exclusion_reported_independently(self):
        df = _make_pool([1, 2, 3, 4, 5])
        result, stats = filter_pool(
            df,
            gold_numbers=set(),
            retrieval_train_numbers={1},
            classifier_train_numbers={2},
            temporal_train_numbers={3},
        )
        assert stats["dropped_retrieval_train_overlap"] == 1
        assert stats["dropped_classifier_train_overlap"] == 1
        assert stats["dropped_temporal_train_overlap"] == 1
        assert stats["dropped_training_union"] == 3
        assert list(result["number"]) == [4, 5]

    def test_union_drop_count_deduplicates_overlap_across_filters(self):
        # Issue 1 is in both retrieval_train and classifier_train — must be counted
        # once in the union even though it is counted once in each individual filter.
        df = _make_pool([1, 2])
        result, stats = filter_pool(
            df,
            gold_numbers=set(),
            retrieval_train_numbers={1},
            classifier_train_numbers={1},
            temporal_train_numbers=set(),
        )
        assert stats["dropped_retrieval_train_overlap"] == 1
        assert stats["dropped_classifier_train_overlap"] == 1
        assert stats["dropped_training_union"] == 1  # not 2 — same issue, not double-dropped
        assert list(result["number"]) == [2]

    def test_classifier_train_overlap_alone_excludes_candidate(self):
        """Regression test for the correctness bug: classifier_train overlap must be
        excluded even when retrieval_train and temporal_train are clean."""
        df = _make_pool([42])
        result, stats = filter_pool(
            df, gold_numbers=set(), retrieval_train_numbers=set(),
            classifier_train_numbers={42}, temporal_train_numbers=set(),
        )
        assert result.empty
        assert stats["dropped_classifier_train_overlap"] == 1

    def test_temporal_train_overlap_alone_excludes_candidate(self):
        """Regression test: temporal_train overlap must be excluded even when
        retrieval_train and classifier_train are clean."""
        df = _make_pool([99])
        result, stats = filter_pool(
            df, gold_numbers=set(), retrieval_train_numbers=set(),
            classifier_train_numbers=set(), temporal_train_numbers={99},
        )
        assert result.empty
        assert stats["dropped_temporal_train_overlap"] == 1


# ---------------------------------------------------------------------------
# load_classifier_train_numbers / load_temporal_train_numbers
# ---------------------------------------------------------------------------


class TestTrainNumberLoaders:

    def test_load_classifier_train_numbers_reads_number_column(self, tmp_path, monkeypatch):
        import w5_t3_generate_candidates as gen

        monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)
        pd.DataFrame({"number": [10, 20, 30]}).to_parquet(
            processed / "kubernetes_kubernetes_classifier_train.parquet"
        )
        result = load_classifier_train_numbers("kubernetes_kubernetes")
        assert result == {10, 20, 30}

    def test_load_temporal_train_numbers_reads_number_column(self, tmp_path, monkeypatch):
        import w5_t3_generate_candidates as gen

        monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
        processed = tmp_path / "data" / "processed"
        processed.mkdir(parents=True)
        pd.DataFrame({"number": [7, 8]}).to_parquet(
            processed / "microsoft_vscode_temporal_train.parquet"
        )
        result = load_temporal_train_numbers("microsoft_vscode")
        assert result == {7, 8}

    def test_load_classifier_train_numbers_missing_file_returns_empty_set(
        self, tmp_path, monkeypatch
    ):
        import w5_t3_generate_candidates as gen

        monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
        result = load_classifier_train_numbers("nonexistent_repo")
        assert result == set()

    def test_load_temporal_train_numbers_missing_file_returns_empty_set(
        self, tmp_path, monkeypatch
    ):
        import w5_t3_generate_candidates as gen

        monkeypatch.setattr(gen, "REPO_ROOT", tmp_path)
        result = load_temporal_train_numbers("nonexistent_repo")
        assert result == set()
