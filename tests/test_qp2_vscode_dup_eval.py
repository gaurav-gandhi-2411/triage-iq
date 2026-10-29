"""Tests for scripts/qp2_vscode_dup_eval.py: time-respecting index, query exclusion, recall math."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp2_vscode_dup_eval as ev  # noqa: E402

# chronological index: positions 0..5; issues 10..15 created at t = 1, 2, 3, 3, 4, 5
NUMBERS = np.array([10, 11, 12, 13, 14, 15])
CREATED = np.array([1, 2, 3, 3, 4, 5], dtype="int64")


def _pairs(rows: list[tuple[int, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["source", "canonical"])


def test_index_is_strictly_before_query_and_excludes_it() -> None:
    qs = ev.build_queries(_pairs([(12, 10), (13, 10), (15, 14)]), NUMBERS, CREATED)
    by = {q.source: q for q in qs}
    assert by[12].m == 2 and by[13].m == 2  # same-timestamp neighbour 13 / 12 is NOT a candidate
    assert by[15].m == 5
    for q in qs:
        assert q.pos >= q.m  # the query itself is never inside its own index
        assert all(t < q.m for t in q.targets)  # no candidate created at/after the query


def test_canonical_created_at_or_after_query_is_unindexed() -> None:
    # canonical 13 has the same timestamp as source 12 -> not indexed; 14 is later -> not indexed
    qs = ev.build_queries(_pairs([(12, 13), (12, 14), (11, 99)]), NUMBERS, CREATED)
    by = {q.source: q for q in qs}
    assert by[12].targets == [] and by[12].n_pairs == 2
    assert by[11].targets == [] and by[11].n_pairs == 1  # canonical 99 is not in the table


def test_source_missing_from_table_is_dropped() -> None:
    assert ev.build_queries(_pairs([(999, 10)]), NUMBERS, CREATED) == []


def test_target_rank_ties_go_to_more_recent() -> None:
    s = np.array([0.5, 0.9, 0.5, 0.5, 0.1])
    assert ev.target_rank(s, 4, 1) == 1
    assert ev.target_rank(s, 4, 0) == 4  # 0.9 beats it, ties at pos 2 and 3 are more recent
    assert ev.target_rank(s, 4, 3) == 2  # only 0.9 beats it; more recent ties absent
    assert ev.target_rank(s, 2, 0) == 2  # prefix m=2 hides the later candidates


def test_query_rank_is_best_target_and_inf_when_unindexed() -> None:
    q = ev.Query(source=15, pos=5, m=5, targets=[0, 1], n_pairs=2)
    s = np.array([0.2, 0.9, 0, 0, 0, 0.0])
    assert ev.query_rank(s, q) == 1
    assert ev.query_rank(s, ev.Query(15, 5, 5, [], 1)) == ev.INF


def test_recall_counts_unindexed_as_miss_and_reports_indexed_only() -> None:
    ranks = np.array([1, 5, 6, ev.INF])  # 4 queries, one unindexed
    t = ev.recall_table(ranks)
    assert t["recall@5"]["k"] == 2 and t["recall@5"]["n"] == 4 and t["recall@5"]["rate"] == 0.5
    assert t["recall@5_indexed_only"]["n"] == 3
    assert t["recall@5_indexed_only"]["rate"] == pytest.approx(2 / 3)
    assert t["recall@1"]["k"] == 1
    assert t["mrr"] == pytest.approx((1 + 0.2 + 1 / 6 + 0) / 4)


def test_score_queries_end_to_end_never_ranks_the_future() -> None:
    x = sp.csr_matrix(np.eye(6))
    queries = ev.build_queries(_pairs([(13, 10), (15, 11)]), NUMBERS, CREATED)
    ranks, per_pair = ev.score_queries(queries, x, x)
    # identity vectors: every earlier candidate scores 0 -> all tie, ties go to the more recent.
    # Query 13 (m=2): canonical pos 0 has 1 more-recent tie (pos 1) -> rank 2; the future
    # candidates (pos 2..5) must not count. Query 15 (m=5): canonical pos 1 -> 3 recent ties.
    assert ranks.tolist() == [2, 4]
    assert per_pair == [[2], [4]]


def test_test_split_requires_final_flag() -> None:
    with pytest.raises(ev.TestGuardError):
        ev.main(["--split", "test"])


def test_scoring_works_without_psutil(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setitem(sys.modules, "psutil", None)  # makes `import psutil` raise ImportError
    monkeypatch.setattr(ev, "_GUARD_WARNED", False)
    x = sp.csr_matrix(np.eye(6))
    queries = ev.build_queries(_pairs([(13, 10)]), NUMBERS, CREATED)
    ranks, _ = ev.score_queries(queries, x, x)
    assert ranks.tolist() == [2]
    assert "guard inactive" in capsys.readouterr().err
