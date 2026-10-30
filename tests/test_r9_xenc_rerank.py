"""Tests for scripts/r9_xenc_rerank.py: retrieval parity, negative sampling, leak-safe frames."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import qp2_vscode_dup_dense as dd  # noqa: E402
import qp2_vscode_dup_eval as qe  # noqa: E402
import r9_xenc_rerank as x  # noqa: E402


def _toy() -> tuple[list[qe.Query], dd.Scorer, dd.Scorer]:
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(12, 6)).astype(np.float32)
    bm = rng.normal(size=(12, 5)).astype(np.float32)
    queries = [qe.Query(100, 11, 11, [2, 7], 2), qe.Query(101, 8, 8, [], 1)]
    return queries, dd.Scorer(bm, bm), dd.Scorer(emb, emb)


def test_retrieve_matches_rank_with_and_is_a_permutation_prefix() -> None:
    queries, bm, dense = _toy()
    ref = dd.rank_with(queries, bm, dense)[x.RRF_K][0]
    got = x.retrieve(queries, bm, dense, 5)
    assert [min((*g.target_ranks, qe.INF)) for g in got] == list(ref)  # same ranks as R8's path
    for g, q in zip(got, queries, strict=True):
        assert len(set(g.pos)) == len(g.pos) == 5 and g.pos.max() < q.m  # time-respecting
    assert got[1].target_ranks == []  # unindexed query: no target rank, caller counts a miss


def test_retrieve_returns_all_when_index_smaller_than_n() -> None:
    queries, bm, dense = _toy()
    got = x.retrieve(queries[1:], bm, dense, 50)
    assert sorted(got[0].pos.tolist()) == list(range(8))


def test_sample_negatives_drops_own_group_and_is_deterministic() -> None:
    comp = {1: 1, 2: 1, 3: 1}
    cand = np.arange(1, 41)  # issue numbers 1..40, ranked best first
    a = x.sample_negatives(cand, 1, comp, np.random.default_rng(42))
    b = x.sample_negatives(cand, 1, comp, np.random.default_rng(42))
    assert a == b and len(a) == len(set(a)) == x.N_NEG
    assert not {1, 2, 3} & set(a)  # group members never become negatives
    assert max(a) <= x.NEG_POOL + 3  # drawn from the first NEG_POOL left after the removal


def test_sample_negatives_none_when_too_few_eligible() -> None:
    assert x.sample_negatives(np.arange(1, 6), 0, {}, np.random.default_rng(42)) is None


def test_candidate_frame_has_no_label_columns() -> None:
    queries, bm, dense = _toy()
    t = type("T", (), {"numbers": np.arange(1000, 1012)})()
    df = x.candidate_frame(t, queries, x.retrieve(queries, bm, dense, 4))
    assert list(df.columns) == ["query", "cand", "ret_rank"]
    assert df.groupby("query")["ret_rank"].apply(list).tolist() == [[1, 2, 3, 4]] * 2
    assert (df["cand"] < df["query"]).all()  # candidates precede their query


def test_build_groups_counts_and_orders_columns() -> None:
    t = type("T", (), {"numbers": np.arange(1, 61)})()
    pairs = pd.DataFrame({"source": [50, 51], "canonical": [3, 4]})
    comp = {50: 3, 3: 3, 51: 4, 4: 4}
    pos = np.arange(0, 45)  # positions -> numbers 1..45
    ret = {50: x.Retrieved(pos, []), 51: x.Retrieved(pos[:5], [])}  # 51 has too few candidates
    df = x.build_groups(t, pairs, comp, ret)
    assert list(df.columns) == ["anchor", "pos", *[f"neg{i}" for i in range(x.N_NEG)]]
    assert df["anchor"].tolist() == [50] and 3 not in df.iloc[0, 2:].tolist()


def test_meta_is_private() -> None:
    assert x.dataset_meta()["isPrivate"] is True
    m = x.kernel_meta()
    assert m["is_private"] is True and m["code_file"] == "xenc_rerank.py"
    assert any(d.endswith(x.DS_SLUG) for d in m["dataset_sources"])


def test_candidate_recall_gate_constants() -> None:
    assert (x.N_LIST, x.RRF_K, x.REF_HITS_R5) == (50, 10, 524)
    assert pytest.approx(0.4533, abs=1e-4) == 524 / 1156
