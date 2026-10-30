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


def test_rerank_ranks_pure_and_tie_to_retriever() -> None:
    s = np.array([0.1, 0.9, 0.9, -1.0])  # candidates in retriever order
    assert x.rerank_ranks(s, "pure").tolist() == [3, 1, 2, 4]  # tie 1,2 -> better retriever rank
    assert x.rerank_ranks(np.zeros(4), "pure").tolist() == [1, 2, 3, 4]  # no signal: unchanged


def test_rerank_ranks_fusion_blends_both_ranks() -> None:
    s = np.array([0.0, 0.0, 5.0])  # cross-encoder loves the retriever-rank-3 candidate
    assert x.rerank_ranks(s, "pure").tolist() == [2, 3, 1]
    assert x.rerank_ranks(s, "rrf10").tolist() == [1, 3, 2]  # fusion: moves it only to 2nd
    s2 = np.array([1.0, 0.9, 0.8, 0.7, 5.0])
    assert x.rerank_ranks(s2, "rrf60")[4] > x.rerank_ranks(s2, "pure")[4]  # large k damps it


def test_final_ranks_keep_retriever_rank_outside_the_list() -> None:
    ret = [x.Retrieved(np.arange(3), [3, 80]), x.Retrieved(np.arange(3), [])]
    sc = [np.array([0.0, 0.0, 9.0]), np.array([1.0, 2.0, 3.0])]
    best, pp = x.final_ranks(ret, sc, "pure")
    assert pp[0] == [1, 80] and best.tolist() == [1, qe.INF]  # 80 > list: unchanged
    best30, _ = x.final_ranks(ret, sc, "pure", n_rerank=2)  # only the top 2 reordered
    assert best30[0] == 3  # target at list rank 3 is outside the reranked prefix


def test_select_x_ties_prefer_simpler() -> None:
    def c(r5: int) -> dict:
        return {"res": {"recall@5": {"k": r5}, "mrr": 0.3}}

    cands = {"X3_pure": c(10), "X1_rrf10": c(10), "X1_pure": c(10), "X2_pure": c(9)}
    assert x.select_x(cands) == "X1_pure"
    assert x.select_x({"X1_rrf10": c(10), "X1_rrf60": c(10)}) == "X1_rrf60"


def test_verdict_needs_gain_and_positive_ci() -> None:
    assert x.verdict(0.02, 0.001) and not x.verdict(0.0199, 0.01) and not x.verdict(0.05, 0.0)


def test_load_scores_requires_exact_rows(tmp_path: Path) -> None:
    ret = [x.Retrieved(np.arange(2), []), x.Retrieved(np.arange(3), [])]
    cand = pd.DataFrame({"query": [9, 9, 8, 8, 8], "cand": [1, 2, 1, 2, 3]})
    ok = cand.assign(score=[0.1, 0.2, 0.3, 0.4, 0.5])
    ok.to_parquet(tmp_path / "ok.parquet")
    got = x.load_scores(tmp_path / "ok.parquet", cand, ret)
    assert [g.tolist() for g in got] == [pytest.approx([0.1, 0.2]), pytest.approx([0.3, 0.4, 0.5])]
    ok.iloc[::-1].reset_index(drop=True).to_parquet(tmp_path / "rev.parquet")
    with pytest.raises(ValueError):
        x.load_scores(tmp_path / "rev.parquet", cand, ret)
    ok.assign(score=[0.1, np.nan, 0.3, 0.4, 0.5]).to_parquet(tmp_path / "nan.parquet")
    with pytest.raises(ValueError):
        x.load_scores(tmp_path / "nan.parquet", cand, ret)


def test_spearman_monotone_and_ties() -> None:
    a = np.array([0.1, 0.5, 0.2, 0.9])
    assert x.spearman(a, a * 3 + 1) == pytest.approx(1.0)  # monotone transform
    assert x.spearman(a, -a) == pytest.approx(-1.0)
    assert x.spearman(np.array([1.0, 1.0, 2.0, 3.0]), np.array([1.0, 1.0, 2.0, 3.0])) == 1.0


def test_latency_stats() -> None:
    s = x.latency_stats([10.0, 20.0, 30.0, 40.0, 100.0])
    assert s["median_ms"] == 30.0 and s["n"] == 5 and s["mean_ms"] == 40.0
    assert 40.0 < s["p95_ms"] <= 100.0
