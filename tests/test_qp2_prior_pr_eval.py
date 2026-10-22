"""Tests for scripts/qp2_prior_pr_eval.py: metric math, strict cutoff, slices, BM25, guards."""

from __future__ import annotations

import gzip
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp2_prior_pr_eval as ev  # noqa: E402

SLUG = "kubernetes__kubernetes"
DISC = [1 / math.log2(r + 1) for r in range(1, 11)]


def test_rank_metrics_hand_computed() -> None:
    rel = np.zeros(12, dtype=bool)
    rel[[1, 9]] = True  # relevant at ranks 2 and 10
    m = dict(zip(ev.METRICS, ev.rank_metrics(rel), strict=True))
    idcg = 1 + 1 / math.log2(3)
    assert m["hit@5"] == 1 and m["hit@10"] == 1 and m["mrr"] == 0.5
    assert m["ndcg@10"] == pytest.approx((DISC[1] + DISC[9]) / idcg)
    assert m["p@10"] == pytest.approx(0.2)


def test_rank_metrics_first_positive_beyond_cutoffs_and_empty() -> None:
    rel = np.zeros(20, dtype=bool)
    rel[10] = True  # rank 11
    m = dict(zip(ev.METRICS, ev.rank_metrics(rel), strict=True))
    assert m["hit@5"] == 0 and m["hit@10"] == 0 and m["p@10"] == 0 and m["ndcg@10"] == 0
    assert m["mrr"] == pytest.approx(1 / 11)
    assert np.isnan(ev.rank_metrics(np.zeros(5, dtype=bool))).all()


def test_ties_resolve_to_more_recent_and_query_metrics() -> None:
    # 4 candidates (chronological 0..3), all scores tied: order is most recent first (3,2,1,0).
    q = ev.Query(issue=1, fix=9, m=4, pos=np.array([0]), nt=np.array([], dtype=int))
    out = ev.query_metrics(np.zeros(4), q)
    assert out[0][ev.METRICS.index("mrr")] == pytest.approx(1 / 4)  # oldest PR ranks last
    assert np.isnan(out[1]).all()  # empty non-trivial slice -> NaN row
    best = ev.query_metrics(np.array([5.0, 0, 0, 0]), q)
    assert best[0][ev.METRICS.index("mrr")] == 1.0
    with pytest.raises(AssertionError):
        ev.query_metrics(np.zeros(3), q)


@pytest.mark.parametrize("k", [1, 3, 11])
def test_random_expectation_matches_enumeration(k: int) -> None:
    m = 12  # every placement of the k relevant PRs among m ranks is equally likely
    sims = []
    for pos in itertools.combinations(range(m), k):
        rel = np.zeros(m, dtype=bool)
        rel[list(pos)] = True
        sims.append(ev.rank_metrics(rel))
    np.testing.assert_allclose(ev.random_expectation(m, k), np.mean(sims, axis=0), atol=1e-12)
    assert np.isnan(ev.random_expectation(m, 0)).all()


def test_bm25_tiny_corpus_matches_hand_computation() -> None:
    docs = ["aa bb", "aa cc"]  # default tokenizer drops 1-char tokens
    got = ev.bm25_scores(docs, docs, ["aa bb"])
    idf_a, idf_b = math.log(1 + 0.5 / 2.5), math.log(1 + 1.5 / 1.5)
    # dl == avgdl == 2, tf == 1: saturation factor (k1 + 1) / (1 + k1) == 1
    np.testing.assert_allclose(got, [[idf_a + idf_b, idf_a]])
    longer = ev.bm25_scores(
        docs, ["aa bb xx yy xx yy", "aa cc"], ["aa"]
    )  # unseen terms add only length
    avgdl = 2.0
    expect = 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 6 / avgdl)) * idf_a
    assert longer[0, 0] == pytest.approx(expect)


def test_rrf_and_ranks() -> None:
    a, b = np.array([3.0, 2.0, 1.0]), np.array([1.0, 2.0, 3.0])  # chronological scores
    np.testing.assert_array_equal(ev.ranks(a), [1, 2, 3])
    np.testing.assert_allclose(ev.rrf(a, a), 2 / (60 + ev.ranks(a)))
    fused = ev.rrf(a, b)  # ranks (1,3) and (3,1) tie in the middle: 1/61 + 1/63 for ends
    assert fused[0] == pytest.approx(fused[2]) and fused[1] == pytest.approx(2 / 62)


def test_bootstrap_is_deterministic_and_paired_diff() -> None:
    clusters = np.array([1, 1, 2, 3, 4, 5])
    a = np.tile(np.array([[1.0, 1, 1, 1, 1]]), (6, 1))
    b = np.zeros((6, 5))
    s1 = ev.summarise(a, ev.Boot(clusters))
    s2 = ev.summarise(a, ev.Boot(clusters))
    assert s1 == s2 and s1["hit@10"]["wilson95"][1] == pytest.approx(1.0)
    d = ev.paired(a, b, ev.Boot(clusters))
    assert d["hit@10"]["diff"] == 1.0 and d["hit@10"]["excludes_zero"] is True
    assert ev.paired(a, a, ev.Boot(clusters))["mrr"]["excludes_zero"] is False


def _write_pr(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _build_data(root: Path) -> tuple[Path, Path, Path]:
    """Synthetic k8s-like data: 120 train filler PRs plus a small hand-built validation/test era."""
    ts = pd.Timestamp
    prs: list[tuple[int, str, list[str]]] = [
        (1000 + i, str(ts("2025-03-01T00:00:00Z") + pd.Timedelta(hours=i)), [f"f/train{i}.go"])
        for i in range(120)
    ]
    prs += [
        (1, "2025-07-01T10:00:00Z", ["a.go", "b.go"]),
        (6, "2025-07-02T12:00:00Z", ["a.go"]),
        (2, "2025-07-02T13:00:00Z", ["b.go"]),
        (3, "2025-07-03T10:00:00Z", ["c.go"]),
        (5, "2025-07-04T00:00:00Z", ["a.go"]),  # merged EXACTLY at t_q of issue 101
        (4, "2025-07-05T10:00:00Z", ["a.go"]),
        (10, "2025-07-10T08:00:00Z", ["a.go", "b.go", "go.mod"]),  # F(101)
        (20, "2025-07-20T08:00:00Z", ["a.go"]),  # F(102)
        (7, "2025-12-01T10:00:00Z", ["d.go"]),
        (30, "2026-01-20T10:00:00Z", ["d.go", "e.go"]),  # F(103)
    ]
    rows = [
        {"number": n, "merged": True, "baseRefName": "master", "mergedAt": t, "files_truncated": False,
         "createdAt": "2024-01-01T00:00:00Z", "files": {"nodes": [{"path": f} for f in fs]}}
        for n, t, fs in prs
    ]  # fmt: skip
    _write_pr(rows, root / "raw" / SLUG / "pr_files" / "x.jsonl.gz")
    d2, sp, qe = root / "derived_v2" / SLUG, root / "splits" / SLUG, root / "qemb"
    em = root / "embeddings" / SLUG
    for d in (d2, sp, em, qe):
        d.mkdir(parents=True)
    pd.DataFrame(
        {"number": [n for n, _, _ in prs], "merged": True, "title": [f"pr {n}" for n, _, _ in prs],
         "body": "body"}
    ).to_parquet(d2 / "prs.parquet")  # fmt: skip
    created = {
        101: "2025-07-04T00:00:00Z",
        102: "2025-07-10T00:00:00Z",
        103: "2026-01-05T00:00:00Z",
    }
    issues = pd.DataFrame(
        {"number": list(created), "created_at": pd.to_datetime(list(created.values())),
         "title": ["bug a b", "bug a", "bug d"], "body": "text"}
    )  # fmt: skip
    issues.to_parquet(d2 / "issues.parquet")
    pd.DataFrame(
        {"issue": [101, 102, 103], "pr": [10, 20, 30], "method": "closed_event_closer_pr",
         "pr_merged": True, "pr_merged_at": pd.to_datetime([p[1] for p in prs if p[0] in (10, 20, 30)])}
    ).to_parquet(d2 / "fix_links.parquet")  # fmt: skip
    for name, nums in (("train", []), ("validation", [101, 102]), ("test", [103])):
        pd.DataFrame({"number": nums, "split": name}).to_parquet(sp / f"{name}.parquet")
    (sp / "manifest.json").write_text("{}")
    pr_ids = [n for n, _, _ in prs]
    for folder, seed in ((em, 1), (qe, 2)):
        r = np.random.default_rng(seed)
        for kind, ids in (("prs", pr_ids), ("issues", list(created))):
            if folder is qe and kind == "prs":
                continue
            v = r.normal(size=(len(ids), 8)).astype(np.float16)
            np.save(folder / f"{SLUG}__{kind}_emb.npy", v)
            pd.DataFrame({"repo": "k", "kind": kind, "number": ids}).to_parquet(
                folder / f"{SLUG}__{kind}_ids.parquet"
            )
    top = root / "top100.json"
    top.write_text(json.dumps({"files": ["a.go", *[f"zz/t{i}.go" for i in range(99)]]}))
    return root, top, qe


@pytest.fixture()
def data(tmp_path: Path) -> tuple[Path, Path, Path]:
    return _build_data(tmp_path / "d")


def test_world_strict_cutoff_fix_never_ranked_and_slices(data: tuple[Path, Path, Path]) -> None:
    root, top, qe = data
    w = ev.World(root, qe, top)
    q1, q2 = w.queries["validation"]
    assert (q1.issue, q2.issue) == (101, 102)
    assert q1.m == 124  # 120 fillers + P1, P6, P2, P3; P5 (merged exactly at t_q) is excluded
    assert 5 not in w.nums[: q1.m] and 10 not in w.nums[: q1.m] and 4 not in w.nums[: q1.m]
    assert set(w.nums[q1.pos]) == {1, 6, 2}  # P3 shares no informative file with F
    assert set(w.nums[q1.nt]) == {1, 2}  # P6 shares only the top-100 file a.go
    assert 10 not in w.nums[q1.pos]  # F(q) is never a positive/candidate
    assert q2.m == 126 and set(w.nums[q2.pos]) == {1, 6, 5, 4}  # all share a.go (F2 = [a.go])
    assert len(q2.nt) == 0  # only trivial positives -> leaves the non-trivial slice
    assert [len(w.queries[s]) for s in ("train", "validation", "test")] == [0, 2, 1]


def _scrub(o: object) -> object:
    if isinstance(o, dict):
        return {
            k: _scrub(v)
            for k, v in o.items()
            if k not in ("seconds", "commit") and "seconds" not in k
        }
    return o


def test_end_to_end_determinism_and_test_once_guard(
    data: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    root, top, qe = data
    argv = ["--data-dir", str(root), "--query-emb", str(qe), "--top100", str(top)]
    rep1, rep2 = tmp_path / "r1.json", tmp_path / "r2.json"
    assert ev.main([*argv, "--report", str(rep1), "--stage", "validation"]) == 0
    assert ev.main([*argv, "--report", str(rep2), "--stage", "validation"]) == 0
    v1, v2 = (_scrub(json.loads(p.read_text())) for p in (rep1, rep2))
    assert v1 == v2  # identical inputs + seed 42 -> identical report
    val = v1["validation"]
    assert val["n_answerable"] == 2 and val["scorers"]["bm25"]["non_trivial"]["n"] == 1
    assert val["scorers"]["bm25"]["n_dropped_from_non_trivial_slice"] == 1
    assert val["selected"] in ev.NON_BASELINES and val["strongest_baseline"] in ev.BASELINES
    with pytest.raises(ev.TestGuardError):  # no --final-test
        ev.main([*argv, "--report", str(rep1), "--stage", "test"])
    assert ev.main([*argv, "--report", str(rep1), "--stage", "test", "--final-test"]) == 0
    test = json.loads(rep1.read_text())["test"]
    assert test["n_answerable"] == 1 and set(ev.BASELINES) <= set(test["models"])
    with pytest.raises(ev.TestGuardError):  # a second test evaluation of the same model ids
        ev.main([*argv, "--report", str(rep1), "--stage", "test", "--final-test"])
    with pytest.raises(ev.TestGuardError):  # selection cannot be redone once test exists
        ev.main([*argv, "--report", str(rep1), "--stage", "validation"])
