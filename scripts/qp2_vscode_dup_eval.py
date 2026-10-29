"""vscode duplicate retrieval evaluation (runbook Q5a; ADR-0066 item 1, D79 item 3).

PRE-REGISTERED DESIGN (2026-10-23; committed before any number is computed).
 * Target (ADR-0066 decision 1, verbatim): "vscode duplicate Recall@5 >= 80% stays the target"
   (spec section 7, unchanged).
 * Task (as scripts/qp0_ceiling_retrieval.py): query = the LATER issue of a duplicate pair
   (``source``), target = its duplicate canonical, from dup_task_pairs.parquet
   (D:/triageiq-data/splits/microsoft__vscode, derived_v2 issues). A query with several pairs has
   several canonicals; it is a hit at k when ANY indexed canonical ranks <= k.
 * Index per query: every vscode issue in derived_v2 issues.parquet (all splits) with created_at
   STRICTLY before the query's created_at (time-respecting), so the query itself is excluded.
 * Text: production recipe, title.strip() + ". " + clean_text(body)[0].strip()
   (scripts/qk_kaggle.build_texts). Sparse vocabulary / IDF / avgdl are fitted on train-split
   issues only (the ceiling script fitted TF-IDF on all issues; this is a deliberate change).
 * Ranking: candidates ordered by descending score; ties go to the MORE RECENT candidate (the
   qp2 prior-PR convention). Rank of a canonical = 1 + #{earlier issues scoring strictly higher}
   + #{tied issues created after the canonical}. Query rank = min over its indexed canonicals.
 * Primary metric: Recall@5 over ALL validation/test queries, a query whose canonicals are all
   unindexed (not in the table, or not created before the query) counts as a MISS. Reported
   also: indexed-only Recall@5 (denominator = queries with >= 1 indexed canonical), Recall@1/10/50,
   MRR (reciprocal query rank, 0 for unindexed), and the ceiling (share of queries with an indexed
   canonical = the maximum attainable recall). Pair-level variants (each pair its own sample)
   are reported alongside. CIs: Wilson 95% for rates; percentile bootstrap (2000 resamples, seed
   42, resampling queries) for MRR and all paired differences.
 * Candidates: B1 TF-IDF (sublinear tf, English stop words, min_df 2, cosine); B2 BM25 (k1 1.2,
   b 0.75, query term counts, idf = ln(1 + (N - df + 0.5) / (df + 0.5)), the qp2 definition);
   D1 BGE dense on the production recipe (Kaggle embeddings; NOT run in this task); H1 RRF(B2, D1)
   k = 60 (needs D1); H2 a learned / fine-tuned retriever if time allows (later).
 * Selection: on VALIDATION Recall@5 (all-queries denominator); ties -> MRR -> the simpler
   candidate (B1 < B2 < D1 < H1 < H2). The strongest baseline is the better of B1/B2 by the same
   rule. The test split is evaluated ONCE, only after every candidate exists and the selection is
   frozen; this script refuses test without --final-test.
 * Success: test Recall@5 of the selected candidate >= 0.80 as a POINT estimate, reported with its
   Wilson 95% CI and the paired bootstrap difference vs the strongest baseline (2000, seed 42);
   if the ceiling is below 0.80 the target is infeasible by construction and is reported as such.
 * This task (Q5a part 1): B1 and B2 on VALIDATION only; no test number is computed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # laptop rule: n_jobs <= 4

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qk_kaggle  # noqa: E402
from qp0_fixlink_audit import wilson  # noqa: E402
from qp2_prior_pr_eval import K1, B  # noqa: E402  (BM25 constants of the qp2 harness)

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
REPO = "microsoft/vscode"
DATA = Path("D:/triageiq-data")
KS = (1, 5, 10, 50)
N_BOOT, SEED = 2000, 42
CHUNK = 128  # queries scored per dense (chunk x index) block: 128 x 110k float64 ~ 110 MB
MIN_FREE_GB = 1.5  # abort rather than starve the machine (runbook RAM rule)
INF = np.iinfo(np.int64).max  # rank of a query with no indexed canonical


class TestGuardError(RuntimeError):
    """Test split requested without --final-test."""

    __test__ = False  # not a pytest class despite the name


@dataclass
class Query:
    """One duplicate query. ``m`` = size of the time-respecting index (a prefix of the
    chronological order); ``targets`` = chronological positions of INDEXED canonicals only."""

    source: int
    pos: int
    m: int
    targets: list[int]
    n_pairs: int


def chrono_order(created: pd.Series, numbers: pd.Series) -> np.ndarray:
    """Row indices sorted by (created_at, number): position = chronological index."""
    return np.lexsort((numbers.to_numpy(), created.to_numpy()))


def build_queries(pairs: pd.DataFrame, numbers: np.ndarray, created: np.ndarray) -> list[Query]:
    """Group (source, canonical) pairs into queries over a chronologically sorted index.

    ``numbers``/``created`` are in chronological order. The index of a query is the prefix of
    issues created STRICTLY before it (``searchsorted`` side='left'), which excludes the query
    itself and any same-timestamp issue; a canonical is indexed only if it lies in that prefix.
    Sources absent from the table are dropped (the caller reports the count).
    """
    pos_of = {int(n): i for i, n in enumerate(numbers)}
    out: list[Query] = []
    for s, g in pairs.groupby("source", sort=True):
        s = int(s)
        if s not in pos_of:
            continue
        m = int(np.searchsorted(created, created[pos_of[s]], side="left"))
        canon = [int(c) for c in g["canonical"]]
        tgt = sorted({pos_of[c] for c in canon if c in pos_of and pos_of[c] < m})
        out.append(Query(s, pos_of[s], m, tgt, len(canon)))
    return out


def target_rank(scores: np.ndarray, m: int, t: int) -> int:
    """1-based rank of chronological position ``t`` among the first ``m`` candidates.

    1 + #strictly higher + #tied and more recent (ties go to the more recent candidate).
    """
    s = scores[:m]
    st = s[t]
    return 1 + int((s > st).sum()) + int(((s == st) & (np.arange(m) > t)).sum())


def query_rank(scores: np.ndarray, q: Query) -> int:
    """Best (minimum) target rank of a query; INF when no canonical is indexed."""
    return min((target_rank(scores, q.m, t) for t in q.targets), default=INF)


def _rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": ci and list(ci)}


def recall_table(ranks: np.ndarray) -> dict[str, Any]:
    """Recall@k (all-queries denominator, unindexed = miss), indexed-only variant, MRR."""
    n, idx = len(ranks), ranks != INF
    n_idx = int(idx.sum())
    out: dict[str, Any] = {"n_queries": n, "n_indexed": n_idx}
    for k in KS:
        hit = int((ranks <= k).sum())
        out[f"recall@{k}"] = _rate(hit, n)
        out[f"recall@{k}_indexed_only"] = _rate(hit, n_idx)
    out["mrr"] = float(per_query_values(ranks)["mrr"].mean()) if n else None
    return out


def per_query_values(ranks: np.ndarray) -> dict[str, np.ndarray]:
    """Per-query metric values (hit@k and reciprocal rank, 0 when unindexed) for bootstrapping."""
    v = {f"recall@{k}": (ranks <= k).astype(float) for k in KS}
    v["mrr"] = np.where(ranks == INF, 0.0, 1.0 / np.minimum(ranks, 10**9))
    return v


def boot_idx(n: int) -> np.ndarray:
    """Shared bootstrap resample indices (2000 x n, seed 42) so paired diffs reuse them."""
    return np.random.default_rng(SEED).integers(0, n, (N_BOOT, n))


def boot_ci(x: np.ndarray, idx: np.ndarray) -> list[float]:
    """Percentile 95% CI of the mean of per-query values ``x``."""
    lo, hi = np.percentile(x[idx].mean(1), [2.5, 97.5])
    return [float(lo), float(hi)]


def pair_level(queries: list[Query], per_pair_ranks: list[list[int]]) -> dict[str, Any]:
    """Each (source, canonical) pair as its own sample; unindexed pairs are misses."""
    flat = [r for rs in per_pair_ranks for r in rs]
    n_pairs = sum(q.n_pairs for q in queries)
    return {
        "n_pairs": n_pairs, "n_pairs_indexed": len(flat),
        **{f"recall@{k}": _rate(sum(r <= k for r in flat), n_pairs) for k in KS},
    }  # fmt: skip


_GUARD_WARNED = False


def _guard_ram() -> None:
    """Abort when free RAM is low; skipped (logged once) when psutil is not installed."""
    global _GUARD_WARNED  # noqa: PLW0603 - warn-once flag
    try:
        import psutil  # lazy and optional: not in requirements.lock, so CI lacks it
    except ImportError:
        if not _GUARD_WARNED:
            print("psutil unavailable: free-RAM guard inactive", file=sys.stderr)
            _GUARD_WARNED = True
        return
    free = psutil.virtual_memory().available / 2**30
    if free < MIN_FREE_GB:
        raise MemoryError(f"free RAM {free:.2f} GB < {MIN_FREE_GB} GB; stopping own process")


def score_queries(
    queries: list[Query], q_mat: sp.csr_matrix, x_mat: sp.csr_matrix
) -> tuple[np.ndarray, list[list[int]]]:
    """(best rank per query, per-target ranks); scores are computed in chunks, never kept whole."""
    ranks = np.empty(len(queries), dtype=np.int64)
    per_pair: list[list[int]] = []
    xt = x_mat.T.tocsc()
    for a in range(0, len(queries), CHUNK):
        _guard_ram()
        blk = queries[a : a + CHUNK]
        s = (q_mat[[q.pos for q in blk]] @ xt).toarray()
        for i, q in enumerate(blk):
            rs = [target_rank(s[i], q.m, t) for t in q.targets]
            per_pair.append(rs)
            ranks[a + i] = min(rs, default=INF)
    return ranks, per_pair


def tfidf_matrix(texts: list[str], fit_mask: np.ndarray) -> sp.csr_matrix:
    """B1: sublinear tf-idf (English stop words, min_df 2), vocabulary/IDF from train issues."""
    v = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, dtype=np.float32)
    v.fit([t for t, f in zip(texts, fit_mask, strict=True) if f])
    return v.transform(texts).tocsr()


def bm25_matrices(texts: list[str], fit_mask: np.ndarray) -> tuple[sp.csr_matrix, sp.csr_matrix]:
    """B2: (query term-count matrix, BM25-weighted doc matrix); stats from train issues only."""
    cv = CountVectorizer()
    x_fit = cv.fit_transform([t for t, f in zip(texts, fit_mask, strict=True) if f])
    n, df = x_fit.shape[0], np.asarray((x_fit > 0).sum(0)).ravel()
    idf = np.log1p((n - df + 0.5) / (df + 0.5))
    avgdl = x_fit.sum() / n
    cnt = cv.transform(texts).tocsr().astype(np.float64)
    an = cv.build_analyzer()
    dl = np.array([len(an(t)) for t in texts])  # true length incl. out-of-vocabulary terms
    x = cnt.copy()
    rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
    tf = x.data.copy()
    x.data = tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl[rows] / avgdl)) * idf[x.indices]
    return cnt, x.tocsr()


def evaluate(
    name: str, queries: list[Query], q_mat: sp.csr_matrix, x_mat: sp.csr_matrix
) -> tuple[dict[str, Any], np.ndarray]:
    """Score one candidate; returns (report block, per-query ranks)."""
    t0 = time.time()
    ranks, per_pair = score_queries(queries, q_mat, x_mat)
    idx = boot_idx(len(queries))
    res = recall_table(ranks)
    for key, v in per_query_values(ranks).items():
        res[key + "_bootstrap95"] = boot_ci(v, idx)
    res["pair_level"] = pair_level(queries, per_pair)
    res["seconds"] = round(time.time() - t0, 1)
    print(f"{name}: recall@5={res['recall@5']['rate']:.4f} mrr={res['mrr']:.4f} {res['seconds']}s")
    return res, ranks


def run(split: str, data: Path = DATA) -> dict[str, Any]:
    """B1 and B2 on one split of the dup task."""
    slug = qk_kaggle.slug_of(REPO)
    t0 = time.time()
    texts_df = qk_kaggle.build_texts(REPO, "issues", data)  # production recipe
    cols = ["number", "created_at"]
    iss = pd.read_parquet(data / "derived_v2" / slug / "issues.parquet", columns=cols)
    assert (texts_df["number"].to_numpy() == iss["number"].to_numpy()).all()  # noqa: S101
    order = chrono_order(iss["created_at"], iss["number"])
    numbers = iss["number"].to_numpy()[order]
    created = iss["created_at"].to_numpy()[order]
    texts = texts_df["text"].to_numpy()[order].tolist()
    del texts_df
    spl = pd.concat(
        [
            pd.read_parquet(data / "splits" / slug / f"{s}.parquet", columns=["number", "split"])
            for s in ("train", "validation", "test")
        ]
    ).set_index("number")["split"]
    fit_mask = np.array([spl.get(int(n)) == "train" for n in numbers])
    pairs = pd.read_parquet(data / "splits" / slug / "dup_task_pairs.parquet")
    pairs = pairs[pairs["split_of_source"] == split]
    queries = build_queries(pairs, numbers, created)
    ceiling = sum(bool(q.targets) for q in queries)
    rep: dict[str, Any] = {
        "design": DESIGN, "split": split, "repo": REPO, "n_index_issues": len(numbers),
        "n_train_issues_fitted": int(fit_mask.sum()), "n_pairs": len(pairs),
        "n_queries": len(queries),
        "n_sources_missing_from_table": int(pairs["source"].nunique()) - len(queries),
        "ceiling_share_queries_with_indexed_canonical": _rate(ceiling, len(queries)),
        "candidates": {},
    }  # fmt: skip
    x1 = tfidf_matrix(texts, fit_mask)
    rep["candidates"]["B1_tfidf"], r1 = evaluate("B1", queries, x1, x1)
    del x1
    qc, x2 = bm25_matrices(texts, fit_mask)
    rep["candidates"]["B2_bm25"], r2 = evaluate("B2", queries, qc, x2)
    idx = boot_idx(len(queries))
    v1, v2 = per_query_values(r1), per_query_values(r2)
    rep["paired_b2_minus_b1"] = {
        k: {"diff": float((v2[k] - v1[k]).mean()), "bootstrap95": boot_ci(v2[k] - v1[k], idx)}
        for k in v1
    }
    rep["total_seconds"] = round(time.time() - t0, 1)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vscode duplicate retrieval baselines (B1, B2)")
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--final-test", action="store_true", help="required for --split test")
    ap.add_argument(
        "--out", type=Path, default=Path("reports/qp2_vscode_dup_validation_2026-10.json")
    )
    a = ap.parse_args(argv)
    if a.split == "test" and not a.final_test:
        raise TestGuardError("test is evaluated once, after selection is frozen: pass --final-test")
    rep = run(a.split)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
