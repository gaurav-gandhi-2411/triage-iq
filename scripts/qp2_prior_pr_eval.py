"""Evaluation harness for the k8s prior-PR retrieval task (ADR-0068): baselines + zero-shot dense.

PRE-REGISTERED DESIGN (2026-10-23; written before any scorer was run).
 * Queries, INDEX(q), positives, answerability: scripts/qp2_prior_pr_counts.py (imported) on
   derived_v2; answerable queries only; F(q) is never a candidate (asserted).
 * Ranking: most-recent-first candidates, STABLE sort by descending score (ties -> more recent).
 * Metrics: hit@5, hit@10 (hit rate = Recall@k), MRR (first positive), nDCG@10 (binary gains),
   precision@10 (denominator 10). Slices: ALL positives, and NON-TRIVIAL (relevant = positives that
   share >= 1 informative file outside the frozen top-100 list; trivial-only positives stay in the
   ranking as non-relevant; queries with no non-trivial positive leave the slice and are counted).
 * CIs: cluster bootstrap over queries clustered by fix PR (2000 resamples, seed 42, percentile
   95%); Wilson 95% for hit rates; paired differences reuse the same resampled clusters.
 * Scorers: random (analytic expectation per query + a seeded simulation, seed 42); recent;
   popularity (sum over the PR's informative files of their train-period change counts); bm25
   (k1 1.2, b 0.75, query term counts, vocabulary/IDF/avgdl from train-period index PRs,
   idf = ln(1 + (N - df + 0.5) / (df + 0.5))); tfidf (sklearn, fit on train-period PR texts,
   cosine); dense_sym (BGE issue embeddings encoded as documents, cosine); dense_query (issues
   re-encoded with the BGE query instruction; skipped when unavailable); hybrid (reciprocal rank
   fusion, k = 60, of bm25 and the better dense variant by validation nDCG@10, tie hit@10).
   Texts are those qk_kaggle.py embeds (issue title+body[:4000]; PR title+body[:2000]+paths).
 * Selection: best non-baseline (dense_*, hybrid) by validation nDCG@10 (tie hit@10). Strongest
   baseline: same rule over random / recent / popularity / bm25 / tfidf. Test: every baseline and
   the selected scorer, one evaluation per model id (a second one is refused).
 * Success (ADR-0068): paired CI of selected minus strongest baseline excludes 0 (hit@10, nDCG@10).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")  # laptop rule: n_jobs <= 4

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qk_kaggle  # noqa: E402
import qp2_prior_pr_counts as c  # noqa: E402
from qp0_fixlink_audit import wilson  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
METRICS = ("hit@5", "hit@10", "mrr", "ndcg@10", "p@10")
N_BOOT, SEED, RRF_K, K1, B = 2000, 42, 60, 1.2, 0.75
DISC = 1.0 / np.log2(np.arange(2, 12))
CUM = np.cumsum(DISC)
BASELINES = ("random", "random_sim", "recent", "popularity", "bm25", "tfidf")
NON_BASELINES = ("dense_sym", "dense_query", "hybrid")
REPO = "kubernetes/kubernetes"
DATA = Path("D:/triageiq-data")


class TestGuardError(RuntimeError):
    """Test evaluated without --final-test, or twice for one model id."""

    __test__ = False  # not a pytest class despite the name


@dataclass
class Query:
    """Positions are chronological indices into the index; ``m`` = candidates merged before t_q."""

    issue: int
    fix: int
    m: int
    pos: np.ndarray
    nt: np.ndarray


def rank_metrics(rel: np.ndarray) -> np.ndarray:
    """Metric vector for a boolean relevance list in ranked order (NaN when nothing is relevant)."""
    n = int(rel.sum())
    if n == 0:
        return np.full(len(METRICS), np.nan)
    top = rel[:10]
    dcg = float((top * DISC[: len(top)]).sum())
    return np.array(
        [rel[:5].any(), top.any(), 1 / (int(np.argmax(rel)) + 1), dcg / CUM[min(n, 10) - 1], top.sum() / 10],
        dtype=float,
    )  # fmt: skip


def rank_order(scores: np.ndarray) -> np.ndarray:
    """Indices into the RECENCY-ordered candidate list, best first (stable: ties -> more recent)."""
    return np.argsort(-scores[::-1], kind="stable")


def query_metrics(scores: np.ndarray, q: Query) -> np.ndarray:
    """2 x 5 metrics (all positives, non-trivial) of one query given chronological scores."""
    assert scores.shape == (q.m,), "one score per candidate"  # noqa: S101
    order, out = rank_order(scores), np.full((2, len(METRICS)), np.nan)
    for row, idx in enumerate((q.pos, q.nt)):
        if len(idx):
            rel = np.zeros(q.m, dtype=bool)
            rel[idx] = True
            out[row] = rank_metrics(rel[::-1][order])
    return out


def random_expectation(m: int, k: int) -> np.ndarray:
    """Exact expected metrics of a uniformly random ranking of m candidates with k relevant."""
    if k == 0:
        return np.full(len(METRICS), np.nan)

    def hit(kk: int) -> float:
        i = np.arange(kk)
        return float(1 - np.prod(np.clip(m - k - i, 0, None) / (m - i)))

    j = np.arange(1, m - k + 2)  # P(first relevant at j) by recursion on the hypergeometric
    ratio = (m - k - j[:-1] + 1) / (m - j[:-1])
    p = (k / m) * np.concatenate([[1.0], np.cumprod(ratio)])
    ndcg = (k / m) * DISC.sum() / CUM[min(k, 10) - 1]
    return np.array([hit(5), hit(10), float((p / j).sum()), ndcg, k / m])


class Boot:
    """Cluster bootstrap: resamples fix-PR clusters (same indices for every scorer)."""

    def __init__(self, clusters: np.ndarray) -> None:
        _, self.inv = np.unique(clusters, return_inverse=True)
        self.n_c = int(self.inv.max()) + 1
        self.cnt = np.bincount(self.inv, minlength=self.n_c).astype(float)
        self.idx = np.random.default_rng(SEED).integers(0, self.n_c, (N_BOOT, self.n_c))

    def means(self, m: np.ndarray) -> np.ndarray:
        s = np.zeros((self.n_c, m.shape[1]))
        np.add.at(s, self.inv, m)
        return s[self.idx].sum(1) / self.cnt[self.idx].sum(1)[:, None]


def summarise(m: np.ndarray, boot: Boot, analytic: bool = False) -> dict[str, Any]:
    lo, hi = np.percentile(boot.means(m), [2.5, 97.5], axis=0)
    out: dict[str, Any] = {"n": len(m), "n_clusters": boot.n_c}
    for j, name in enumerate(METRICS):
        out[name] = {"mean": float(m[:, j].mean()), "ci95": [float(lo[j]), float(hi[j])]}
        if name.startswith("hit") and not analytic:
            out[name]["wilson95"] = list(wilson(int(m[:, j].sum()), len(m)) or [])
    return out


def paired(ma: np.ndarray, mb: np.ndarray, boot: Boot) -> dict[str, Any]:
    """Selected-minus-comparator differences with percentile CIs on shared resamples."""
    d = boot.means(ma) - boot.means(mb)
    lo, hi = np.percentile(d, [2.5, 97.5], axis=0)
    return {
        name: {"diff": float(ma[:, j].mean() - mb[:, j].mean()), "ci95": [float(lo[j]), float(hi[j])],
               "excludes_zero": bool(lo[j] > 0 or hi[j] < 0)}
        for j, name in enumerate(METRICS)
    }  # fmt: skip


def bm25_scores(fit: list[str], docs: list[str], queries: list[str]) -> np.ndarray:
    """Dense (queries x docs) BM25 matrix; vocabulary, IDF and avgdl come from ``fit`` only."""
    cv = CountVectorizer()
    x_fit = cv.fit_transform(fit)
    n, df = x_fit.shape[0], np.asarray((x_fit > 0).sum(0)).ravel()
    idf = np.log1p((n - df + 0.5) / (df + 0.5))
    avgdl = x_fit.sum() / n
    x = cv.transform(docs).tocsr().astype(float)
    dl = np.array(list(map(len, map(cv.build_analyzer(), docs))))  # true length incl. unseen terms
    rows = np.repeat(np.arange(x.shape[0]), np.diff(x.indptr))
    tf = x.data.copy()
    x.data = tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl[rows] / avgdl)) * idf[x.indices]
    return (cv.transform(queries).astype(float) @ x.T).toarray()


def ranks(scores: np.ndarray) -> np.ndarray:
    """1-based rank per chronological position under the stable most-recent-first tie rule."""
    r = np.empty(len(scores))
    r[rank_order(scores)] = np.arange(1, len(scores) + 1)
    return r[::-1]


def rrf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Reciprocal rank fusion (k = 60) of two chronological score rows."""
    return 1 / (RRF_K + ranks(a)) + 1 / (RRF_K + ranks(b))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class World:
    """Index, per-split answerable queries and texts/embeddings (all read-only inputs)."""

    def __init__(self, data_dir: Path, query_emb: Path | None, top100: Path) -> None:
        slug = qk_kaggle.slug_of(REPO)
        prs = c.load_prs(data_dir / "raw" / slug / "pr_files")
        self.by_num = {p["number"]: p for p in prs}
        excl = c.excluded_files(prs, "master")
        frozen = json.loads(top100.read_text(encoding="utf-8"))
        self.top100 = set(frozen["files"])
        assert len(self.top100) == 100  # noqa: S101
        self.index = c.Index(prs, "master")
        self.nums = np.array(list(self.index.pr_files))  # insertion order == (mergedAt, number)
        self.pos_of = {n: i for i, n in enumerate(self.nums)}
        sp, d2 = data_dir / "splits" / slug, data_dir / "derived_v2" / slug
        split = pd.concat([pd.read_parquet(sp / f"{s}.parquet", columns=["number", "split"]) for s in c.SPLITS])  # fmt: skip
        issues = pd.read_parquet(d2 / "issues.parquet", columns=["number", "created_at"])
        links = pd.read_parquet(d2 / "fix_links.parquet")
        raw_q = c.fix_queries(issues, links, split.set_index("number")["split"])
        self.queries: dict[str, list[Query]] = {s: [] for s in c.SPLITS}
        self.unanswerable = dict.fromkeys(c.SPLITS, 0)
        for q in raw_q:
            f = self.by_num.get(q["fix"])
            info = {x for x in f["files"] if x not in excl} if f else set()
            pos = self.index.positives(info, q["t"], q["fix"]) if info else set()
            m = self.index.size_before(q["t"])
            if not pos:
                self.unanswerable[q["split"]] += 1
                continue
            assert q["fix"] not in set(self.nums[:m].tolist())  # noqa: S101 - F(q) never ranked
            nt = [n for n in pos if (self.index.pr_files[n] & info) - self.top100]
            idx = [np.sort([self.pos_of[n] for n in ns]).astype(int) for ns in (pos, nt)]
            self.queries[q["split"]].append(Query(q["issue"], q["fix"], m, *idx))
        self.train_mask = np.array([t <= c.TRAIN_END for t in self.index.times])
        cnt: dict[str, int] = {}
        for p in prs:
            if p["merged"] and p["base"] == "master" and p["merged_at"] <= c.TRAIN_END:
                for fl in set(p["files"]) - excl:
                    cnt[fl] = cnt.get(fl, 0) + 1
        self.popularity = np.array([sum(cnt.get(f, 0) for f in self.index.pr_files[n] - excl) for n in self.nums], float)  # fmt: skip
        ptxt = qk_kaggle.build_texts(REPO, "prs", data_dir).set_index("number")["text"]
        self.pr_text = [ptxt.get(int(n), "") for n in self.nums]
        self.n_missing_text = int(sum(int(n) not in ptxt.index for n in self.nums))
        itxt = qk_kaggle.build_texts(REPO, "issues", data_dir).set_index("number")["text"]
        self.issue_text = itxt.to_dict()
        emb = data_dir / "embeddings" / slug
        self.pr_emb = self._load(emb, "prs", self.nums)
        self.issue_emb = {"dense_sym": self._load(emb, "issues", None)}
        if query_emb is not None and query_emb.exists():
            self.issue_emb["dense_query"] = self._load(query_emb, "issues", None)

    @staticmethod
    def _load(folder: Path, kind: str, nums: np.ndarray | None) -> Any:
        stem = f"{qk_kaggle.slug_of(REPO)}__{kind}"
        e = np.load(folder / f"{stem}_emb.npy").astype(np.float32)
        ids = pd.read_parquet(folder / f"{stem}_ids.parquet")["number"].tolist()
        row = {int(n): i for i, n in enumerate(ids)}
        if nums is not None:
            return e[[row[int(n)] for n in nums]]
        return {n: e[i] for n, i in row.items()}

    def scorers(self, split: str, hybrid_dense: str = "") -> dict[str, Callable[..., np.ndarray]]:
        """Name -> f(query_row, query) -> chronological scores (first q.m entries)."""
        qs = self.queries[split]
        qtext = [self.issue_text[q.issue] for q in qs]
        fit = [t for t, ok in zip(self.pr_text, self.train_mask, strict=True) if ok]
        bm = bm25_scores(fit, self.pr_text, qtext)
        tf = TfidfVectorizer().fit(fit)
        tfm = (tf.transform(qtext) @ tf.transform(self.pr_text).T).toarray()
        rng = np.random.default_rng(SEED)
        out: dict[str, Callable[[int, Query], np.ndarray]] = {
            "random_sim": lambda i, q: rng.random(q.m),
            "recent": lambda i, q: np.arange(q.m, dtype=float),
            "popularity": lambda i, q: self.popularity[: q.m],
            "bm25": lambda i, q: bm[i, : q.m],
            "tfidf": lambda i, q: tfm[i, : q.m],
        }  # fmt: skip
        dense = {n: np.stack([e[q.issue] for q in qs]) @ self.pr_emb.T for n, e in self.issue_emb.items()}  # fmt: skip
        for name, mat in dense.items():
            out[name] = lambda i, q, mat=mat: mat[i, : q.m]
        if hybrid_dense:
            out["hybrid"] = lambda i, q: rrf(bm[i, : q.m], dense[hybrid_dense][i, : q.m])
        return out


def split_summary(world: World, split: str, name: str, fn: Callable | None, boots: dict) -> dict:
    """Metrics (all positives / non-trivial slice) of one scorer; keeps per-query arrays in boots."""
    t0, qs = time.time(), world.queries[split]
    if name == "random":
        m = np.stack([[random_expectation(q.m, len(q.pos)), random_expectation(q.m, len(q.nt))] for q in qs])  # fmt: skip
    else:
        m = np.stack([query_metrics(fn(i, q), q) for i, q in enumerate(qs)])
    a, nt = m[:, 0], m[:, 1]
    keep = ~np.isnan(nt[:, 0])
    cl = np.array([q.fix for q in qs])
    if "all" not in boots:  # same resamples for every scorer of a split (paired differences)
        boots |= {"all": Boot(cl), "nt": Boot(cl[keep])}
    boots[f"M_{name}"] = (a, nt, keep)
    return {
        "all": summarise(a, boots["all"], name == "random"),
        "non_trivial": summarise(nt[keep], boots["nt"], name == "random"),
        "n_dropped_from_non_trivial_slice": int((~keep).sum()), "seconds": round(time.time() - t0, 2),
    }  # fmt: skip


def better(res: dict[str, dict], names: tuple[str, ...]) -> str:
    """Best by validation nDCG@10, tie hit@10 (names not present are ignored)."""
    key = lambda n: (res[n]["all"]["ndcg@10"]["mean"], res[n]["all"]["hit@10"]["mean"])  # noqa: E731
    return max((n for n in names if n in res), key=key)


def run_validation(world: World) -> dict[str, Any]:
    boots: dict = {}
    fns = world.scorers("validation")
    res = {n: split_summary(world, "validation", n, fns.get(n), boots) for n in ("random", *fns)}
    dense = better(res, ("dense_sym", "dense_query"))
    fns = world.scorers("validation", dense)
    res["hybrid"] = split_summary(world, "validation", "hybrid", fns["hybrid"], boots)
    return {
        "scorers": res, "dense_variant_for_hybrid": dense,
        "selected": better(res, NON_BASELINES), "strongest_baseline": better(res, BASELINES),
        "n_answerable": len(world.queries["validation"]),
    }  # fmt: skip


def run_test(world: World, val: dict[str, Any], done: set[str], final: bool) -> dict[str, Any]:
    if not final:
        raise TestGuardError("test metrics require --final-test")
    ids = list(dict.fromkeys([*BASELINES, val["selected"], val["strongest_baseline"]]))
    for mid in ids:
        if mid in done:
            raise TestGuardError(f"test already evaluated for {mid} in this report")
    fns = world.scorers("test", val["dense_variant_for_hybrid"])
    boots: dict = {}
    res = {n: split_summary(world, "test", n, fns.get(n), boots) for n in ids}
    a_s, nt_s, keep_s = boots[f"M_{val['selected']}"]
    a_b, nt_b, keep_b = boots[f"M_{val['strongest_baseline']}"]
    diffs = {
        "all": paired(a_s, a_b, boots["all"]),
        "non_trivial": paired(nt_s[keep_s], nt_b[keep_b], boots["nt"]),
    }
    win = {s: all(d[m]["diff"] > 0 and d[m]["excludes_zero"] for m in ("hit@10", "ndcg@10")) for s, d in diffs.items()}  # fmt: skip
    return {
        "models": res, "paired_selected_minus_strongest_baseline": diffs,
        "selected": val["selected"], "strongest_baseline": val["strongest_baseline"],
        "adr0068_criterion_met": win,
        "n_answerable": len(world.queries["test"]),
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("validation", "test"), required=True)
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=DATA)
    ap.add_argument("--query-emb", type=Path, default=DATA / "embeddings/kubernetes__kubernetes/issues_query_instr")  # fmt: skip
    ap.add_argument("--top100", type=Path, default=Path("reports/qp2_prior_pr_top100_informative_kubernetes__kubernetes_2026-10.json"))  # fmt: skip
    ap.add_argument("--report", type=Path, default=Path("reports/qp2_prior_pr_eval_kubernetes__kubernetes_2026-10.json"))  # fmt: skip
    a = ap.parse_args(argv)
    emb_dirs = [d for d in (a.data_dir / "embeddings/kubernetes__kubernetes", a.query_emb) if d.exists()]  # fmt: skip
    rep: dict[str, Any] = json.loads(a.report.read_text("utf-8")) if a.report.exists() else {}
    if a.stage == "test" and "validation" not in rep:
        raise RuntimeError("run --stage validation first (selection must precede test)")
    if a.stage == "validation" and "test" in rep:
        raise TestGuardError("test exists: validation/selection is frozen, refusing to redo it")
    t0, world = time.time(), World(a.data_dir, a.query_emb, a.top100)
    rep |= {
        "design": DESIGN, "generated_with": "scripts/qp2_prior_pr_eval.py",
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip(),  # noqa: S607
        "inputs": {"top100_sha256": sha256(a.top100), "split_manifest_sha256": sha256(a.data_dir / "splits/kubernetes__kubernetes/manifest.json"),
                   "dense_variants_available": sorted(world.issue_emb),
                   "embedding_sha256": {f"{d.name}/{f.name}": sha256(f) for d in emb_dirs for f in sorted(d.glob("*_emb.npy"))}, "pr_text_missing": world.n_missing_text,
                   "unanswerable_dropped_by_split": world.unanswerable},
    }  # fmt: skip
    if a.stage == "validation":
        rep["validation"] = run_validation(world)
    else:
        done = set(rep.get("test", {}).get("models", {}))
        rep["test"] = run_test(world, rep["validation"], done, a.final_test)
    rep[f"seconds_{a.stage}_stage_total"] = round(time.time() - t0, 1)
    a.report.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in rep[a.stage].items() if k not in ("scorers", "models")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
