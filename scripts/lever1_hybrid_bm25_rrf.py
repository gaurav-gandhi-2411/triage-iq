"""Lever 1 (spec.md / ADR-0031): hybrid BM25 + dense retrieval, fused with RRF.

Hypothesis: dense embeddings (BGE) systematically miss EXACT-TERM matches -- error codes
(ImagePullBackOff, CrashLoopBackOff), stack traces, API names, file paths, CLI flags -- which
is exactly what GitHub issue text is made of. BM25 catches lexical overlap dense retrieval
blurs away.

Method:
  - BM25 (rank_bm25.BM25Okapi) built over the SAME corpus text and SAME issue-number set as
    the live dense index (`detector.texts` / `detector.issue_numbers` after `.load()` --
    guarantees the two systems see an identical document universe, not a re-derived one).
  - Candidate pool: top-100 from each system (dense via `detector.retrieve`, BM25 via
    `get_scores` + argsort), fused with Reciprocal Rank Fusion, k=60 (rank_bm25 / RRF
    standard default). UNTUNED: the only pairs available (277 k8s / 292 vscode) are already
    the full powered eval set; carving out a separate tuning slice would either shrink the
    eval set below its current power or require re-tuning on the test pairs themselves (the
    leak the spec explicitly rules out). Stated per spec.md's fallback: "if there's no clean
    tuning slice, use RRF's standard k=60 untuned and say so."
  - Weighted score-fusion (min-max normalized, 0.5/0.5) also computed and reported as a
    secondary comparison; RRF is primary per spec.md.
  - Dense-only hit vectors are recomputed in the SAME loop (not reused from the separately
    committed baseline reports) so the paired bootstrap is guaranteed to compare identically
    ordered, identically selected pairs.
  - Paired bootstrap CI (scripts/_retrieval_eval_common.py::paired_bootstrap_ci, verbatim
    method from scripts/w3_t5_eval.py, ADR-0027's primary/corrected method) on the R@5 delta.
  - Diagnostic: pairs where hybrid hits @5 and dense-only doesn't -- reports the shared
    high-IDF terms between query and target text (the BM25-recovered signal dense missed).

Reads:
  data/gold_related_v2.parquet
  data/models/dup_index_kubernetes_kubernetes_bge/
  data/models/dup_index_microsoft_vscode_bge/

Output: reports/lever1_hybrid_bm25_rrf.json
Reproduce: python scripts/lever1_hybrid_bm25_rrf.py
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _retrieval_eval_common import (  # noqa: E402
    K_VALUES,
    N_BOOTSTRAP,
    SEED,
    paired_bootstrap_ci,
    query_text,
    select_live_product_pairs,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

GOLD_PATH = Path("data/gold_related_v2.parquet")
OUTPUT_PATH = Path("reports/lever1_hybrid_bm25_rrf.json")

RRF_K = 60  # standard default, untuned -- see module docstring
CANDIDATE_POOL = 100
K_MAX = max(K_VALUES)
TOKEN_RE = re.compile(r"[a-z0-9]+")

REPOS = [
    {"repo": "kubernetes_kubernetes", "index_dir": "data/models/dup_index_kubernetes_kubernetes_bge"},
    {"repo": "microsoft_vscode", "index_dir": "data/models/dup_index_microsoft_vscode_bge"},
]


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def normalize(d: dict[int, float]) -> dict[int, float]:
    if not d:
        return {}
    vals = list(d.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return dict.fromkeys(d, 0.5)
    return {k: (v - lo) / (hi - lo) for k, v in d.items()}


def eval_repo(repo: str, index_dir: str, gold: pd.DataFrame) -> dict:
    detector = SimilarIssueRetriever.load(index_dir)
    live_numbers = {int(n) for n in detector.issue_numbers}
    number_to_idx = {int(n): i for i, n in enumerate(detector.issue_numbers)}
    log.info("[%s] loaded live index: %d records", repo, len(live_numbers))

    t0 = time.perf_counter()
    tokenized_corpus = [tokenize(t) for t in detector.texts]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_build_s = time.perf_counter() - t0
    log.info("[%s] BM25 index built over %d docs in %.2fs", repo, len(tokenized_corpus), bm25_build_s)

    pairs = select_live_product_pairs(gold, repo, live_numbers)
    log.info("[%s] %d in-range product-task pairs", repo, len(pairs))

    dense_hits, rrf_hits, weighted_hits = [], [], []
    dense_lat, bm25_lat, fuse_lat = [], [], []
    diagnostics = []

    for row in pairs.itertuples():
        qtext = query_text(pd.Series({"query_title": row.query_title, "query_body": row.query_body}))
        query_num = int(row.query_number)
        target_num = int(row.original_number)

        t0 = time.perf_counter()
        dense_results = detector.retrieve(qtext, k=CANDIDATE_POOL, exclude_number=query_num)
        dense_lat.append(time.perf_counter() - t0)
        dense_ranked = [r["number"] for r in dense_results]
        dense_score_map = {r["number"]: r["score"] for r in dense_results}

        t0 = time.perf_counter()
        qtok = tokenize(qtext)
        scores = bm25.get_scores(qtok)
        order = np.argsort(-scores)
        bm25_ranked: list[int] = []
        bm25_score_map: dict[int, float] = {}
        for idx in order:
            num = int(detector.issue_numbers[idx])
            if num == query_num:
                continue
            bm25_ranked.append(num)
            bm25_score_map[num] = float(scores[idx])
            if len(bm25_ranked) >= CANDIDATE_POOL:
                break
        bm25_lat.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        rrf_scores: dict[int, float] = {}
        for rank, num in enumerate(dense_ranked, start=1):
            rrf_scores[num] = rrf_scores.get(num, 0.0) + 1.0 / (RRF_K + rank)
        for rank, num in enumerate(bm25_ranked, start=1):
            rrf_scores[num] = rrf_scores.get(num, 0.0) + 1.0 / (RRF_K + rank)
        rrf_ranked = [n for n, _ in sorted(rrf_scores.items(), key=lambda kv: -kv[1])]

        dn = normalize(dense_score_map)
        bn = normalize(bm25_score_map)
        weighted_scores = {
            n: 0.5 * dn.get(n, 0.0) + 0.5 * bn.get(n, 0.0) for n in set(dn) | set(bn)
        }
        weighted_ranked = [n for n, _ in sorted(weighted_scores.items(), key=lambda kv: -kv[1])]
        fuse_lat.append(time.perf_counter() - t0)

        dense_hit = [n == target_num for n in dense_ranked[:K_MAX]]
        rrf_hit = [n == target_num for n in rrf_ranked[:K_MAX]]
        weighted_hit = [n == target_num for n in weighted_ranked[:K_MAX]]
        dense_hits.append(dense_hit)
        rrf_hits.append(rrf_hit)
        weighted_hits.append(weighted_hit)

        if any(rrf_hit[:5]) and not any(dense_hit[:5]):
            target_idx = number_to_idx.get(target_num)
            target_text = detector.texts[target_idx] if target_idx is not None else ""
            shared = set(tokenize(target_text)) & set(qtok)
            shared_scored = sorted(shared, key=lambda t: -bm25.idf.get(t, 0.0))[:5]
            diagnostics.append(
                {
                    "query_number": query_num,
                    "target_number": target_num,
                    "query_title": row.query_title,
                    "target_title": row.original_title,
                    "dense_rank_of_target": (
                        dense_ranked.index(target_num) + 1 if target_num in dense_ranked else None
                    ),
                    "bm25_rank_of_target": (
                        bm25_ranked.index(target_num) + 1 if target_num in bm25_ranked else None
                    ),
                    "shared_high_idf_terms": shared_scored,
                }
            )

    def recall_table(hit_lists: list[list[bool]]) -> dict[str, float]:
        return {f"recall_at_{k}": float(np.mean([any(h[:k]) for h in hit_lists])) for k in K_VALUES}

    dense_r5 = np.array([float(any(h[:5])) for h in dense_hits])
    rrf_r5 = np.array([float(any(h[:5])) for h in rrf_hits])
    weighted_r5 = np.array([float(any(h[:5])) for h in weighted_hits])

    rrf_lo, rrf_hi, rrf_delta = paired_bootstrap_ci(dense_r5, rrf_r5)
    w_lo, w_hi, w_delta = paired_bootstrap_ci(dense_r5, weighted_r5)

    result = {
        "repo": repo,
        "n_pairs": len(pairs),
        "dense_only": recall_table(dense_hits),
        "hybrid_rrf": recall_table(rrf_hits),
        "hybrid_weighted": recall_table(weighted_hits),
        "rrf_vs_dense_r5_delta_pp": round(rrf_delta * 100, 2),
        "rrf_vs_dense_r5_ci95_pp": [round(rrf_lo * 100, 2), round(rrf_hi * 100, 2)],
        "rrf_ships": rrf_lo > 0,
        "weighted_vs_dense_r5_delta_pp": round(w_delta * 100, 2),
        "weighted_vs_dense_r5_ci95_pp": [round(w_lo * 100, 2), round(w_hi * 100, 2)],
        "weighted_ships": w_lo > 0,
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": SEED, "method": "paired percentile"},
        "rrf_k": RRF_K,
        "candidate_pool": CANDIDATE_POOL,
        "tuning_provenance": (
            "RRF k=60 is the untuned standard default. No held-out tuning slice was carved "
            "out of the product-task pairs -- doing so would either shrink this repo's "
            "already-thin powered eval set or require tuning on the test pairs themselves "
            "(the leak spec.md explicitly rules out). Nothing was tuned on the test pairs."
        ),
        "latency_s": {
            "dense_retrieve_mean": float(np.mean(dense_lat)),
            "bm25_score_mean": float(np.mean(bm25_lat)),
            "fuse_mean": float(np.mean(fuse_lat)),
            "bm25_index_build_total": bm25_build_s,
            "added_per_query_mean": float(np.mean(bm25_lat) + np.mean(fuse_lat)),
        },
        "diagnostic_bm25_recovers_dense_misses": {
            "n_cases": len(diagnostics),
            "n_cases_pct_of_pairs": round(100 * len(diagnostics) / len(pairs), 2) if pairs.shape[0] else 0.0,
            "examples": diagnostics[:15],
        },
    }
    return result


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    results = [eval_repo(r["repo"], r["index_dir"], gold) for r in REPOS]

    for r in results:
        log.info(
            "[%s] dense R@5=%.4f  RRF R@5=%.4f  delta=%.2fpp  CI=%s  ships=%s",
            r["repo"],
            r["dense_only"]["recall_at_5"],
            r["hybrid_rrf"]["recall_at_5"],
            r["rrf_vs_dense_r5_delta_pp"],
            r["rrf_vs_dense_r5_ci95_pp"],
            r["rrf_ships"],
        )
        log.info(
            "[%s] BM25 recovers %d/%d (%.1f%%) of pairs dense misses @5",
            r["repo"],
            r["diagnostic_bm25_recovers_dense_misses"]["n_cases"],
            r["n_pairs"],
            r["diagnostic_bm25_recovers_dense_misses"]["n_cases_pct_of_pairs"],
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"results": results}, indent=2))
    log.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
