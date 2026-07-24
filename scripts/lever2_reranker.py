"""Lever 2 (spec.md / ADR-0031): pretrained cross-encoder reranker on the PRODUCT task.

The ADR-0006 retry. ADR-0006 screened 7 off-the-shelf cross-encoders across two slates and
rejected all of them -- but against the PROXY metric (gold_related.parquet v1, ~74%
PR-reference pairs for vscode / ~8% for k8s), now known (ADR-0030) to be the wrong task.
The strongest candidate from that screening was BAAI/bge-reranker-v2-m3 (Apache-2.0, the
only one that improved either repo at all, later killed on a robustness re-check at n=300
k8s -- CI[-0.037,+0.053], noise). The other six candidates (mxbai, jina, bge-reranker-base,
quora-distilroberta, quora-roberta, stsb-distilroberta) failed both repos by wide margins in
both slates; re-running them on the honest product-task metric is not a productive use of a
second screening pass, so this retry uses bge-reranker-v2-m3 alone -- the one candidate whose
prior failure could plausibly be metric-attributable rather than model-attributable.

Method:
  - First stage: dense-only retrieval (Lever 1's hybrid fusion was rejected on both repos --
    see reports/lever1_hybrid_bm25_rrf.json -- so "best first-stage" per spec.md is dense).
  - Rerank top-30 dense candidates (spec.md's stated k=20-50 range, midpoint) with
    CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512).
  - RECALL/CI: computed on the full pair set using whatever device is available (GPU in this
    dev environment) -- recall/CI correctness doesn't depend on device, and GPU makes the
    full ~570-query x 30-candidate run fast.
  - LATENCY: measured SEPARATELY on a CPU-forced subsample (LATENCY_N queries, fixed seed).
    The production target is CPU-only inference (scripts/08_build_similar_issue_index.py),
    so the number that matters for the ship/reject latency call has to reflect the
    deployment hardware profile, not the GPU used to compute recall quickly.
  - No training -- zero-leakage reasoning (ADR-0030) holds unchanged.
  - Paired bootstrap CI (scripts/_retrieval_eval_common.py, ADR-0027's method) on the R@5
    delta vs the dense-only baseline, recomputed in the same loop for exact pairing.

CORRECTED (see the ADR superseding ADR-0031/0033/0034): originally read gold_related_v2.parquet
via select_live_product_pairs() against the stale served dup_index_*, with truncated/effectively
title-only queries. Now reads D1's canonical, hand-verified, disjoint eval sets against the
full-corpus d1_full_corpus_index_*, with untruncated title+body queries matching production.

Reads:
  reports/d1_eval_set_k8s_related.json
  reports/d1_eval_set_vscode_duplicate.json
  data/models/d1_full_corpus_index_kubernetes_kubernetes_bge/
  data/models/d1_full_corpus_index_microsoft_vscode_bge/

Output: reports/lever2_reranker.json
Reproduce: python scripts/lever2_reranker.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _retrieval_eval_common import (  # noqa: E402
    K_VALUES,
    N_BOOTSTRAP,
    SEED,
    load_d1_eval_pairs,
    paired_bootstrap_ci,
    query_text,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

OUTPUT_PATH = Path("reports/lever2_reranker.json")

MODEL_ID = "BAAI/bge-reranker-v2-m3"
RERANK_POOL = 30  # spec.md's stated k=20-50 range, midpoint
K_MAX = max(K_VALUES)
LATENCY_N = 30  # CPU latency subsample size per repo


def load_pairs(repo: str, index_dir: str) -> tuple[SimilarIssueRetriever, pd.DataFrame]:
    detector = SimilarIssueRetriever.load(index_dir)
    pairs = load_d1_eval_pairs(repo)
    return detector, pairs


def rerank_one(detector, reranker, row) -> tuple[list[int], list[int], float, float]:
    qtext = query_text(pd.Series({"query_title": row.query_title, "query_body": row.query_body}))
    query_num = int(row.query_number)

    t0 = time.perf_counter()
    dense_results = detector.retrieve(qtext, k=RERANK_POOL, exclude_number=query_num)
    dense_lat = time.perf_counter() - t0
    dense_ranked = [r["number"] for r in dense_results]

    if dense_results:
        t0 = time.perf_counter()
        ce_pairs = [(qtext, r["text"]) for r in dense_results]
        ce_scores = reranker.predict(ce_pairs, show_progress_bar=False)
        ce_lat = time.perf_counter() - t0
        order = np.argsort(-np.asarray(ce_scores))
        reranked_ranked = [dense_results[j]["number"] for j in order]
    else:
        ce_lat = 0.0
        reranked_ranked = []

    return dense_ranked, reranked_ranked, dense_lat, ce_lat


def eval_recall(repo: str, detector, pairs: pd.DataFrame, reranker) -> dict:
    dense_hits, reranked_hits = [], []

    for i, row in enumerate(pairs.itertuples()):
        if i % 50 == 0:
            log.info("  [%s] recall %d/%d", repo, i, len(pairs))
        target_num = int(row.original_number)
        dense_ranked, reranked_ranked, _, _ = rerank_one(detector, reranker, row)
        dense_hits.append([n == target_num for n in dense_ranked[:K_MAX]])
        reranked_hits.append([n == target_num for n in reranked_ranked[:K_MAX]])

    def recall_table(hit_lists: list[list[bool]]) -> dict[str, float]:
        return {f"recall_at_{k}": float(np.mean([any(h[:k]) for h in hit_lists])) for k in K_VALUES}

    dense_r5 = np.array([float(any(h[:5])) for h in dense_hits])
    rerank_r5 = np.array([float(any(h[:5])) for h in reranked_hits])
    ci_lo, ci_hi, delta = paired_bootstrap_ci(dense_r5, rerank_r5)

    return {
        "repo": repo,
        "n_pairs": len(pairs),
        "model_id": MODEL_ID,
        "rerank_pool": RERANK_POOL,
        "dense_only": recall_table(dense_hits),
        "reranked": recall_table(reranked_hits),
        "r5_delta_pp": round(delta * 100, 2),
        "r5_ci95_pp": [round(ci_lo * 100, 2), round(ci_hi * 100, 2)],
        "ships": ci_lo > 0,
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": SEED, "method": "paired percentile"},
    }


def eval_latency_cpu(repo: str, detector, pairs: pd.DataFrame, cpu_reranker) -> dict:
    sample = pairs.sample(n=min(LATENCY_N, len(pairs)), random_state=SEED)
    dense_lat, ce_lat = [], []
    for row in sample.itertuples():
        _, _, d_lat, c_lat = rerank_one(detector, cpu_reranker, row)
        dense_lat.append(d_lat)
        ce_lat.append(c_lat)

    dense_mean = float(np.mean(dense_lat))
    ce_mean = float(np.mean(ce_lat))
    return {
        "repo": repo,
        "n_sample": len(sample),
        "seed": SEED,
        "device": "cpu",
        "dense_retrieve_mean_s": dense_mean,
        "ce_rerank_mean_s": ce_mean,
        "ce_rerank_p50_s": float(np.percentile(ce_lat, 50)),
        "ce_rerank_p95_s": float(np.percentile(ce_lat, 95)),
        "total_dense_only_mean_s": dense_mean,
        "total_with_rerank_mean_s": dense_mean + ce_mean,
        "added_latency_pct": round(100 * ce_mean / dense_mean, 1) if dense_mean else None,
    }


def main() -> None:
    from sentence_transformers import CrossEncoder

    log.info("Loading CrossEncoder for recall pass (default device): %s", MODEL_ID)
    gpu_reranker = CrossEncoder(MODEL_ID, max_length=512, trust_remote_code=False)
    log.info("Recall-pass device: %s", gpu_reranker.model.device)

    log.info("Loading second CrossEncoder instance forced to CPU for latency subsample")
    cpu_reranker = CrossEncoder(MODEL_ID, max_length=512, trust_remote_code=False, device="cpu")

    repos = [
        {"repo": "kubernetes_kubernetes", "index_dir": "data/models/d1_full_corpus_index_kubernetes_kubernetes_bge"},
        {"repo": "microsoft_vscode", "index_dir": "data/models/d1_full_corpus_index_microsoft_vscode_bge"},
    ]

    results = []
    for r in repos:
        detector, pairs = load_pairs(r["repo"], r["index_dir"])
        log.info("[%s] %d D1 canonical eval pairs", r["repo"], len(pairs))

        recall_result = eval_recall(r["repo"], detector, pairs, gpu_reranker)
        latency_result = eval_latency_cpu(r["repo"], detector, pairs, cpu_reranker)
        recall_result["latency_cpu_subsample"] = latency_result
        results.append(recall_result)

        log.info(
            "[%s] dense R@5=%.4f  reranked R@5=%.4f  delta=%.2fpp  CI=%s  ships=%s",
            r["repo"],
            recall_result["dense_only"]["recall_at_5"],
            recall_result["reranked"]["recall_at_5"],
            recall_result["r5_delta_pp"],
            recall_result["r5_ci95_pp"],
            recall_result["ships"],
        )
        log.info(
            "[%s] CPU latency (n=%d): dense=%.3fs  +rerank=%.3fs  added=+%.1f%%",
            r["repo"],
            latency_result["n_sample"],
            latency_result["total_dense_only_mean_s"],
            latency_result["total_with_rerank_mean_s"],
            latency_result["added_latency_pct"] or 0.0,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"results": results}, indent=2))
    log.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
