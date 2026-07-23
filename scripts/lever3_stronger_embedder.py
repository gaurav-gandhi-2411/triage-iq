"""Lever 3 (spec.md / ADR-0031): stronger pretrained base embedder, no fine-tuning.

Candidate: BAAI/bge-large-en-v1.5 -- the larger sibling of the currently-deployed
BAAI/bge-base-en-v1.5 (same training family/methodology, so this isolates a single
variable -- model capacity -- rather than also changing training data/objective the way
a cross-family swap would). 335M params vs base's 109M, 1024-dim embeddings vs 768-dim.
Pretrained only, no fine-tuning -- zero-leakage reasoning (ADR-0030) holds unchanged.
Zero-cost / local: both models are free, Apache-2.0-licensed, CPU-runnable.

Method:
  - Re-embed the EXACT SAME corpus text used by the live index (`detector.texts` /
    `detector.issue_numbers` copied verbatim from the loaded baseline retriever, not
    re-derived from data/processed/*.parquet -- guarantees the only variable that changes
    is the embedding model, not the corpus).
  - Build a fresh FAISS IndexFlatIP over the new embeddings (same index type, same
    normalize-for-cosine convention as SimilarIssueRetriever.build_index).
  - Product-task R@5 (+1/10/20) vs the dense-only baseline, paired bootstrap CI
    (scripts/_retrieval_eval_common.py, ADR-0027's method), recomputed in the same loop.
  - Cost: embedding dimension, index size (float32 bytes), corpus embedding build time,
    and per-query CPU encode latency (subsample, same rationale as Lever 2 -- the
    production target is CPU-only inference).

CORRECTED (see the ADR superseding ADR-0031/0033/0034): originally read gold_related_v2.parquet
via select_live_product_pairs() against the stale served dup_index_*, with truncated/effectively
title-only queries. Now reads D1's canonical, hand-verified, disjoint eval sets against the
full-corpus d1_full_corpus_index_*, with untruncated title+body queries matching production.

Reads:
  reports/d1_eval_set_k8s_related.json
  reports/d1_eval_set_vscode_duplicate.json
  data/models/d1_full_corpus_index_kubernetes_kubernetes_bge/
  data/models/d1_full_corpus_index_microsoft_vscode_bge/

Output: reports/lever3_stronger_embedder.json
Reproduce: python scripts/lever3_stronger_embedder.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import faiss
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

OUTPUT_PATH = Path("reports/lever3_stronger_embedder.json")

CANDIDATE_MODEL_ID = "BAAI/bge-large-en-v1.5"
K_MAX = max(K_VALUES)
LATENCY_N = 30

REPOS = [
    {"repo": "kubernetes_kubernetes", "index_dir": "data/models/d1_full_corpus_index_kubernetes_kubernetes_bge"},
    {"repo": "microsoft_vscode", "index_dir": "data/models/d1_full_corpus_index_microsoft_vscode_bge"},
]


def build_candidate_retriever(repo: str, baseline: SimilarIssueRetriever) -> tuple[SimilarIssueRetriever, dict]:
    candidate = SimilarIssueRetriever(repo=repo, model_key=CANDIDATE_MODEL_ID)
    candidate.texts = baseline.texts
    candidate.issue_numbers = baseline.issue_numbers

    t0 = time.perf_counter()
    embs = candidate.model.encode(
        candidate.texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    build_s = time.perf_counter() - t0

    dim = embs.shape[1]
    candidate.index = faiss.IndexFlatIP(dim)
    candidate.index.add(embs)

    cost = {
        "model_id": CANDIDATE_MODEL_ID,
        "embedding_dim": dim,
        "n_docs": len(candidate.texts),
        "index_bytes_float32": int(embs.nbytes),
        "index_mb_float32": round(embs.nbytes / (1024 * 1024), 1),
        "corpus_embed_build_s": build_s,
    }
    return candidate, cost


def eval_recall(repo: str, baseline: SimilarIssueRetriever, candidate: SimilarIssueRetriever, pairs: pd.DataFrame) -> dict:
    base_hits, cand_hits = [], []

    for i, row in enumerate(pairs.itertuples()):
        if i % 50 == 0:
            log.info("  [%s] recall %d/%d", repo, i, len(pairs))
        qtext = query_text(pd.Series({"query_title": row.query_title, "query_body": row.query_body}))
        query_num = int(row.query_number)
        target_num = int(row.original_number)

        base_results = baseline.retrieve(qtext, k=K_MAX, exclude_number=query_num)
        cand_results = candidate.retrieve(qtext, k=K_MAX, exclude_number=query_num)

        base_hits.append([r["number"] == target_num for r in base_results])
        cand_hits.append([r["number"] == target_num for r in cand_results])

    def recall_table(hit_lists: list[list[bool]]) -> dict[str, float]:
        return {f"recall_at_{k}": float(np.mean([any(h[:k]) for h in hit_lists])) for k in K_VALUES}

    base_r5 = np.array([float(any(h[:5])) for h in base_hits])
    cand_r5 = np.array([float(any(h[:5])) for h in cand_hits])
    ci_lo, ci_hi, delta = paired_bootstrap_ci(base_r5, cand_r5)

    return {
        "repo": repo,
        "n_pairs": len(pairs),
        "baseline_bge_base": recall_table(base_hits),
        "candidate_bge_large": recall_table(cand_hits),
        "r5_delta_pp": round(delta * 100, 2),
        "r5_ci95_pp": [round(ci_lo * 100, 2), round(ci_hi * 100, 2)],
        "ships": ci_lo > 0,
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": SEED, "method": "paired percentile"},
    }


def eval_latency_cpu(repo: str, baseline: SimilarIssueRetriever, candidate: SimilarIssueRetriever, pairs: pd.DataFrame) -> dict:
    sample = pairs.sample(n=min(LATENCY_N, len(pairs)), random_state=SEED)
    base_lat, cand_lat = [], []
    for row in sample.itertuples():
        qtext = query_text(pd.Series({"query_title": row.query_title, "query_body": row.query_body}))
        query_num = int(row.query_number)

        t0 = time.perf_counter()
        baseline.retrieve(qtext, k=K_MAX, exclude_number=query_num)
        base_lat.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        candidate.retrieve(qtext, k=K_MAX, exclude_number=query_num)
        cand_lat.append(time.perf_counter() - t0)

    base_mean, cand_mean = float(np.mean(base_lat)), float(np.mean(cand_lat))
    return {
        "n_sample": len(sample),
        "seed": SEED,
        "device": "cpu",
        "baseline_query_encode_mean_s": base_mean,
        "candidate_query_encode_mean_s": cand_mean,
        "added_latency_pct": round(100 * (cand_mean - base_mean) / base_mean, 1) if base_mean else None,
    }


def main() -> None:
    results = []

    for r in REPOS:
        baseline = SimilarIssueRetriever.load(r["index_dir"])
        pairs = load_d1_eval_pairs(r["repo"])
        log.info("[%s] %d D1 canonical eval pairs", r["repo"], len(pairs))

        candidate, cost = build_candidate_retriever(r["repo"], baseline)
        recall_result = eval_recall(r["repo"], baseline, candidate, pairs)
        latency_result = eval_latency_cpu(r["repo"], baseline, candidate, pairs)

        recall_result["cost"] = cost
        recall_result["latency_cpu_subsample"] = latency_result
        results.append(recall_result)

        log.info(
            "[%s] BGE-base R@5=%.4f  BGE-large R@5=%.4f  delta=%.2fpp  CI=%s  ships=%s",
            r["repo"],
            recall_result["baseline_bge_base"]["recall_at_5"],
            recall_result["candidate_bge_large"]["recall_at_5"],
            recall_result["r5_delta_pp"],
            recall_result["r5_ci95_pp"],
            recall_result["ships"],
        )
        log.info(
            "[%s] cost: dim %d (was 768), index %.1fMB, build %.1fs, CPU query latency +%.1f%%",
            r["repo"],
            cost["embedding_dim"],
            cost["index_mb_float32"],
            cost["corpus_embed_build_s"],
            latency_result["added_latency_pct"] or 0.0,
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps({"results": results}, indent=2))
    log.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
