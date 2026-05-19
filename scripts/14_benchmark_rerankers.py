"""W1.3 — Reranker candidate benchmark.

Evaluates three cross-encoder models on the existing gold duplicate pairs:
  - BAAI/bge-reranker-v2-m3       (~568M)
  - jinaai/jina-reranker-v2-base-multilingual (~278M)
  - mixedbread-ai/mxbai-rerank-base-v1        (~184M)

For each model and each repo:
  1. Retrieve top-RETRIEVAL_K candidates via BGE+FAISS (same as production).
  2. Rerank all RETRIEVAL_K with the cross-encoder.
  3. Compute Recall@1/5/10, MRR on the reranked list.
  4. Measure p50/p95 end-to-end latency per query.

Baseline (BGE alone, k=5) is included for comparison.
Output: reports/reranker_benchmark.json
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
RETRIEVAL_K = 50   # FAISS candidates to rerank
FINAL_K = 5        # final results returned to caller

RERANKER_CANDIDATES = [
    {
        "key": "mxbai-rerank-base-v1",
        "model_id": "mixedbread-ai/mxbai-rerank-base-v1",
        "size_mb_approx": 184,
        "trust_remote_code": False,
    },
    {
        "key": "jina-reranker-v2-base-multilingual",
        "model_id": "jinaai/jina-reranker-v2-base-multilingual",
        "size_mb_approx": 278,
        "trust_remote_code": True,
    },
    {
        "key": "bge-reranker-v2-m3",
        "model_id": "BAAI/bge-reranker-v2-m3",
        "size_mb_approx": 568,
        "trust_remote_code": False,
    },
]

REPOS = [
    ("microsoft_vscode", "dup_index_microsoft_vscode_bge"),
    ("kubernetes_kubernetes", "dup_index_kubernetes_kubernetes_bge"),
]


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------

def _mrr_and_recalls(ranked_hits: list[int], true_orig: int, at_ks: list[int]) -> dict:
    """Given a ranked list of issue numbers, return MRR and Recall@k."""
    try:
        rank = ranked_hits.index(true_orig) + 1  # 1-indexed
        mrr = 1.0 / rank
    except ValueError:
        mrr = 0.0
        rank = None
    return {
        "mrr": mrr,
        **{f"recall_at_{k}": int(rank is not None and rank <= k) for k in at_ks},
    }


# ---------------------------------------------------------------------------
# Baseline (BGE alone)
# ---------------------------------------------------------------------------

def eval_baseline(repo_key: str, index_dir_name: str, gold: pd.DataFrame) -> dict:
    from triage_iq.models.duplicates import DuplicateDetector  # noqa: PLC0415

    log.info("[baseline] Loading BGE index for %s …", repo_key)
    det = DuplicateDetector.load(str(DATA_DIR / "models" / index_dir_name))

    repo_gold = gold[gold["repo"] == repo_key].copy()
    mrrs, r1, r5, r10 = [], [], [], []
    latencies: list[float] = []

    for _, row in repo_gold.iterrows():
        query_text = f"{row['query_title']}. {row['query_body'][:512]}"
        t0 = time.perf_counter()
        # Retrieve at least 20 so R@10 is meaningful; report R@5 from top slice
        hits = det._faiss_retrieve(query_text, k=max(20, FINAL_K), exclude_number=int(row["query_number"]))
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [h["number"] for h in hits]
        m = _mrr_and_recalls(ranked, int(row["original_number"]), [1, 5, 10])
        mrrs.append(m["mrr"]); r1.append(m["recall_at_1"]); r5.append(m["recall_at_5"]); r10.append(m["recall_at_10"])

    return {
        "repo": repo_key, "model": "baseline_bge_k5",
        "mrr": float(np.mean(mrrs)),
        "recall_at_1": float(np.mean(r1)),
        "recall_at_5": float(np.mean(r5)),
        "recall_at_10": float(np.mean(r10)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "n_queries": len(repo_gold),
    }


# ---------------------------------------------------------------------------
# Reranker eval
# ---------------------------------------------------------------------------

def eval_reranker(
    repo_key: str,
    index_dir_name: str,
    gold: pd.DataFrame,
    candidate: dict,
) -> dict:
    from sentence_transformers import CrossEncoder  # noqa: PLC0415
    from triage_iq.models.duplicates import DuplicateDetector  # noqa: PLC0415

    model_id = candidate["model_id"]
    key = candidate["key"]

    log.info("[%s/%s] Loading BGE index …", repo_key, key)
    det = DuplicateDetector.load(str(DATA_DIR / "models" / index_dir_name))

    log.info("[%s/%s] Loading CrossEncoder %s …", repo_key, key, model_id)
    reranker = CrossEncoder(
        model_id,
        max_length=512,
        trust_remote_code=candidate["trust_remote_code"],
    )

    repo_gold = gold[gold["repo"] == repo_key].copy()
    mrrs, r1, r5, r10 = [], [], [], []
    latencies: list[float] = []

    for i, (_, row) in enumerate(repo_gold.iterrows()):
        if i % 100 == 0:
            log.info("[%s/%s] %d/%d …", repo_key, key, i, len(repo_gold))

        query_text = f"{row['query_title']}. {row['query_body'][:512]}"
        t0 = time.perf_counter()

        # Stage 1: BGE retrieval of top-RETRIEVAL_K
        hits = det.retrieve(query_text, k=RETRIEVAL_K, exclude_number=int(row["query_number"]))
        candidate_texts = [h["text"] for h in hits]
        candidate_numbers = [h["number"] for h in hits]

        # Stage 2: cross-encoder reranking
        pairs = [(query_text, ct) for ct in candidate_texts]
        ce_scores = reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(ce_scores)[::-1]
        ranked = [candidate_numbers[i] for i in order]

        latencies.append((time.perf_counter() - t0) * 1000)

        m = _mrr_and_recalls(ranked, int(row["original_number"]), [1, 5, 10])
        mrrs.append(m["mrr"]); r1.append(m["recall_at_1"]); r5.append(m["recall_at_5"]); r10.append(m["recall_at_10"])

    return {
        "repo": repo_key,
        "model": key,
        "model_id": model_id,
        "size_mb_approx": candidate["size_mb_approx"],
        "retrieval_k": RETRIEVAL_K,
        "mrr": float(np.mean(mrrs)),
        "recall_at_1": float(np.mean(r1)),
        "recall_at_5": float(np.mean(r5)),
        "recall_at_10": float(np.mean(r10)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "n_queries": len(repo_gold),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    gold = pd.read_parquet(DATA_DIR / "gold_duplicates.parquet")
    log.info("Gold pairs: %d total (%d vscode, %d kubernetes)",
             len(gold),
             len(gold[gold["repo"] == "microsoft_vscode"]),
             len(gold[gold["repo"] == "kubernetes_kubernetes"]))

    results: list[dict] = []

    # Baseline
    for repo_key, index_dir in REPOS:
        r = eval_baseline(repo_key, index_dir, gold)
        results.append(r)
        log.info("[baseline/%s] MRR=%.3f R@5=%.3f R@10=%.3f p50=%.1fms p95=%.1fms",
                 repo_key, r["mrr"], r["recall_at_5"], r["recall_at_10"],
                 r["latency_p50_ms"], r["latency_p95_ms"])

    # Rerankers
    for cand in RERANKER_CANDIDATES:
        for repo_key, index_dir in REPOS:
            try:
                r = eval_reranker(repo_key, index_dir, gold, cand)
                results.append(r)
                log.info("[%s/%s] MRR=%.3f R@5=%.3f R@10=%.3f p50=%.1fms p95=%.1fms",
                         cand["key"], repo_key, r["mrr"], r["recall_at_5"], r["recall_at_10"],
                         r["latency_p50_ms"], r["latency_p95_ms"])
            except Exception as exc:
                log.error("[%s/%s] FAILED: %s", cand["key"], repo_key, exc)
                results.append({
                    "repo": repo_key,
                    "model": cand["key"],
                    "error": str(exc),
                })

    out = REPORTS_DIR / "reranker_benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Saved to %s", out)

    # Summary table
    print("\n=== BENCHMARK SUMMARY ===")
    print(f"{'model':<40} {'repo':<25} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'p50ms':>7} {'p95ms':>7}")
    for r in results:
        if "error" in r:
            print(f"{'ERROR: '+r['model']:<40} {r['repo']:<25}")
            continue
        print(f"{r['model']:<40} {r['repo']:<25} {r['recall_at_5']:>6.3f} {r['recall_at_10']:>6.3f} {r['mrr']:>6.3f} {r['latency_p50_ms']:>7.1f} {r['latency_p95_ms']:>7.1f}")


if __name__ == "__main__":
    main()
