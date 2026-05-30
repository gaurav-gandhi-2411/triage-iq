"""Phase 2 T3 — CPU-only rerank latency for BAAI/bge-reranker-v2-m3.

Measures ONLY ce.predict() time on CPU (no FAISS, no BGE encode).
Pre-fetches FAISS candidates offline so timing is pure rerank cost.

Protocol: 10 warmup queries, 100 timed queries, k8s repo, seed=42.
Reports p50/p95 for top-50 candidates; if p95>2.5s also reports top-25.

Hard stop rules:
  - p95 > 2.5s at top-50 AND top-25 → infeasible. Phase 2 STOP.
  - p95 ≤ 2.5s at top-25 → re-check R@5 at top-25 (calls robustness script result).

Output: reports/phase2_cpu_latency.json
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Force CPU before any torch/transformers import
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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

N_WARMUP = 10
N_TIME = 100
SEED = 42
REPO_KEY = "kubernetes_kubernetes"
INDEX_DIR = DATA_DIR / "models" / "dup_index_kubernetes_kubernetes_bge"
MODEL_ID = "BAAI/bge-reranker-v2-m3"
LATENCY_BUDGET_P95_MS = 2500.0


def sample_queries(gold: pd.DataFrame, n: int, seed: int) -> list[tuple[str, int]]:
    """Return (query_text, query_number) pairs for k8s, sampled deterministically."""
    repo_gold = gold[gold["repo"] == REPO_KEY].copy()
    rng = random.Random(seed)
    idxs = rng.sample(range(len(repo_gold)), min(n, len(repo_gold)))
    rows = repo_gold.iloc[idxs]
    return [
        (f"{row['query_title']}. {row['query_body'][:512]}", int(row["query_number"]))
        for _, row in rows.iterrows()
    ]


def prefetch_candidates(det, queries: list[tuple[str, int]], retrieval_k: int) -> list[tuple[str, list[dict]]]:
    log.info("Pre-fetching FAISS candidates (%d queries, k=%d) …", len(queries), retrieval_k)
    result = []
    for query_text, qnum in queries:
        hits = det.retrieve(query_text, k=retrieval_k, exclude_number=qnum)
        result.append((query_text, hits))
    log.info("Pre-fetch done: %d queries × %d candidates", len(result), retrieval_k)
    return result


def time_rerank(reranker, prefetched: list[tuple[str, list[dict]]], top_k: int) -> list[float]:
    latencies_ms = []
    for query_text, hits in prefetched:
        candidates = hits[:top_k]
        if not candidates:
            continue
        pairs = [(query_text, h["text"]) for h in candidates]
        t0 = time.perf_counter()
        reranker.predict(pairs, show_progress_bar=False)
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return latencies_ms


def measure_at_k(reranker, prefetched_50: list, k: int) -> dict:
    warmup = prefetched_50[:N_WARMUP]
    timed = prefetched_50[N_WARMUP: N_WARMUP + N_TIME]

    log.info("Warmup (%d queries, k=%d) …", len(warmup), k)
    time_rerank(reranker, warmup, k)

    log.info("Timing %d queries (k=%d) …", len(timed), k)
    latencies = time_rerank(reranker, timed, k)

    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    log.info("k=%d  p50=%.1fms  p95=%.1fms  (n=%d)", k, p50, p95, len(latencies))
    return {"candidate_k": k, "n_timed": len(latencies), "p50_ms": p50, "p95_ms": p95}


def main() -> None:
    from sentence_transformers import CrossEncoder
    from triage_iq.models.similar_issues import SimilarIssueRetriever

    log.info("CUDA_VISIBLE_DEVICES='%s' — CrossEncoder will run on CPU", os.environ.get("CUDA_VISIBLE_DEVICES", ""))

    gold = pd.read_parquet(DATA_DIR / "gold_related.parquet")
    all_queries = sample_queries(gold, N_WARMUP + N_TIME, SEED)
    log.info("Queries sampled: %d", len(all_queries))

    log.info("Loading FAISS index: %s", INDEX_DIR)
    det = SimilarIssueRetriever.load(str(INDEX_DIR))

    prefetched_50 = prefetch_candidates(det, all_queries, retrieval_k=50)

    log.info("Loading CrossEncoder on CPU: %s", MODEL_ID)
    reranker = CrossEncoder(MODEL_ID, max_length=512, trust_remote_code=False)
    # Verify device
    import torch
    device = next(reranker.model.parameters()).device
    log.info("CrossEncoder device: %s", device)
    assert str(device) == "cpu", f"Expected CPU, got {device}"

    results_by_k = {}

    # Measure at top-50
    r50 = measure_at_k(reranker, prefetched_50, k=50)
    results_by_k[50] = r50

    feasible_k = None
    if r50["p95_ms"] <= LATENCY_BUDGET_P95_MS:
        feasible_k = 50
        log.info("top-50 FEASIBLE (p95=%.1fms ≤ %.0fms budget)", r50["p95_ms"], LATENCY_BUDGET_P95_MS)
    else:
        log.warning("top-50 OVER BUDGET (p95=%.1fms > %.0fms) — testing top-25 mitigation", r50["p95_ms"], LATENCY_BUDGET_P95_MS)
        r25 = measure_at_k(reranker, prefetched_50, k=25)
        results_by_k[25] = r25
        if r25["p95_ms"] <= LATENCY_BUDGET_P95_MS:
            feasible_k = 25
            log.info("top-25 FEASIBLE (p95=%.1fms ≤ %.0fms)", r25["p95_ms"], LATENCY_BUDGET_P95_MS)
        else:
            log.error("top-25 ALSO OVER BUDGET (p95=%.1fms) — bge-v2-m3 CPU infeasible. STOP Phase 2.", r25["p95_ms"])

    decision = (
        f"feasible_k{feasible_k}_proceed_T4" if feasible_k
        else "STOP_Phase2_cpu_infeasible"
    )

    output = {
        "repo": REPO_KEY,
        "model_id": MODEL_ID,
        "latency_budget_p95_ms": LATENCY_BUDGET_P95_MS,
        "n_warmup": N_WARMUP,
        "n_timed": N_TIME,
        "seed": SEED,
        "measurements": results_by_k,
        "feasible_k": feasible_k,
        "decision": decision,
    }

    out = REPORTS_DIR / "phase2_cpu_latency.json"
    out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    log.info("Saved → %s", out)

    print(f"\n{'='*60}")
    print(f"T3 CPU LATENCY — {REPO_KEY}  model={MODEL_ID}")
    print(f"{'='*60}")
    for k, r in sorted(results_by_k.items()):
        flag = " ← FEASIBLE" if k == feasible_k else (" ← OVER BUDGET" if r["p95_ms"] > LATENCY_BUDGET_P95_MS else "")
        print(f"  top-{k:2d}:  p50={r['p50_ms']:7.1f}ms   p95={r['p95_ms']:7.1f}ms{flag}")
    print(f"  Budget:  p95 ≤ {LATENCY_BUDGET_P95_MS:.0f}ms")
    print(f"  Feasible k: {feasible_k}")
    print(f"  Decision: {decision}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
