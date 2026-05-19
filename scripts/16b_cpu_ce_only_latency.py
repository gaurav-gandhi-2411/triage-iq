"""W1.3 — CPU-only CrossEncoder latency (CE step only, no BGE contention).

Approach:
  1. Pre-fetch FAISS candidates for all queries (uses BGE/FAISS normally).
  2. Store (query_text, candidates) pairs.
  3. Time ONLY ce.predict() on CPU for 100 pairs per repo.

This separates the CE inference time from BGE encoding so GPU contention
from the parallel benchmark does not pollute the measurement.

Production /triage latency is:
  BGE CPU encode (≈26ms vscode, ≈40ms k8s from audit) + CE CPU predict (this script)

Output: reports/cpu_ce_latency.json
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# Force CE to CPU — set before any torch import
os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"
FAISS_K = 50
N_QUERIES = 100
WARMUP = 10
SEED = 42

REPOS = [
    ("microsoft_vscode",      "dup_index_microsoft_vscode_bge"),
    ("kubernetes_kubernetes",  "dup_index_kubernetes_kubernetes_bge"),
]

MODELS = {
    "mxbai": {
        "model_id": "mixedbread-ai/mxbai-rerank-base-v1",
        "trust_remote_code": False,
        "size_mb_approx": 184,
        "license": "Apache-2.0",
    },
    # Uncomment to benchmark other candidates (jina is CC-BY-NC-4.0, eliminated):
    # "bge-reranker": {
    #     "model_id": "BAAI/bge-reranker-v2-m3",
    #     "trust_remote_code": False,
    #     "size_mb_approx": 568,
    #     "license": "Apache-2.0",
    # },
}


def prefetch_candidates(
    gold: pd.DataFrame,
    repo_key: str,
    index_path: str,
    n: int,
    seed: int = SEED,
) -> list[tuple[str, list[str]]]:
    """Return (query_text, [candidate_texts]) for n sampled queries."""
    from triage_iq.models.duplicates import DuplicateDetector

    log.info("[%s] Pre-fetching FAISS candidates for %d queries …", repo_key, n + WARMUP)
    det = DuplicateDetector.load(index_path)

    repo_gold = gold[gold["repo"] == repo_key].copy()
    rng = random.Random(seed)
    idxs = rng.sample(range(len(repo_gold)), min(n + WARMUP, len(repo_gold)))
    rows = repo_gold.iloc[idxs]

    pairs = []
    for _, row in rows.iterrows():
        query = f"{row['query_title']}. {row['query_body'][:512]}"
        exclude = int(row["query_number"])
        hits = det._faiss_retrieve(query, FAISS_K, exclude)
        candidate_texts = [h["text"] for h in hits]
        pairs.append((query, candidate_texts))

    log.info("[%s] Pre-fetch done — %d queries × %d candidates", repo_key, len(pairs), FAISS_K)
    return pairs


def time_ce_cpu(model_id: str, trust_remote_code: bool, pairs: list[tuple[str, list[str]]]) -> list[float]:
    """Load CrossEncoder on CPU and time predict() for each (query, candidates) pair."""
    from sentence_transformers import CrossEncoder

    log.info("Loading CrossEncoder on CPU: %s", model_id)
    ce = CrossEncoder(model_id, max_length=512, trust_remote_code=trust_remote_code, device="cpu")
    log.info("CrossEncoder loaded on CPU. Device check: %s", ce.model.device)

    warmup_pairs = pairs[:WARMUP]
    timed_pairs = pairs[WARMUP:]

    log.info("Warmup (%d) …", len(warmup_pairs))
    for query, candidates in warmup_pairs:
        input_pairs = [(query, c) for c in candidates]
        ce.predict(input_pairs, show_progress_bar=False)

    log.info("Timing %d queries (CE only) …", len(timed_pairs))
    latencies: list[float] = []
    for i, (query, candidates) in enumerate(timed_pairs):
        if i % 20 == 0:
            log.info("  %d/%d …", i, len(timed_pairs))
        input_pairs = [(query, c) for c in candidates]
        t0 = time.perf_counter()
        ce.predict(input_pairs, show_progress_bar=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies


def main() -> None:
    gold = pd.read_parquet(DATA_DIR / "gold_duplicates.parquet")
    results: list[dict] = []

    for model_key, cfg in MODELS.items():
        model_id = cfg["model_id"]

        for repo_key, index_dir in REPOS:
            index_path = str(DATA_DIR / "models" / index_dir)
            log.info("=== %s / %s ===", model_key, repo_key)

            try:
                pairs = prefetch_candidates(gold, repo_key, index_path, N_QUERIES)
                latencies = time_ce_cpu(model_id, cfg["trust_remote_code"], pairs)

                row = {
                    "model_key": model_key,
                    "model_id": model_id,
                    "size_mb_approx": cfg["size_mb_approx"],
                    "repo": repo_key,
                    "n_queries": len(latencies),
                    "faiss_k": FAISS_K,
                    "device": "cpu",
                    "measurement": "ce_predict_only",
                    "p50_ms": float(np.percentile(latencies, 50)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "mean_ms": float(np.mean(latencies)),
                    "max_ms": float(np.max(latencies)),
                }
                results.append(row)
                log.info("[%s/%s] CE-only CPU p50=%.0fms p95=%.0fms mean=%.0fms max=%.0fms",
                         model_key, repo_key, row["p50_ms"], row["p95_ms"],
                         row["mean_ms"], row["max_ms"])

            except Exception as exc:
                log.error("[%s/%s] FAILED: %s", model_key, repo_key, exc)
                results.append({
                    "model_key": model_key,
                    "repo": repo_key,
                    "device": "cpu",
                    "measurement": "ce_predict_only",
                    "error": str(exc),
                })

    out = REPORTS_DIR / "cpu_ce_latency.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Saved → %s", out)

    print("\n=== CE-ONLY CPU LATENCY SUMMARY (add BGE baseline for total retrieve+rerank) ===")
    print(f"{'Model':<20} {'Repo':<25} {'p50':>7} {'p95':>7} {'mean':>7}")
    for r in results:
        if "error" in r:
            print(f"{'ERROR:'+r['model_key']:<20} {r['repo']:<25}")
            continue
        print(f"{r['model_key']:<20} {r['repo']:<25} {r['p50_ms']:>6.0f}ms {r['p95_ms']:>6.0f}ms {r['mean_ms']:>6.0f}ms")

    # Remind caller of the BGE baseline to add
    print("\nBGE CPU baseline (from production audit):")
    print("  microsoft_vscode:     p50=26.7ms")
    print("  kubernetes_kubernetes: p50=39.5ms")
    print("Total rerank latency = CE p50 + BGE baseline")


if __name__ == "__main__":
    main()
