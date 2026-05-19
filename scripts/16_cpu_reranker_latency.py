"""W1.3 — CPU latency measurement for cross-encoder reranker.

Forces CPU-only inference (CUDA_VISIBLE_DEVICES="") to reproduce Cloud Run
production conditions. Times the full rerank path (50 FAISS candidates →
cross-encoder → top-5) over 100 real queries per repo.

Reports p50 / p95 / mean latency in milliseconds for each candidate model.

Usage:
    python scripts/16_cpu_reranker_latency.py
    python scripts/16_cpu_reranker_latency.py --models mxbai  # single model
    python scripts/16_cpu_reranker_latency.py --n-queries 100 --warmup 10

Output: reports/cpu_reranker_latency.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

# Force CPU BEFORE any torch import so GPU is invisible
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

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

CANDIDATE_MODELS = {
    "mxbai": {
        "model_id": "mixedbread-ai/mxbai-rerank-base-v1",
        "trust_remote_code": False,
        "size_mb_approx": 184,
    },
    "jina": {
        "model_id": "jinaai/jina-reranker-v2-base-multilingual",
        "trust_remote_code": True,
        "size_mb_approx": 278,
    },
    "bge-reranker": {
        "model_id": "BAAI/bge-reranker-v2-m3",
        "trust_remote_code": False,
        "size_mb_approx": 568,
    },
}

REPOS = [
    ("microsoft_vscode", "dup_index_microsoft_vscode_bge"),
    ("kubernetes_kubernetes", "dup_index_kubernetes_kubernetes_bge"),
]

FAISS_K = 50  # must match production FAISS_RERANK_K
FINAL_K = 5


def load_real_queries(gold: pd.DataFrame, repo_key: str, n: int, seed: int = 42) -> list[tuple[str, int]]:
    """Sample n (query_text, exclude_number) pairs from gold for the given repo."""
    repo_gold = gold[gold["repo"] == repo_key].copy()
    rng = random.Random(seed)
    idxs = rng.sample(range(len(repo_gold)), min(n, len(repo_gold)))
    rows = repo_gold.iloc[idxs]
    return [
        (f"{row['query_title']}. {row['query_body'][:512]}", int(row["query_number"]))
        for _, row in rows.iterrows()
    ]


def time_reranker(
    model_id: str,
    trust_remote_code: bool,
    queries: list[tuple[str, int]],
    index_path: str,
    warmup: int = 5,
) -> list[float]:
    """Return per-query end-to-end latency in ms (FAISS retrieve + CE rerank)."""
    from sentence_transformers import CrossEncoder
    from triage_iq.models.duplicates import DuplicateDetector

    log.info("Loading BGE index: %s", index_path)
    det = DuplicateDetector.load(index_path)

    log.info("Loading CrossEncoder (CPU-forced): %s", model_id)
    ce = CrossEncoder(model_id, max_length=512, trust_remote_code=trust_remote_code, device="cpu")

    log.info("Warming up (%d queries) …", warmup)
    for query_text, exclude_num in queries[:warmup]:
        hits = det._faiss_retrieve(query_text, FAISS_K, exclude_num)
        pairs = [(query_text, h["text"]) for h in hits]
        ce.predict(pairs, show_progress_bar=False)

    log.info("Timing %d queries …", len(queries))
    latencies: list[float] = []
    for i, (query_text, exclude_num) in enumerate(queries):
        if i % 20 == 0:
            log.info("  %d/%d …", i, len(queries))
        t0 = time.perf_counter()
        hits = det._faiss_retrieve(query_text, FAISS_K, exclude_num)
        pairs = [(query_text, h["text"]) for h in hits]
        _ = ce.predict(pairs, show_progress_bar=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    return latencies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(CANDIDATE_MODELS.keys()),
        choices=list(CANDIDATE_MODELS.keys()),
        help="Which models to time. Default: all.",
    )
    parser.add_argument("--n-queries", type=int, default=100, help="Queries per repo per model.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup queries (not timed).")
    args = parser.parse_args()

    gold = pd.read_parquet(DATA_DIR / "gold_duplicates.parquet")
    results: list[dict] = []

    for model_key in args.models:
        cfg = CANDIDATE_MODELS[model_key]
        model_id = cfg["model_id"]
        log.info("=== %s (CPU-only) ===", model_id)

        for repo_key, index_dir in REPOS:
            index_path = str(DATA_DIR / "models" / index_dir)
            queries = load_real_queries(gold, repo_key, args.n_queries)
            log.info("[%s/%s] %d queries, %d warmup", model_key, repo_key, len(queries), args.warmup)

            try:
                latencies = time_reranker(
                    model_id=model_id,
                    trust_remote_code=cfg["trust_remote_code"],
                    queries=queries,
                    index_path=index_path,
                    warmup=args.warmup,
                )
                row = {
                    "model_key": model_key,
                    "model_id": model_id,
                    "size_mb_approx": cfg["size_mb_approx"],
                    "repo": repo_key,
                    "n_queries": len(latencies),
                    "faiss_k": FAISS_K,
                    "final_k": FINAL_K,
                    "device": "cpu",
                    "p50_ms": float(np.percentile(latencies, 50)),
                    "p95_ms": float(np.percentile(latencies, 95)),
                    "mean_ms": float(np.mean(latencies)),
                    "max_ms": float(np.max(latencies)),
                }
                results.append(row)
                log.info(
                    "[%s/%s] CPU p50=%.0fms p95=%.0fms mean=%.0fms",
                    model_key, repo_key, row["p50_ms"], row["p95_ms"], row["mean_ms"],
                )
            except Exception as exc:
                log.error("[%s/%s] FAILED: %s", model_key, repo_key, exc)
                results.append({
                    "model_key": model_key,
                    "repo": repo_key,
                    "device": "cpu",
                    "error": str(exc),
                })

    out = REPORTS_DIR / "cpu_reranker_latency.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Saved → %s", out)

    print("\n=== CPU LATENCY SUMMARY ===")
    print(f"{'Model':<35} {'Repo':<25} {'p50':>7} {'p95':>7} {'mean':>7}")
    for r in results:
        if "error" in r:
            print(f"{'ERROR:'+r['model_key']:<35} {r['repo']:<25}")
            continue
        print(f"{r['model_key']:<35} {r['repo']:<25} {r['p50_ms']:>6.0f}ms {r['p95_ms']:>6.0f}ms {r['mean_ms']:>6.0f}ms")


if __name__ == "__main__":
    main()
