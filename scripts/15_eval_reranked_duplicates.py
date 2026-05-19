"""W1.3 — Duplicate eval with the chosen cross-encoder reranker.

Produces reports/duplicate_results_reranked.json in the same format as
the existing reports/duplicate_results.json (baseline BGE-only).

Run after the W1.3 reranker is integrated:
    python scripts/15_eval_reranked_duplicates.py

Uses the RERANKER_MODEL env var (default: mixedbread-ai/mxbai-rerank-base-v1).
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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"

REPOS = [
    ("microsoft_vscode", "dup_index_microsoft_vscode_bge"),
    ("kubernetes_kubernetes", "dup_index_kubernetes_kubernetes_bge"),
]
AT_KS = [1, 5, 10, 20]


def _mrr_and_recalls(ranked: list[int], true_orig: int, at_ks: list[int]) -> dict:
    try:
        rank = ranked.index(true_orig) + 1
        mrr = 1.0 / rank
    except ValueError:
        mrr = 0.0
        rank = None
    return {"mrr": mrr, **{f"recall_at_{k}": int(rank is not None and rank <= k) for k in at_ks}}


def eval_repo(repo_key: str, index_dir_name: str, gold: pd.DataFrame) -> dict:
    import os
    from triage_iq.models.duplicates import DuplicateDetector
    from triage_iq.models.reranker import DEFAULT_RERANKER_MODEL, Reranker

    model_id = os.environ.get("RERANKER_MODEL", DEFAULT_RERANKER_MODEL)

    log.info("[%s] Loading BGE index + reranker %s …", repo_key, model_id)
    det = DuplicateDetector.load(str(DATA_DIR / "models" / index_dir_name))
    det.reranker = Reranker(model_id=model_id)

    repo_gold = gold[gold["repo"] == repo_key].copy()
    metrics: dict[str, list] = {f"recall_at_{k}": [] for k in AT_KS}
    metrics["mrr"] = []
    latencies: list[float] = []

    for i, (_, row) in enumerate(repo_gold.iterrows()):
        if i % 100 == 0:
            log.info("[%s] %d/%d …", repo_key, i, len(repo_gold))
        query = f"{row['query_title']}. {row['query_body'][:512]}"
        t0 = time.perf_counter()
        hits = det.retrieve(query, k=20, exclude_number=int(row["query_number"]))
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [h["number"] for h in hits]
        m = _mrr_and_recalls(ranked, int(row["original_number"]), AT_KS)
        for k_name, v in m.items():
            metrics[k_name].append(v)

    return {
        "repo": repo_key,
        "model": f"bge+{model_id.split('/')[-1]}",
        "model_id": model_id,
        "n_pairs": len(repo_gold),
        "mrr": float(np.mean(metrics["mrr"])),
        "recall_at_1": float(np.mean(metrics["recall_at_1"])),
        "recall_at_5": float(np.mean(metrics["recall_at_5"])),
        "recall_at_10": float(np.mean(metrics["recall_at_10"])),
        "recall_at_20": float(np.mean(metrics["recall_at_20"])),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
        "latency_mean_ms": float(np.mean(latencies)),
        "retrieval_k_faiss": 50,
        "retrieval_k_final": 20,
    }


def main() -> None:
    gold = pd.read_parquet(DATA_DIR / "gold_duplicates.parquet")
    results = []

    for repo_key, index_dir in REPOS:
        r = eval_repo(repo_key, index_dir, gold)
        results.append(r)
        log.info("[%s] MRR=%.3f R@5=%.3f R@10=%.3f p50=%.1fms p95=%.1fms",
                 repo_key, r["mrr"], r["recall_at_5"], r["recall_at_10"],
                 r["latency_p50_ms"], r["latency_p95_ms"])

    out = REPORTS_DIR / "duplicate_results_reranked.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Saved to %s", out)

    # Print delta vs baseline
    try:
        baseline = json.loads((REPORTS_DIR / "duplicate_results.json").read_text())
        b_map = {r["repo"]: r for r in baseline if r["model"] == "bge"}
        print("\n=== RERANKER DELTA vs BGE BASELINE ===")
        print(f"{'Repo':<28} {'R@5 base':>9} {'R@5 new':>9} {'Delta':>8} {'R@10 base':>10} {'R@10 new':>10} {'Delta':>8}")
        for r in results:
            b = b_map.get(r["repo"], {})
            b5 = b.get("recall_at_5", 0); b10 = b.get("recall_at_10", 0)
            n5 = r["recall_at_5"]; n10 = r["recall_at_10"]
            print(f"{r['repo']:<28} {b5:>9.3f} {n5:>9.3f} {n5-b5:>+8.3f} {b10:>10.3f} {n10:>10.3f} {n10-b10:>+8.3f}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
