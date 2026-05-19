"""W1.3 — Targeted reranker benchmark (fast version).

Evaluates only the two Apache-2.0 candidates (jina eliminated — CC-BY-NC-4.0):
  - mixedbread-ai/mxbai-rerank-base-v1      (~184 MB)
  - BAAI/bge-reranker-v2-m3                  (~568 MB)

Uses N_EVAL=100 randomly sampled queries per repo (instead of all 1435)
for a run time of ~15 minutes instead of 16+ hours.

Output: reports/reranker_benchmark.json (same schema as 14_benchmark_rerankers.py)
"""

from __future__ import annotations

import json
import logging
import random
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

RETRIEVAL_K = 50
FINAL_K = 5
N_EVAL = 100   # queries per repo
SEED = 42

RERANKER_CANDIDATES = [
    {
        "key": "mxbai-rerank-base-v1",
        "model_id": "mixedbread-ai/mxbai-rerank-base-v1",
        "size_mb_approx": 184,
        "license": "Apache-2.0",
        "trust_remote_code": False,
    },
    {
        "key": "bge-reranker-v2-m3",
        "model_id": "BAAI/bge-reranker-v2-m3",
        "size_mb_approx": 568,
        "license": "Apache-2.0",
        "trust_remote_code": False,
    },
    # jina: CC-BY-NC-4.0 — eliminated regardless of performance
    {
        "key": "jina-reranker-v2-base-multilingual",
        "model_id": "jinaai/jina-reranker-v2-base-multilingual",
        "size_mb_approx": 278,
        "license": "CC-BY-NC-4.0",
        "trust_remote_code": True,
        "skip": True,  # eliminated: non-commercial license
    },
]

REPOS = [
    ("microsoft_vscode", "dup_index_microsoft_vscode_bge"),
    ("kubernetes_kubernetes", "dup_index_kubernetes_kubernetes_bge"),
]


def _mrr_and_recalls(ranked_hits: list[int], true_orig: int, at_ks: list[int]) -> dict:
    try:
        rank = ranked_hits.index(true_orig) + 1
        mrr = 1.0 / rank
    except ValueError:
        mrr = 0.0
        rank = None
    return {"mrr": mrr, **{f"recall_at_{k}": int(rank is not None and rank <= k) for k in at_ks}}


def sample_gold(gold: pd.DataFrame, repo_key: str, n: int, seed: int) -> pd.DataFrame:
    repo_gold = gold[gold["repo"] == repo_key].copy()
    rng = random.Random(seed)
    idxs = rng.sample(range(len(repo_gold)), min(n, len(repo_gold)))
    return repo_gold.iloc[idxs]


def eval_baseline(repo_key: str, index_dir_name: str, gold_sample: pd.DataFrame) -> dict:
    from triage_iq.models.duplicates import DuplicateDetector
    log.info("[baseline/%s] Loading BGE index …", repo_key)
    det = DuplicateDetector.load(str(DATA_DIR / "models" / index_dir_name))

    mrrs, r1, r5, r10 = [], [], [], []
    latencies: list[float] = []
    for _, row in gold_sample.iterrows():
        query_text = f"{row['query_title']}. {row['query_body'][:512]}"
        t0 = time.perf_counter()
        hits = det._faiss_retrieve(query_text, k=max(20, FINAL_K), exclude_number=int(row["query_number"]))
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [h["number"] for h in hits]
        m = _mrr_and_recalls(ranked, int(row["original_number"]), [1, 5, 10])
        mrrs.append(m["mrr"]); r1.append(m["recall_at_1"]); r5.append(m["recall_at_5"]); r10.append(m["recall_at_10"])

    r = {
        "repo": repo_key, "model": "baseline_bge_k5", "n_eval": len(gold_sample),
        "mrr": float(np.mean(mrrs)),
        "recall_at_1": float(np.mean(r1)),
        "recall_at_5": float(np.mean(r5)),
        "recall_at_10": float(np.mean(r10)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }
    log.info("[baseline/%s] MRR=%.3f R@5=%.3f R@10=%.3f p50=%.1fms p95=%.1fms",
             repo_key, r["mrr"], r["recall_at_5"], r["recall_at_10"],
             r["latency_p50_ms"], r["latency_p95_ms"])
    return r


def eval_reranker(repo_key: str, index_dir_name: str, gold_sample: pd.DataFrame, candidate: dict) -> dict:
    from sentence_transformers import CrossEncoder
    from triage_iq.models.duplicates import DuplicateDetector

    key = candidate["key"]
    model_id = candidate["model_id"]

    log.info("[%s/%s] Loading BGE index …", repo_key, key)
    det = DuplicateDetector.load(str(DATA_DIR / "models" / index_dir_name))

    log.info("[%s/%s] Loading CrossEncoder %s …", repo_key, key, model_id)
    reranker = CrossEncoder(model_id, max_length=512, trust_remote_code=candidate["trust_remote_code"])

    mrrs, r1, r5, r10 = [], [], [], []
    latencies: list[float] = []

    for i, (_, row) in enumerate(gold_sample.iterrows()):
        if i % 20 == 0:
            log.info("[%s/%s] %d/%d …", repo_key, key, i, len(gold_sample))
        query_text = f"{row['query_title']}. {row['query_body'][:512]}"
        t0 = time.perf_counter()
        hits = det.retrieve(query_text, k=RETRIEVAL_K, exclude_number=int(row["query_number"]))
        pairs = [(query_text, h["text"]) for h in hits]
        ce_scores = reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(ce_scores)[::-1]
        ranked = [hits[j]["number"] for j in order]
        latencies.append((time.perf_counter() - t0) * 1000)
        m = _mrr_and_recalls(ranked, int(row["original_number"]), [1, 5, 10])
        mrrs.append(m["mrr"]); r1.append(m["recall_at_1"]); r5.append(m["recall_at_5"]); r10.append(m["recall_at_10"])

    r = {
        "repo": repo_key,
        "model": key,
        "model_id": model_id,
        "size_mb_approx": candidate["size_mb_approx"],
        "license": candidate.get("license", "unknown"),
        "n_eval": len(gold_sample),
        "retrieval_k": RETRIEVAL_K,
        "mrr": float(np.mean(mrrs)),
        "recall_at_1": float(np.mean(r1)),
        "recall_at_5": float(np.mean(r5)),
        "recall_at_10": float(np.mean(r10)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }
    log.info("[%s/%s] MRR=%.3f R@5=%.3f R@10=%.3f p50=%.1fms p95=%.1fms",
             repo_key, key, r["mrr"], r["recall_at_5"], r["recall_at_10"],
             r["latency_p50_ms"], r["latency_p95_ms"])
    return r


def main() -> None:
    gold = pd.read_parquet(DATA_DIR / "gold_duplicates.parquet")
    log.info("Gold pairs: %d total", len(gold))

    results: list[dict] = []

    # Baseline
    for repo_key, index_dir in REPOS:
        sample = sample_gold(gold, repo_key, N_EVAL, SEED)
        results.append(eval_baseline(repo_key, index_dir, sample))

    # Rerankers (skip jina)
    for cand in RERANKER_CANDIDATES:
        if cand.get("skip"):
            log.info("[SKIP] %s — license: %s", cand["key"], cand.get("license", "?"))
            results.append({
                "model": cand["key"],
                "model_id": cand["model_id"],
                "license": cand.get("license", "?"),
                "size_mb_approx": cand["size_mb_approx"],
                "skipped": True,
                "reason": "CC-BY-NC-4.0 license — not suitable for production portfolio use",
            })
            continue

        for repo_key, index_dir in REPOS:
            sample = sample_gold(gold, repo_key, N_EVAL, SEED)
            try:
                results.append(eval_reranker(repo_key, index_dir, sample, cand))
            except Exception as exc:
                log.error("[%s/%s] FAILED: %s", cand["key"], repo_key, exc)
                results.append({"model": cand["key"], "repo": repo_key, "error": str(exc)})

    out = REPORTS_DIR / "reranker_benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Saved → %s", out)

    print("\n=== BENCHMARK SUMMARY (n=100 queries per repo) ===")
    print(f"{'Model':<40} {'Repo':<25} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'p50ms':>7} {'p95ms':>7}")
    for r in results:
        if r.get("skipped") or "error" in r:
            print(f"{'SKIP: ' + r['model']:<40}")
            continue
        print(f"{r['model']:<40} {r['repo']:<25} {r['recall_at_5']:>6.3f} {r['recall_at_10']:>6.3f} {r['mrr']:>6.3f} {r['latency_p50_ms']:>7.1f} {r['latency_p95_ms']:>7.1f}")


if __name__ == "__main__":
    main()
