"""W1.3 — Focused GPU screening for BAAI/bge-reranker-v2-m3.

Runs bge-reranker-v2-m3 only (jina eliminated: CC-BY-NC-4.0 + BFloat16 fail;
mxbai eliminated: R@5 dropped 16pp on vscode vs baseline).

Output: reports/bge_v2m3_benchmark.json
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
N_EVAL = 100
SEED = 42

MODEL_ID = "BAAI/bge-reranker-v2-m3"
MODEL_KEY = "bge-reranker-v2-m3"

REPOS = [
    ("microsoft_vscode", "dup_index_microsoft_vscode_bge"),
    ("kubernetes_kubernetes", "dup_index_kubernetes_kubernetes_bge"),
]


def _mrr_and_recalls(ranked: list[int], true_orig: int, at_ks: list[int]) -> dict:
    try:
        rank = ranked.index(true_orig) + 1
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
        "repo": repo_key, "model": "baseline_bge_k20", "n_eval": len(gold_sample),
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


def eval_reranker(repo_key: str, index_dir_name: str, gold_sample: pd.DataFrame) -> dict:
    from sentence_transformers import CrossEncoder
    from triage_iq.models.duplicates import DuplicateDetector

    log.info("[%s/%s] Loading BGE index …", repo_key, MODEL_KEY)
    det = DuplicateDetector.load(str(DATA_DIR / "models" / index_dir_name))

    log.info("[%s/%s] Loading CrossEncoder %s …", repo_key, MODEL_KEY, MODEL_ID)
    reranker = CrossEncoder(MODEL_ID, max_length=512, trust_remote_code=False)

    mrrs, r1, r5, r10 = [], [], [], []
    latencies: list[float] = []

    for i, (_, row) in enumerate(gold_sample.iterrows()):
        if i % 10 == 0:
            log.info("[%s/%s] %d/%d …", repo_key, MODEL_KEY, i, len(gold_sample))
        query_text = f"{row['query_title']}. {row['query_body'][:512]}"
        t0 = time.perf_counter()
        hits = det._faiss_retrieve(query_text, k=RETRIEVAL_K, exclude_number=int(row["query_number"]))
        pairs = [(query_text, h["text"]) for h in hits]
        ce_scores = reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(ce_scores)[::-1]
        ranked = [hits[j]["number"] for j in order]
        latencies.append((time.perf_counter() - t0) * 1000)
        m = _mrr_and_recalls(ranked, int(row["original_number"]), [1, 5, 10])
        mrrs.append(m["mrr"]); r1.append(m["recall_at_1"]); r5.append(m["recall_at_5"]); r10.append(m["recall_at_10"])

    r = {
        "repo": repo_key,
        "model": MODEL_KEY,
        "model_id": MODEL_ID,
        "size_mb_approx": 568,
        "license": "Apache-2.0",
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
             repo_key, MODEL_KEY, r["mrr"], r["recall_at_5"], r["recall_at_10"],
             r["latency_p50_ms"], r["latency_p95_ms"])
    return r


def main() -> None:
    gold = pd.read_parquet(DATA_DIR / "gold_duplicates.parquet")
    log.info("Gold pairs: %d total | sampling %d per repo (seed=%d)", len(gold), N_EVAL, SEED)

    results: list[dict] = []

    for repo_key, index_dir in REPOS:
        sample = sample_gold(gold, repo_key, N_EVAL, SEED)
        results.append(eval_baseline(repo_key, index_dir, sample))

    for repo_key, index_dir in REPOS:
        sample = sample_gold(gold, repo_key, N_EVAL, SEED)
        try:
            results.append(eval_reranker(repo_key, index_dir, sample))
        except Exception as exc:
            log.error("[%s/%s] FAILED: %s", MODEL_KEY, repo_key, exc)
            results.append({"model": MODEL_KEY, "repo": repo_key, "error": str(exc)})

    out = REPORTS_DIR / "bge_v2m3_benchmark.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Saved → %s", out)

    print("\n=== BGE-v2-m3 SCREENING (n=100 queries per repo) ===")
    print(f"{'Model':<40} {'Repo':<28} {'R@5':>6} {'R@10':>6} {'MRR':>6} {'p50ms':>8} {'p95ms':>8}")
    for r in results:
        if "error" in r:
            print(f"FAILED: {r['model']} / {r['repo']}: {r['error']}")
            continue
        print(f"{r['model']:<40} {r['repo']:<28} {r['recall_at_5']:>6.3f} {r['recall_at_10']:>6.3f} {r['mrr']:>6.3f} {r['latency_p50_ms']:>8.1f} {r['latency_p95_ms']:>8.1f}")

    # Check if bge-v2-m3 beats baseline on both repos
    b = {r["repo"]: r for r in results if r["model"] == "baseline_bge_k20"}
    c = {r["repo"]: r for r in results if r["model"] == MODEL_KEY}
    print("\n=== PASS/FAIL vs BASELINE ===")
    all_pass = True
    for repo in ["microsoft_vscode", "kubernetes_kubernetes"]:
        if repo not in b or repo not in c:
            print(f"  {repo}: INCOMPLETE")
            all_pass = False
            continue
        delta = c[repo]["recall_at_5"] - b[repo]["recall_at_5"]
        status = "PASS" if delta > 0 else "FAIL (no improvement)"
        print(f"  {repo}: baseline R@5={b[repo]['recall_at_5']:.3f}, bge-v2-m3 R@5={c[repo]['recall_at_5']:.3f}, delta={delta:+.3f} → {status}")
        if delta <= 0:
            all_pass = False
    print(f"\nOverall: {'PASS — proceed to CPU latency' if all_pass else 'FAIL — surface to GG before proceeding'}")


if __name__ == "__main__":
    main()
