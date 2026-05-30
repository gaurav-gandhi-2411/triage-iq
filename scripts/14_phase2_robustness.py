"""Phase 2 T2 — bge-v2-m3 robustness check on kubernetes/kubernetes.

n=300 random queries, seed=42. Same protocol as W1.3 (FAISS BGE-base k=20
baseline vs bge-v2-m3 rerank of top-50 → top-5). Bootstrap 95% CI on delta.

Decision rule:
  - CI lower bound > 0 → robust, proceed to T3 (CPU latency).
  - CI crosses zero → +6pp was noise, W1.3 rejection stands. STOP Phase 2.

Output: reports/phase2_robustness.json
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

N_EVAL = 300
SEED = 42
RETRIEVAL_K = 50
FINAL_K = 5
BASELINE_K = 20
N_BOOTSTRAP = 1000
REPO_KEY = "kubernetes_kubernetes"
INDEX_DIR = DATA_DIR / "models" / "dup_index_kubernetes_kubernetes_bge"
MODEL_ID = "BAAI/bge-reranker-v2-m3"


def _mrr_and_recalls(ranked: list[int], target: int, at_ks: list[int]) -> dict:
    try:
        rank = ranked.index(target) + 1
        mrr = 1.0 / rank
    except ValueError:
        mrr = 0.0
        rank = None
    return {"mrr": mrr, **{f"recall_at_{k}": int(rank is not None and rank <= k) for k in at_ks}}


def sample_gold(gold: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    repo_gold = gold[gold["repo"] == REPO_KEY].copy()
    rng = random.Random(seed)
    idxs = rng.sample(range(len(repo_gold)), min(n, len(repo_gold)))
    return repo_gold.iloc[idxs].reset_index(drop=True)


def bootstrap_ci(deltas: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    means = np.array([
        rng.choice(deltas, size=len(deltas), replace=True).mean()
        for _ in range(n_boot)
    ])
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def run_baseline(det, gold_sample: pd.DataFrame) -> list[int]:
    r5_hits = []
    for _, row in gold_sample.iterrows():
        query_text = f"{row['query_title']}. {row['query_body'][:512]}"
        hits = det.retrieve(query_text, k=BASELINE_K, exclude_number=int(row["query_number"]))
        ranked = [h["number"] for h in hits]
        m = _mrr_and_recalls(ranked, int(row["original_number"]), [5])
        r5_hits.append(m["recall_at_5"])
    return r5_hits


def run_reranker(det, reranker, gold_sample: pd.DataFrame) -> list[int]:
    r5_hits = []
    for i, (_, row) in enumerate(gold_sample.iterrows()):
        if i % 50 == 0:
            log.info("  reranker %d/%d …", i, len(gold_sample))
        query_text = f"{row['query_title']}. {row['query_body'][:512]}"
        hits = det.retrieve(query_text, k=RETRIEVAL_K, exclude_number=int(row["query_number"]))
        if not hits:
            r5_hits.append(0)
            continue
        pairs = [(query_text, h["text"]) for h in hits]
        ce_scores = reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(ce_scores)[::-1]
        ranked = [hits[j]["number"] for j in order[:FINAL_K]]
        m = _mrr_and_recalls(ranked, int(row["original_number"]), [5])
        r5_hits.append(m["recall_at_5"])
    return r5_hits


def main() -> None:
    from sentence_transformers import CrossEncoder
    from triage_iq.models.similar_issues import SimilarIssueRetriever

    gold = pd.read_parquet(DATA_DIR / "gold_related.parquet")
    log.info("Gold pairs: %d total (%s only) | sampling %d (seed=%d)",
             len(gold[gold["repo"] == REPO_KEY]), REPO_KEY, N_EVAL, SEED)

    sample = sample_gold(gold, N_EVAL, SEED)
    log.info("Sample size: %d", len(sample))

    log.info("Loading FAISS index: %s", INDEX_DIR)
    det = SimilarIssueRetriever.load(str(INDEX_DIR))

    log.info("=== Baseline (BGE-base FAISS k=%d) ===", BASELINE_K)
    t0 = time.perf_counter()
    baseline_hits = run_baseline(det, sample)
    baseline_r5 = float(np.mean(baseline_hits))
    log.info("Baseline R@5=%.4f  (%.1fs)", baseline_r5, time.perf_counter() - t0)

    log.info("=== Reranker (bge-v2-m3 top-%d → top-%d) ===", RETRIEVAL_K, FINAL_K)
    log.info("Loading CrossEncoder: %s", MODEL_ID)
    reranker = CrossEncoder(MODEL_ID, max_length=512, trust_remote_code=False)
    t0 = time.perf_counter()
    reranker_hits = run_reranker(det, reranker, sample)
    reranker_r5 = float(np.mean(reranker_hits))
    log.info("Reranker R@5=%.4f  (%.1fs)", reranker_r5, time.perf_counter() - t0)

    deltas = np.array(reranker_hits) - np.array(baseline_hits)
    mean_delta = float(deltas.mean())
    ci_lo, ci_hi = bootstrap_ci(deltas)
    robust = ci_lo > 0.0

    log.info("Delta mean=%.4f  95%% CI [%.4f, %.4f]  robust=%s",
             mean_delta, ci_lo, ci_hi, robust)

    result = {
        "repo": REPO_KEY,
        "n_eval": len(sample),
        "seed": SEED,
        "retrieval_k_baseline": BASELINE_K,
        "retrieval_k_reranker": RETRIEVAL_K,
        "final_k": FINAL_K,
        "model_id": MODEL_ID,
        "baseline_r5": baseline_r5,
        "reranker_r5": reranker_r5,
        "delta_mean": mean_delta,
        "ci_lo_95": ci_lo,
        "ci_hi_95": ci_hi,
        "n_bootstrap": N_BOOTSTRAP,
        "robust": robust,
        "decision": "proceed_to_T3" if robust else "STOP_Phase2_noise",
    }

    out = REPORTS_DIR / "phase2_robustness.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Saved → %s", out)

    print(f"\n{'='*60}")
    print(f"T2 ROBUSTNESS — {REPO_KEY}  n={len(sample)}")
    print(f"{'='*60}")
    print(f"  Baseline R@5      : {baseline_r5:.4f}")
    print(f"  Reranker R@5      : {reranker_r5:.4f}")
    print(f"  Delta             : {mean_delta:+.4f}")
    print(f"  Bootstrap 95% CI  : [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Robust (CI>0)     : {robust}")
    print(f"  Decision          : {result['decision']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
