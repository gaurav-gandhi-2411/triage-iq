"""W1.3 pivot — GPU screening for duplicate/paraphrase-trained cross-encoders.

First slate (mxbai/bge-v2-m3/jina) failed because they're search-relevance models,
not duplicate-detection models. This script screens:
  1. cross-encoder/quora-distilroberta-base  — Quora Duplicate Questions, ~82MB, Apache-2.0
  2. cross-encoder/quora-roberta-base        — Quora-trained, larger base, ~125MB, Apache-2.0
  3. cross-encoder/stsb-distilroberta-base   — STS-Benchmark, ~82MB, Apache-2.0
  4. BAAI/bge-reranker-base                  — BGE reranker base variant, ~278MB, Apache-2.0

Protocol: same as 14c — n=100 random queries seed=42, both repos.
Baseline: BGE-base FAISS k=20 (same anchor as previous screening).
Reranker eval: FAISS top-50 → cross-encoder rerank → evaluate R@5 from all 50.

Output: reports/dup_trained_reranker_screening.json
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

CANDIDATES = [
    {
        "key": "quora-distilroberta-base",
        "model_id": "cross-encoder/quora-distilroberta-base",
        "size_mb_approx": 82,
        "license": "Apache-2.0",
        "trust_remote_code": False,
        "training": "Quora Duplicate Questions",
    },
    {
        "key": "quora-roberta-base",
        "model_id": "cross-encoder/quora-roberta-base",
        "size_mb_approx": 125,
        "license": "Apache-2.0",
        "trust_remote_code": False,
        "training": "Quora Duplicate Questions (larger)",
    },
    {
        "key": "stsb-distilroberta-base",
        "model_id": "cross-encoder/stsb-distilroberta-base",
        "size_mb_approx": 82,
        "license": "Apache-2.0",
        "trust_remote_code": False,
        "training": "STS-Benchmark (semantic similarity)",
    },
    {
        "key": "bge-reranker-base",
        "model_id": "BAAI/bge-reranker-base",
        "size_mb_approx": 278,
        "license": "MIT",
        "trust_remote_code": False,
        "training": "Mixed retrieval/reranking",
    },
]

REPOS = [
    ("microsoft_vscode",      "dup_index_microsoft_vscode_bge"),
    ("kubernetes_kubernetes",  "dup_index_kubernetes_kubernetes_bge"),
]

BASELINE = {
    "microsoft_vscode":      {"recall_at_5": 0.470, "recall_at_10": 0.550, "mrr": 0.385},
    "kubernetes_kubernetes": {"recall_at_5": 0.430, "recall_at_10": 0.520, "mrr": 0.286},
}


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
    log.info("[baseline/%s] Running BGE-base FAISS k=20 …", repo_key)
    det = DuplicateDetector.load(str(DATA_DIR / "models" / index_dir_name))
    mrrs, r1, r5, r10 = [], [], [], []
    latencies: list[float] = []
    for _, row in gold_sample.iterrows():
        q = f"{row['query_title']}. {row['query_body'][:512]}"
        t0 = time.perf_counter()
        hits = det._faiss_retrieve(q, k=max(20, FINAL_K), exclude_number=int(row["query_number"]))
        latencies.append((time.perf_counter() - t0) * 1000)
        ranked = [h["number"] for h in hits]
        m = _mrr_and_recalls(ranked, int(row["original_number"]), [1, 5, 10])
        mrrs.append(m["mrr"]); r1.append(m["recall_at_1"]); r5.append(m["recall_at_5"]); r10.append(m["recall_at_10"])
    r = {
        "repo": repo_key, "model": "baseline_bge_k20", "n_eval": len(gold_sample),
        "mrr": float(np.mean(mrrs)), "recall_at_1": float(np.mean(r1)),
        "recall_at_5": float(np.mean(r5)), "recall_at_10": float(np.mean(r10)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }
    log.info("[baseline/%s] MRR=%.3f R@5=%.3f R@10=%.3f p50=%.1fms",
             repo_key, r["mrr"], r["recall_at_5"], r["recall_at_10"], r["latency_p50_ms"])
    return r


def eval_candidate(repo_key: str, index_dir_name: str, gold_sample: pd.DataFrame, cand: dict) -> dict:
    from sentence_transformers import CrossEncoder
    from triage_iq.models.duplicates import DuplicateDetector

    key = cand["key"]
    model_id = cand["model_id"]

    log.info("[%s/%s] Loading index + CrossEncoder %s …", repo_key, key, model_id)
    det = DuplicateDetector.load(str(DATA_DIR / "models" / index_dir_name))
    reranker = CrossEncoder(model_id, max_length=512, trust_remote_code=cand["trust_remote_code"])

    mrrs, r1, r5, r10 = [], [], [], []
    latencies: list[float] = []

    for i, (_, row) in enumerate(gold_sample.iterrows()):
        if i % 10 == 0:
            log.info("[%s/%s] %d/%d …", repo_key, key, i, len(gold_sample))
        q = f"{row['query_title']}. {row['query_body'][:512]}"
        t0 = time.perf_counter()
        hits = det._faiss_retrieve(q, k=RETRIEVAL_K, exclude_number=int(row["query_number"]))
        pairs = [(q, h["text"]) for h in hits]
        ce_scores = reranker.predict(pairs, show_progress_bar=False)
        order = np.argsort(ce_scores)[::-1]
        ranked = [hits[j]["number"] for j in order]
        latencies.append((time.perf_counter() - t0) * 1000)
        m = _mrr_and_recalls(ranked, int(row["original_number"]), [1, 5, 10])
        mrrs.append(m["mrr"]); r1.append(m["recall_at_1"]); r5.append(m["recall_at_5"]); r10.append(m["recall_at_10"])

    r = {
        "repo": repo_key, "model": key, "model_id": model_id,
        "size_mb_approx": cand["size_mb_approx"], "license": cand["license"],
        "training": cand["training"],
        "n_eval": len(gold_sample), "retrieval_k": RETRIEVAL_K,
        "mrr": float(np.mean(mrrs)), "recall_at_1": float(np.mean(r1)),
        "recall_at_5": float(np.mean(r5)), "recall_at_10": float(np.mean(r10)),
        "latency_p50_ms": float(np.percentile(latencies, 50)),
        "latency_p95_ms": float(np.percentile(latencies, 95)),
    }
    log.info("[%s/%s] MRR=%.3f R@5=%.3f R@10=%.3f p50=%.1fms p95=%.1fms",
             repo_key, key, r["mrr"], r["recall_at_5"], r["recall_at_10"],
             r["latency_p50_ms"], r["latency_p95_ms"])
    return r


def main() -> None:
    gold = pd.read_parquet(DATA_DIR / "gold_duplicates.parquet")
    log.info("Gold pairs: %d total | screening %d per repo (seed=%d)", len(gold), N_EVAL, SEED)

    results: list[dict] = []

    # Baseline for both repos
    for repo_key, index_dir in REPOS:
        sample = sample_gold(gold, repo_key, N_EVAL, SEED)
        results.append(eval_baseline(repo_key, index_dir, sample))

    # All 4 candidates
    for cand in CANDIDATES:
        for repo_key, index_dir in REPOS:
            sample = sample_gold(gold, repo_key, N_EVAL, SEED)
            try:
                results.append(eval_candidate(repo_key, index_dir, sample, cand))
            except Exception as exc:
                log.error("[%s/%s] FAILED: %s", cand["key"], repo_key, exc)
                results.append({"model": cand["key"], "repo": repo_key, "error": str(exc), "model_id": cand["model_id"]})

    out = REPORTS_DIR / "dup_trained_reranker_screening.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log.info("Saved -> %s", out)

    # Print screening table
    print("\n=== DUP-TRAINED RERANKER SCREENING (n=100 per repo) ===")
    print(f"{'Candidate':<35} {'Params':>7} {'Repo':<25} {'R@5':>6} {'vs base':>8} {'R@10':>6} {'MRR':>6} {'p50ms':>8} {'p95ms':>8}")
    b_map = {r["repo"]: r for r in results if r.get("model") == "baseline_bge_k20"}
    for r in results:
        if r.get("model") == "baseline_bge_k20":
            print(f"{'[baseline BGE k=20]':<35} {'—':>7} {r['repo']:<25} {r['recall_at_5']:>6.3f} {'—':>8} {r['recall_at_10']:>6.3f} {r['mrr']:>6.3f} {r['latency_p50_ms']:>8.1f} {r['latency_p95_ms']:>8.1f}")
            continue
        if "error" in r:
            print(f"{r['model']:<35} {'ERROR':<7} {r['repo']:<25} {r.get('error','?')[:40]}")
            continue
        b = b_map.get(r["repo"], {})
        delta = r["recall_at_5"] - b.get("recall_at_5", 0)
        size = f"{r['size_mb_approx']}MB"
        print(f"{r['model']:<35} {size:>7} {r['repo']:<25} {r['recall_at_5']:>6.3f} {delta:>+8.3f} {r['recall_at_10']:>6.3f} {r['mrr']:>6.3f} {r['latency_p50_ms']:>8.1f} {r['latency_p95_ms']:>8.1f}")

    # Decision tree
    print("\n=== DECISION TREE ===")
    cand_keys = [c["key"] for c in CANDIDATES]
    cand_results = {k: {"microsoft_vscode": {}, "kubernetes_kubernetes": {}} for k in cand_keys}
    for r in results:
        if "error" not in r and r.get("model") in cand_keys:
            cand_results[r["model"]][r["repo"]] = r

    case = None
    winner = None
    for key in cand_keys:
        vs = cand_results[key]["microsoft_vscode"]
        ks = cand_results[key]["kubernetes_kubernetes"]
        if not vs or not ks:
            continue
        b_vs = b_map.get("microsoft_vscode", {}).get("recall_at_5", 0)
        b_ks = b_map.get("kubernetes_kubernetes", {}).get("recall_at_5", 0)
        delta_vs = vs.get("recall_at_5", 0) - b_vs
        delta_ks = ks.get("recall_at_5", 0) - b_ks
        both_pass = delta_vs >= 0.03 and delta_ks >= 0.03
        ks_pass_no_vs_regress = delta_ks >= 0.03 and delta_vs >= -0.01
        ks_pass_vs_regress = delta_ks >= 0.03 and delta_vs < -0.01
        print(f"  {key:35s}: vscode delta={delta_vs:+.3f}, k8s delta={delta_ks:+.3f}", end="")
        if both_pass:
            print(f" → CASE A eligible (both repos +3pp)")
            if case is None or case > "A":
                case = "A"
                winner = key
        elif ks_pass_no_vs_regress:
            print(f" → CASE B eligible (k8s +3pp, vscode within ±1pp)")
            if case is None or case > "B":
                case = "B"
                winner = key
        elif ks_pass_vs_regress:
            print(f" → CASE C eligible (k8s +3pp, vscode regresses >{-0.01:.0%})")
            if case is None or case > "C":
                case = "C"
                winner = key
        else:
            print(f" → no case")

    if case is None:
        print("\nRESULT: CASE D — No candidate beats baseline on either repo by >=3pp.")
        print("  Action: Drop W1.3. Close PR #1 without merging. Document as ADR-0006 negative result.")
    else:
        size = next((c["size_mb_approx"] for c in CANDIDATES if c["key"] == winner), "?")
        print(f"\nRESULT: {case} — winner = {winner} ({size}MB)")
        if case == "A":
            print("  Action: Proceed to CPU latency, full eval, ADR-0006, merge.")
        elif case == "B":
            print("  Action: Repo-gated reranker (k8s only). ADR-0006 documents asymmetric design.")
        elif case == "C":
            print("  Action: Repo-gated reranker (k8s only). ADR-0006 flags vscode corpus finding.")


if __name__ == "__main__":
    main()
