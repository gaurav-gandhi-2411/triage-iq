"""Build FAISS duplicate detection indices and evaluate Recall@K.

Runs both BGE-base and MiniLM-L6 embeddings per repo.
CPU-only inference (production target).

Usage:
    python scripts/08_build_duplicate_index.py
    python scripts/08_build_duplicate_index.py --repos microsoft_vscode --models bge
"""

import argparse
import json
import logging
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage_iq.models.duplicates import DuplicateDetector, _build_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]
DEFAULT_MODELS = ["bge", "minilm"]
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
CHARTS_DIR = Path("reports/charts")
GOLD_PATH = Path("data/gold_duplicates.parquet")
RESULTS_PATH = Path("reports/duplicate_results.json")

K_VALUES = [1, 5, 10, 20]


# ── Evaluation ────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_numbers: list[int], relevant_number: int, k: int) -> float:
    return float(relevant_number in retrieved_numbers[:k])


def reciprocal_rank(retrieved_numbers: list[int], relevant_number: int) -> float:
    for i, n in enumerate(retrieved_numbers, 1):
        if n == relevant_number:
            return 1.0 / i
    return 0.0


def evaluate_detector(
    detector: DuplicateDetector,
    gold_df: pd.DataFrame,
    repo: str,
    k_max: int = 20,
) -> dict:
    repo_gold = gold_df[gold_df["repo"] == repo].reset_index(drop=True)
    if len(repo_gold) == 0:
        log.warning("No gold pairs for %s", repo)
        return {}

    log.info("[%s/%s] Evaluating on %d gold pairs...", repo, detector.model_key, len(repo_gold))

    recalls = {k: [] for k in K_VALUES}
    mrr_scores = []
    latencies_ms = []

    for _, row in repo_gold.iterrows():
        query_text = str(row["query_title"]) + ". " + str(row["query_body"])[:512]
        t0 = time.perf_counter()
        results = detector.retrieve(query_text, k=k_max, exclude_number=int(row["query_number"]))
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        retrieved = [r["number"] for r in results]
        orig = int(row["original_number"])

        for k in K_VALUES:
            recalls[k].append(recall_at_k(retrieved, orig, k))
        mrr_scores.append(reciprocal_rank(retrieved, orig))

    arr_lat = np.array(latencies_ms)
    result = {
        "repo": repo,
        "model": detector.model_key,
        "n_pairs": len(repo_gold),
        "mrr": float(np.mean(mrr_scores)),
        "latency_p50_ms": float(np.percentile(arr_lat, 50)),
        "latency_p95_ms": float(np.percentile(arr_lat, 95)),
        "latency_mean_ms": float(arr_lat.mean()),
    }
    for k in K_VALUES:
        result[f"recall_at_{k}"] = float(np.mean(recalls[k]))
        log.info("[%s/%s] Recall@%d = %.3f", repo, detector.model_key, k, result[f"recall_at_{k}"])

    log.info("[%s/%s] MRR=%.3f  p50=%.1fms  p95=%.1fms",
             repo, detector.model_key, result["mrr"],
             result["latency_p50_ms"], result["latency_p95_ms"])
    return result


def latency_single_sample_benchmark(detector: DuplicateDetector, sample_texts: list[str], n=100) -> dict:
    """Single-query latency (p50/p95) without batch encoding."""
    import random
    rng = random.Random(42)
    times = []
    for _ in range(n):
        text = rng.choice(sample_texts)
        t0 = time.perf_counter()
        detector.retrieve(text, k=5)
        times.append((time.perf_counter() - t0) * 1000)
    arr = np.array(times)
    return {"p50_ms": float(np.percentile(arr, 50)), "p95_ms": float(np.percentile(arr, 95))}


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_recall_at_k(all_results: list[dict], out_path: str) -> None:
    repos = sorted({r["repo"] for r in all_results})
    models = sorted({r["model"] for r in all_results})
    colors = {"bge": "steelblue", "minilm": "darkorange"}

    fig, axes = plt.subplots(1, len(repos), figsize=(6 * len(repos), 5), sharey=True)
    if len(repos) == 1:
        axes = [axes]

    for ax, repo in zip(axes, repos):
        for model in models:
            row = next((r for r in all_results if r["repo"] == repo and r["model"] == model), None)
            if row is None:
                continue
            ks = [k for k in K_VALUES if k <= 20]
            vals = [row.get(f"recall_at_{k}", 0) for k in ks]
            ax.plot(ks, vals, "o-", color=colors.get(model, "gray"), label=model.upper(), linewidth=2)
            for k, v in zip(ks, vals):
                ax.annotate(f"{v:.2f}", (k, v), textcoords="offset points",
                            xytext=(0, 6), ha="center", fontsize=8)

        ax.set(
            title=repo.replace("_", "/"),
            xlabel="K",
            ylabel="Recall@K",
            xticks=K_VALUES,
            ylim=(0, 1.05),
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.4)

    plt.suptitle("Duplicate Retrieval — Recall@K", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved recall@K chart to %s", out_path)


def plot_score_distribution(
    detector: DuplicateDetector,
    gold_df: pd.DataFrame,
    repo: str,
    out_path: str,
    n_neg_sample: int = 500,
) -> None:
    """Score distribution: true-positive (gold pairs) vs negative (random pairs)."""
    repo_gold = gold_df[gold_df["repo"] == repo].reset_index(drop=True)
    if len(repo_gold) == 0:
        return

    # True-positive scores: query vs original
    tp_scores = []
    for _, row in repo_gold.iterrows():
        qtext = str(row["query_title"]) + ". " + str(row["query_body"])[:512]
        results = detector.retrieve(qtext, k=50, exclude_number=int(row["query_number"]))
        for r in results:
            if r["number"] == int(row["original_number"]):
                tp_scores.append(r["score"])
                break

    # Negative scores: random pairs not in gold set
    gold_pairs_set = set(zip(gold_df["query_number"], gold_df["original_number"]))
    rng = np.random.default_rng(42)
    n_idx = len(detector.issue_numbers)
    neg_scores = []
    attempts = 0
    while len(neg_scores) < n_neg_sample and attempts < n_neg_sample * 10:
        attempts += 1
        i, j = rng.integers(0, n_idx, size=2)
        if i == j:
            continue
        ni, nj = int(detector.issue_numbers[i]), int(detector.issue_numbers[j])
        if (ni, nj) in gold_pairs_set or (nj, ni) in gold_pairs_set:
            continue
        # Score via inner product
        ei = detector.index.reconstruct(int(i))
        ej = detector.index.reconstruct(int(j))
        neg_scores.append(float(np.dot(ei, ej)))

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 40)
    if tp_scores:
        ax.hist(tp_scores, bins=bins, alpha=0.7, color="steelblue", label=f"True positive (n={len(tp_scores)})")
    if neg_scores:
        ax.hist(neg_scores, bins=bins, alpha=0.7, color="gray", label=f"Random negative (n={len(neg_scores)})")
    ax.set(
        xlabel="Cosine similarity",
        ylabel="Count",
        title=f"Similarity score distribution — {repo.replace('_', '/')} ({detector.model_key.upper()})",
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved score distribution chart to %s", out_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_repo_model(repo: str, model_key: str, gold_df: pd.DataFrame) -> dict:
    df = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")

    index_dir = str(MODELS_DIR / f"dup_index_{repo}_{model_key}")
    index_path = Path(index_dir) / "index.faiss"

    if index_path.exists():
        log.info("[%s/%s] Loading cached index from %s", repo, model_key, index_dir)
        detector = DuplicateDetector.load(index_dir)
    else:
        detector = DuplicateDetector(repo=repo, model_key=model_key)
        t0 = time.perf_counter()
        detector.build_index(df)
        log.info("[%s/%s] Index built in %.1fs", repo, model_key, time.perf_counter() - t0)
        detector.save(index_dir)

    # Evaluate on gold pairs
    eval_result = evaluate_detector(detector, gold_df, repo)

    # Single-sample latency benchmark
    sample_texts = _build_text(df["title"].head(200), df["body_clean"].head(200))
    lat = latency_single_sample_benchmark(detector, sample_texts, n=100)
    eval_result["latency_single_p50_ms"] = lat["p50_ms"]
    eval_result["latency_single_p95_ms"] = lat["p95_ms"]
    log.info("[%s/%s] Single-query p50=%.1fms  p95=%.1fms", repo, model_key,
             lat["p50_ms"], lat["p95_ms"])

    # Score distribution chart
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_score_distribution(
        detector, gold_df, repo,
        str(CHARTS_DIR / f"duplicate_score_dist_{repo}_{model_key}.png"),
    )

    # Index disk size
    idx_size_mb = sum(f.stat().st_size for f in Path(index_dir).rglob("*") if f.is_file()) / 1e6
    eval_result["index_size_mb"] = round(idx_size_mb, 1)
    log.info("[%s/%s] Index size: %.1f MB", repo, model_key, idx_size_mb)

    return eval_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS
    models = args.models or DEFAULT_MODELS

    gold_df = pd.read_parquet(GOLD_PATH)
    log.info("Gold pairs loaded: %d total", len(gold_df))

    all_results = []
    total_t = time.perf_counter()

    for repo in repos:
        if not (PROCESSED_DIR / f"issues_{repo}.parquet").exists():
            log.warning("Skipping %s — data not found", repo)
            continue
        for model_key in models:
            result = run_repo_model(repo, model_key, gold_df)
            if result:
                all_results.append(result)

    # Recall@K chart (all models + repos together)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_recall_at_k(all_results, str(CHARTS_DIR / "recall_at_k_curve.png"))

    # Save results JSON
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved results to %s", RESULTS_PATH)

    # Summary
    total_elapsed = time.perf_counter() - total_t
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("%-30s %-8s  R@1    R@5   R@10   R@20   MRR   p50(ms)",
             "Repo", "Model")
    for r in all_results:
        log.info("%-30s %-8s  %.3f  %.3f  %.3f  %.3f  %.3f  %.1f",
                 r["repo"], r["model"],
                 r.get("recall_at_1", 0), r.get("recall_at_5", 0),
                 r.get("recall_at_10", 0), r.get("recall_at_20", 0),
                 r.get("mrr", 0), r.get("latency_single_p50_ms", 0))
    log.info("Total time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)


if __name__ == "__main__":
    main()
