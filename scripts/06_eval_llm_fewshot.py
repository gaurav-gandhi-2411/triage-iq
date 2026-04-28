"""Evaluate LLM few-shot classification (Groq Llama 3.1 8B) on component labels.

Samples 200 test issues per repo. Requires GROQ_API_KEY in environment or .env.

Usage:
    export GROQ_API_KEY=gsk_...
    python scripts/06_eval_llm_fewshot.py
    python scripts/06_eval_llm_fewshot.py --repos microsoft_vscode --n-samples 100
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from triage_iq.models.component_classifier import _build_text
from triage_iq.models.llm_classifier import run_llm_fewshot_eval

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]
PROCESSED_DIR = Path("data/processed")
CHARTS_DIR = Path("reports/charts")
RESULTS_PATH = Path("reports/llm_fewshot_results.json")

TFIDF_BASELINE = {
    "microsoft_vscode": {"accuracy": 0.690, "macro_f1": 0.585},
    "kubernetes_kubernetes": {"accuracy": 0.514, "macro_f1": 0.466},
}


def plot_llm_per_class_f1(per_class_f1: dict, repo: str, out_path: str) -> None:
    labels = list(per_class_f1.keys())
    f1s = [per_class_f1[l] for l in labels]
    order = np.argsort(f1s)
    labels = [labels[i] for i in order]
    f1s = [f1s[i] for i in order]

    fig, ax = plt.subplots(figsize=(8, max(5, len(labels) * 0.3)))
    ax.barh(range(len(labels)), f1s, color="darkorange")
    ax.set(
        yticks=range(len(labels)),
        yticklabels=labels,
        xlabel="F1 score",
        title=f"LLM Few-shot Per-class F1 — {repo.replace('_', '/')}",
        xlim=(0, 1),
    )
    ax.axvline(np.mean(f1s), color="red", linestyle="--", alpha=0.6,
               label=f"macro avg={np.mean(f1s):.3f}")
    ax.legend(fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved LLM per-class F1 chart to %s", out_path)


def run_repo(repo: str, n_samples: int) -> dict:
    log.info("=" * 60)
    log.info("LLM few-shot eval — Repo: %s  n_samples=%d", repo, n_samples)

    train = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_train.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_test.parquet")

    X_train = _build_text(train["title"], train["body_clean"])
    X_test  = _build_text(test["title"],  test["body_clean"])
    y_train = train["component"]
    y_test  = test["component"]

    candidate_labels = sorted(y_train.unique().tolist())

    t0 = time.perf_counter()
    results = run_llm_fewshot_eval(
        X_test, y_test, X_train, y_train,
        candidate_labels=candidate_labels,
        n_samples=n_samples,
        n_few_shot=5,
        seed=42,
    )
    elapsed = time.perf_counter() - t0

    log.info(
        "Results — accuracy=%.3f  macro_f1=%.3f  weighted_f1=%.3f  "
        "parsed=%d/%d  time=%.1fs",
        results["accuracy"], results["macro_f1"], results["weighted_f1"],
        results["n_parsed"], results["n_samples"], elapsed,
    )

    baseline = TFIDF_BASELINE.get(repo, {})
    delta_acc = results["accuracy"] - baseline.get("accuracy", 0)
    delta_f1  = results["macro_f1"] - baseline.get("macro_f1", 0)
    log.info("vs TF-IDF — Δacc=%.3f  Δmacro_f1=%.3f", delta_acc, delta_f1)

    # Chart
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    if results["per_class_f1"]:
        plot_llm_per_class_f1(
            results["per_class_f1"], repo,
            str(CHARTS_DIR / f"llm_fewshot_per_class_f1_{repo}.png"),
        )

    results["elapsed_s"] = round(elapsed, 1)
    results["vs_tfidf"] = {"delta_accuracy": round(delta_acc, 4), "delta_macro_f1": round(delta_f1, 4)}
    results["repo"] = repo
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--n-samples", type=int, default=200)
    args = parser.parse_args()

    if not os.environ.get("GROQ_API_KEY"):
        log.error(
            "GROQ_API_KEY not set.\n"
            "Add to .env file:  GROQ_API_KEY=gsk_...\n"
            "Or export: export GROQ_API_KEY=gsk_..."
        )
        sys.exit(1)

    repos = args.repos or DEFAULT_REPOS
    all_results = {}
    total_t = time.perf_counter()

    for repo in repos:
        if not (PROCESSED_DIR / f"{repo}_classifier_test.parquet").exists():
            log.warning("Skipping %s — splits not found", repo)
            continue
        all_results[repo] = run_repo(repo, n_samples=args.n_samples)

    total_elapsed = time.perf_counter() - total_t

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Saved LLM results to %s", RESULTS_PATH)

    log.info("=" * 60)
    log.info("SUMMARY (n=%d sample per repo)", args.n_samples)
    log.info("%-30s  %8s  %8s  %8s", "Repo", "Accuracy", "MacroF1", "ΔF1vsTFIDF")
    for repo, r in all_results.items():
        log.info("%-30s  %8.3f  %8.3f  %8.3f",
                 repo, r["accuracy"], r["macro_f1"],
                 r["vs_tfidf"]["delta_macro_f1"])
    log.info("Total time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)


if __name__ == "__main__":
    main()
