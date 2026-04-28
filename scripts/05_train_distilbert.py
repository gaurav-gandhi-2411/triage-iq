"""Fine-tune DistilBERT per-repo and evaluate vs TF-IDF baseline.

Training uses GPU if available. CPU latency is benchmarked separately
(production target — no GPU assumed in serving).

Usage:
    python scripts/05_train_distilbert.py
    python scripts/05_train_distilbert.py --repos microsoft_vscode --epochs 5
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch

from triage_iq.models.component_classifier import _build_text
from triage_iq.models.distilbert_classifier import DistilBERTComponentClassifier
from triage_iq.evaluation.classifier_eval import (
    evaluate_classifier,
    latency_benchmark,
    batch_latency_benchmark,
    calibration_analysis,
    plot_confusion_heatmap,
    plot_per_class_f1,
    plot_calibration,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
CHARTS_DIR = Path("reports/charts")
RESULTS_PATH = Path("reports/distilbert_results.json")

# Existing TF-IDF baseline for comparison table
TFIDF_BASELINE = {
    "microsoft_vscode": {"accuracy": 0.690, "macro_f1": 0.585, "latency_p50_ms": 4.91},
    "kubernetes_kubernetes": {"accuracy": 0.514, "macro_f1": 0.466, "latency_p50_ms": 5.57},
}


def run_repo(repo: str, epochs: int) -> dict:
    log.info("=" * 60)
    log.info("Repo: %s", repo)

    # ── Load splits ──────────────────────────────────────────────
    train = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_val.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_test.parquet")

    X_train = _build_text(train["title"], train["body_clean"])
    X_val   = _build_text(val["title"],   val["body_clean"])
    X_test  = _build_text(test["title"],  test["body_clean"])

    y_train = train["component"]
    y_val   = val["component"]
    y_test  = test["component"]

    n_classes = y_train.nunique()
    log.info("Split sizes — train=%d  val=%d  test=%d  classes=%d",
             len(train), len(val), len(test), n_classes)

    # ── Train ─────────────────────────────────────────────────────
    model_dir = str(MODELS_DIR / f"distilbert_component_{repo}")
    clf = DistilBERTComponentClassifier(repo=repo, num_labels=n_classes)

    t_train = time.perf_counter()
    clf.fit(X_train, y_train, X_val, y_val, epochs=epochs, output_dir=model_dir)
    train_time = time.perf_counter() - t_train
    log.info("Train time: %.1fs", train_time)

    # ── GPU evaluation ────────────────────────────────────────────
    device_label = "GPU" if torch.cuda.is_available() else "CPU"
    log.info("Evaluating on %s...", device_label)
    if torch.cuda.is_available():
        clf.to_gpu()

    t_eval = time.perf_counter()
    results = evaluate_classifier(clf, X_test, y_test)
    eval_time = time.perf_counter() - t_eval

    log.info(
        "Test (%s) — accuracy=%.3f  macro_f1=%.3f  weighted_f1=%.3f",
        device_label, results["accuracy"], results["macro_f1"], results["weighted_f1"],
    )

    # ── CPU latency benchmark (production target) ─────────────────
    log.info("Benchmarking CPU latency (moving model to CPU)...")
    clf.to_cpu()
    lat_single = latency_benchmark(clf, X_test, n_iters=100)
    lat_batch  = batch_latency_benchmark(clf, X_test, batch_size=100)
    log.info(
        "CPU Latency — single p50=%.1fms p95=%.1fms  batch/sample=%.1fms  %d pred/sec",
        lat_single["p50_ms"], lat_single["p95_ms"],
        lat_batch["per_sample_ms"], int(lat_batch["samples_per_sec"]),
    )

    # ── Calibration ───────────────────────────────────────────────
    cal = calibration_analysis(y_test, results["y_proba"], results["classes"],
                               clf.label_encoder)
    log.info("Calibration — ECE=%.4f  mean_conf=%.3f  overconfident=%s",
             cal["ece"], cal["mean_confidence"], cal["overconfident"])

    # ── Charts ────────────────────────────────────────────────────
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_heatmap(
        results["confusion_matrix"], results["classes"], repo,
        str(CHARTS_DIR / f"distilbert_confusion_{repo}.png"),
    )
    plot_per_class_f1(
        results["per_class_metrics"], repo,
        str(CHARTS_DIR / f"distilbert_per_class_f1_{repo}.png"),
    )
    plot_calibration(
        cal, repo,
        str(CHARTS_DIR / f"distilbert_calibration_{repo}.png"),
    )

    # ── vs baseline comparison ────────────────────────────────────
    baseline = TFIDF_BASELINE.get(repo, {})
    delta_acc  = results["accuracy"]  - baseline.get("accuracy",  0)
    delta_f1   = results["macro_f1"]  - baseline.get("macro_f1",  0)
    log.info(
        "vs TF-IDF — Δacc=%.3f  Δmacro_f1=%.3f  latency: %.0fms vs %.0fms",
        delta_acc, delta_f1,
        lat_single["p50_ms"], baseline.get("latency_p50_ms", 0),
    )

    # ── Per-class summary ─────────────────────────────────────────
    pcm = results["per_class_metrics"]
    class_f1s = {
        k: v["f1-score"]
        for k, v in pcm.items()
        if k not in ("accuracy", "macro avg", "weighted avg")
    }
    best_classes  = sorted(class_f1s, key=lambda k: -class_f1s[k])[:5]
    worst_classes = sorted(class_f1s, key=lambda k:  class_f1s[k])[:5]

    return {
        "repo": repo,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "n_classes": n_classes,
        "accuracy": results["accuracy"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 2),
        "latency_single_cpu": lat_single,
        "latency_batch_cpu": lat_batch,
        "calibration": cal,
        "top_confusions": results["top_confusions"][:10],
        "best_classes": [(c, round(class_f1s[c], 3)) for c in best_classes],
        "worst_classes": [(c, round(class_f1s[c], 3)) for c in worst_classes],
        "per_class_f1": {k: round(v, 3) for k, v in class_f1s.items()},
        "vs_tfidf": {
            "delta_accuracy": round(delta_acc, 4),
            "delta_macro_f1": round(delta_f1, 4),
            "latency_ratio": round(lat_single["p50_ms"] / baseline.get("latency_p50_ms", 1), 1),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=8)
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS
    all_results = {}
    total_t = time.perf_counter()

    for repo in repos:
        if not (PROCESSED_DIR / f"{repo}_classifier_train.parquet").exists():
            log.warning("Skipping %s — splits not found", repo)
            continue
        all_results[repo] = run_repo(repo, epochs=args.epochs)

    total_elapsed = time.perf_counter() - total_t

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Saved results to %s", RESULTS_PATH)

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("%-30s  %8s  %8s  %10s  %10s", "Repo", "Accuracy", "MacroF1", "p50-CPU(ms)", "ΔF1vsTFIDF")
    for repo, r in all_results.items():
        log.info(
            "%-30s  %8.3f  %8.3f  %10.1f  %10.3f",
            repo, r["accuracy"], r["macro_f1"],
            r["latency_single_cpu"]["p50_ms"],
            r["vs_tfidf"]["delta_macro_f1"],
        )
    log.info("Total time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)


if __name__ == "__main__":
    main()
