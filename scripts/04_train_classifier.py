"""Train and evaluate TF-IDF component classifiers per repo.

Usage:
    python scripts/04_train_classifier.py
    python scripts/04_train_classifier.py --repos microsoft_vscode
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from triage_iq.models.component_classifier import TFIDFComponentClassifier, _build_text
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


def run_repo(repo: str, eval_only: bool = False) -> dict:
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

    log.info(
        "Split sizes — train=%d  val=%d  test=%d  classes=%d",
        len(train), len(val), len(test), y_train.nunique(),
    )

    model_path = MODELS_DIR / f"component_classifier_{repo}.pkl"

    if eval_only:
        # LogisticRegression(solver="saga") has no fixed random_state, so re-fitting
        # is NOT reproducible -- it silently swaps the shipped model for a different
        # one. --eval-only loads the existing artifact instead, for report/metric
        # regeneration that must not touch the deployed model.
        #
        # This script trains/represents the single-label architecture specifically
        # (see the `else` branch below) -- component_classifier_{repo}.pkl IS the
        # multi-label model post-ADR-0036, so --eval-only now loads the archived
        # single-label baseline for consistency with what this script actually trains.
        eval_only_path = MODELS_DIR / f"component_classifier_{repo}_PRE_MULTILABEL.pkl"
        log.info("--eval-only: loading existing model from %s (not retraining)", eval_only_path)
        clf = TFIDFComponentClassifier.load(str(eval_only_path))
        train_time = 0.0
    else:
        # ── Train ─────────────────────────────────────────────────
        clf = TFIDFComponentClassifier(repo=repo)
        t_train = time.perf_counter()
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        train_time = time.perf_counter() - t_train

        # ── Save model ────────────────────────────────────────────
        clf.save(str(model_path))

    # ── Evaluate on test ──────────────────────────────────────────
    # repo/labels_raw enable top-3 (the product's own definition of "correct" — see
    # grounding.py::verify_plan_grounding) and multi-label credit accuracy, alongside
    # top-1.
    t_eval = time.perf_counter()
    results = evaluate_classifier(clf, X_test, y_test, repo=repo, labels_raw=test["labels_raw"])
    eval_time = time.perf_counter() - t_eval

    log.info(
        "Test — top1=%.3f %s  top3=%.3f %s  macro_f1=%.3f  weighted_f1=%.3f",
        results["top1_accuracy"], results["top1_accuracy_ci95"],
        results["top3_accuracy"], results["top3_accuracy_ci95"],
        results["macro_f1"], results["weighted_f1"],
    )
    log.info(
        "Multi-label credit=%.3f %s  (%d/%d test rows have >1 valid component label)",
        results["multi_label_credit_accuracy"], results["multi_label_credit_accuracy_ci95"],
        results["n_multi_label_test_rows"], len(test),
    )

    # ── Latency benchmarks ────────────────────────────────────────
    lat_single = latency_benchmark(clf, X_test, n_iters=200)
    lat_batch  = batch_latency_benchmark(clf, X_test, batch_size=100)
    log.info(
        "Latency — single p50=%.2fms p95=%.2fms  batch/sample=%.3fms  %d pred/sec",
        lat_single["p50_ms"], lat_single["p95_ms"],
        lat_batch["per_sample_ms"], int(lat_batch["samples_per_sec"]),
    )

    # ── Calibration ───────────────────────────────────────────────
    cal = calibration_analysis(y_test, results["y_proba"], results["classes"],
                               clf.label_encoder)
    log.info("Calibration — ECE=%.4f  mean_conf=%.3f  overconfident=%s",
             cal["ece"], cal["mean_confidence"], cal["overconfident"])

    # ── Plots ─────────────────────────────────────────────────────
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion_heatmap(
        results["confusion_matrix"], results["classes"], repo,
        str(CHARTS_DIR / f"classifier_confusion_{repo}.png"),
    )
    plot_per_class_f1(
        results["per_class_metrics"], repo,
        str(CHARTS_DIR / f"classifier_per_class_f1_{repo}.png"),
    )
    plot_calibration(
        cal, repo,
        str(CHARTS_DIR / f"classifier_calibration_{repo}.png"),
    )

    # ── Top confusions ────────────────────────────────────────────
    log.info("Top-10 confusion pairs:")
    for true_l, pred_l, count in results["top_confusions"][:10]:
        log.info("  %s → %s  (%d)", true_l, pred_l, count)

    # ── Per-class highlights ──────────────────────────────────────
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
        "n_classes": y_train.nunique(),
        "accuracy": results["accuracy"],  # deprecated alias for top1_accuracy — kept for callers
        "top1_accuracy": results["top1_accuracy"],
        "top1_accuracy_ci95": results["top1_accuracy_ci95"],
        "top3_accuracy": results["top3_accuracy"],
        "top3_accuracy_ci95": results["top3_accuracy_ci95"],
        "multi_label_credit_accuracy": results["multi_label_credit_accuracy"],
        "multi_label_credit_accuracy_ci95": results["multi_label_credit_accuracy_ci95"],
        "n_multi_label_test_rows": results["n_multi_label_test_rows"],
        "multi_label_test_row_rate": results["multi_label_test_row_rate"],
        "macro_f1": results["macro_f1"],
        "weighted_f1": results["weighted_f1"],
        "train_time_s": round(train_time, 1),
        "eval_time_s": round(eval_time, 2),
        "latency_single": lat_single,
        "latency_batch": lat_batch,
        "calibration": cal,
        "top_confusions": results["top_confusions"][:10],
        "best_classes": [(c, round(class_f1s[c], 3)) for c in best_classes],
        "worst_classes": [(c, round(class_f1s[c], 3)) for c in worst_classes],
        "per_class_f1": {k: round(v, 3) for k, v in class_f1s.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Load the existing shipped model and re-run evaluation only -- does not "
             "retrain or overwrite data/models/component_classifier_{repo}.pkl. Use this "
             "for regenerating reports/classifier_results.json (LogisticRegression's saga "
             "solver has no fixed seed, so a plain re-run is NOT reproducible and would "
             "silently swap the deployed model).",
    )
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS
    all_results = {}
    total_t = time.perf_counter()

    for repo in repos:
        if not (PROCESSED_DIR / f"{repo}_classifier_train.parquet").exists():
            log.warning("Skipping %s — classifier splits not found", repo)
            continue
        all_results[repo] = run_repo(repo, eval_only=args.eval_only)

    total_elapsed = time.perf_counter() - total_t

    # ── Save results JSON (for report generation) ─────────────────
    out = Path("reports/classifier_results.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Saved eval results to %s", out)

    # ── Summary ───────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info(
        "%-30s  %8s  %8s  %8s  %8s  %10s",
        "Repo", "Top1", "Top3", "MacroF1", "WtdF1", "p50 (ms)",
    )
    for repo, r in all_results.items():
        log.info(
            "%-30s  %8.3f  %8.3f  %8.3f  %8.3f  %10.2f",
            repo, r["top1_accuracy"], r["top3_accuracy"], r["macro_f1"], r["weighted_f1"],
            r["latency_single"]["p50_ms"],
        )
    log.info("Total time: %.1fs", total_elapsed)


if __name__ == "__main__":
    main()
