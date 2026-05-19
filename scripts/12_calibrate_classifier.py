"""Fit temperature-scaling calibrators for component classifiers.

Loads existing trained classifier .pkl, finds the optimal temperature T
that minimises NLL on the val split, then writes a TemperatureScaler
back into the same .pkl.

Hard stops (revised thresholds from W1.2 diagnostic):
  - val ECE after calibration >= 0.18 → abort
  - test accuracy regresses > 1pp vs uncalibrated → abort

Usage:
    python scripts/12_calibrate_classifier.py
    python scripts/12_calibrate_classifier.py --repos microsoft_vscode
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage_iq.models.component_classifier import (
    TFIDFComponentClassifier,
    TemperatureScaler,
    _build_text,
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
REPORTS_DIR = Path("reports")

ECE_THRESHOLD = 0.18
ACC_REGRESS_MAX_PP = 1.0


def _compute_ece(y_enc: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    max_p = proba.max(axis=1)
    y_pred = proba.argmax(axis=1)
    correct = (y_pred == y_enc).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_enc)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (max_p >= lo) & (max_p < hi)
        if not mask.any():
            continue
        ece += mask.sum() * abs(correct[mask].mean() - max_p[mask].mean())
    return float(ece / max(n, 1))


def run_repo(repo: str) -> dict:
    log.info("=" * 60)
    log.info("Repo: %s", repo)

    model_path = MODELS_DIR / f"component_classifier_{repo}.pkl"
    if not model_path.exists():
        log.error("Model not found: %s", model_path)
        sys.exit(1)

    clf = TFIDFComponentClassifier.load(str(model_path))
    assert clf.pipeline is not None
    assert clf.label_encoder is not None

    val = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_test.parquet")
    X_val = _build_text(val["title"], val["body_clean"])
    X_test = _build_text(test["title"], test["body_clean"])
    y_val_enc = clf.label_encoder.transform(val["component"])
    y_test_enc = clf.label_encoder.transform(test["component"])

    # Baseline (uncalibrated) ECE and accuracy
    proba_raw_val = clf.pipeline.predict_proba(X_val)
    proba_raw_test = clf.pipeline.predict_proba(X_test)
    ece_before = _compute_ece(y_val_enc, proba_raw_val)
    acc_before_test = float((proba_raw_test.argmax(axis=1) == y_test_enc).mean())

    log.info("Before: val_ECE=%.4f  test_acc=%.4f", ece_before, acc_before_test)

    # Temperature scaling: minimise NLL on val logits
    X_tfidf_val = clf.pipeline["tfidf"].transform(X_val)
    X_tfidf_test = clf.pipeline["tfidf"].transform(X_test)
    logits_val = clf.pipeline["lr"].decision_function(X_tfidf_val)
    logits_test = clf.pipeline["lr"].decision_function(X_tfidf_test)

    def nll(T: float) -> float:
        scaled = softmax(logits_val / T, axis=1)
        log_p = np.log(scaled[np.arange(len(y_val_enc)), y_val_enc] + 1e-15)
        return float(-log_p.mean())

    opt = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
    T_opt = float(opt.x)

    proba_cal_val = softmax(logits_val / T_opt, axis=1)
    proba_cal_test = softmax(logits_test / T_opt, axis=1)
    ece_after = _compute_ece(y_val_enc, proba_cal_val)
    acc_after_test = float((proba_cal_test.argmax(axis=1) == y_test_enc).mean())
    acc_delta_pp = (acc_after_test - acc_before_test) * 100

    log.info("After:  val_ECE=%.4f  test_acc=%.4f  T_opt=%.4f  Δacc=%.4fpp",
             ece_after, acc_after_test, T_opt, acc_delta_pp)

    # Hard stops
    if ece_after >= ECE_THRESHOLD:
        log.error("HARD STOP: val ECE %.4f >= threshold %.2f", ece_after, ECE_THRESHOLD)
        sys.exit(1)
    if acc_delta_pp < -ACC_REGRESS_MAX_PP:
        log.error("HARD STOP: test acc regressed %.2fpp (threshold -%s pp)", acc_delta_pp, ACC_REGRESS_MAX_PP)
        sys.exit(1)

    log.info("Hard stops passed (ECE=%.4f < %.2f; acc_delta=%.4fpp > -%.1fpp)",
             ece_after, ECE_THRESHOLD, acc_delta_pp, ACC_REGRESS_MAX_PP)

    # Attach calibrator and re-save
    clf.calibrator = TemperatureScaler(pipeline=clf.pipeline, T=T_opt)
    clf.save(str(model_path))
    log.info("TemperatureScaler(T=%.4f) saved to %s", T_opt, model_path)

    return {
        "repo": repo,
        "T_opt": round(T_opt, 4),
        "ece_before_val": round(ece_before, 4),
        "ece_after_val": round(ece_after, 4),
        "acc_test_before": round(acc_before_test, 4),
        "acc_test_after": round(acc_after_test, 4),
        "acc_test_delta_pp": round(acc_delta_pp, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=None)
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS
    all_results: dict[str, dict] = {}
    for repo in repos:
        if not (PROCESSED_DIR / f"{repo}_classifier_train.parquet").exists():
            log.warning("Skipping %s — splits not found", repo)
            continue
        all_results[repo] = run_repo(repo)

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("%-30s  %6s  %8s  %8s  %8s  %8s",
             "Repo", "T_opt", "ECE_bef", "ECE_aft", "acc_bef", "acc_aft")
    for repo, r in all_results.items():
        log.info("%-30s  %6.4f  %8.4f  %8.4f  %8.4f  %8.4f",
                 repo, r["T_opt"], r["ece_before_val"], r["ece_after_val"],
                 r["acc_test_before"], r["acc_test_after"])

    out = REPORTS_DIR / "calibration_results.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Saved to %s", out)


if __name__ == "__main__":
    main()
