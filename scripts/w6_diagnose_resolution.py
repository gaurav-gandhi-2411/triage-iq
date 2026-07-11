"""Bias-vs-variance diagnosis of the CURRENT resolution model (W6, ADR-0025).

ADR-0021/0023 established that this model's uncertainty (conformal interval width) is not
diagnostic of its own errors. This script answers the prior question: is the error itself
high-BIAS (systematic, fixable with better features) or high-VARIANCE (irreducible noise,
in which case task reframing — already partially done via the bucket classifier, ADR-0009 —
is the right direction, not more feature engineering)?

Method: reload the CURRENTLY SHIPPED resolution_predictor_{slug}.pkl (no retraining), replay
it over the held-out temporal test set (same split used to produce reports/resolution_results.json),
and decompose the residual:
  - Per-bucket bias: mean(predicted - actual) within each true bucket. Large systematic bias
    (all issues in a bucket predicted too high/low by a consistent amount) is fixable with
    features that shift the systematic offset.
  - Per-bucket variance: std(predicted - actual) within each true bucket. High spread with
    near-zero mean bias means the model can't distinguish WHICH issue in a bucket will resolve
    fast vs. slow -- that's irreducible from available signal, not a feature-engineering gap.
  - Oracle-bucket comparison: if we replaced the point prediction with the TRUE bucket's median
    (an oracle that magically knows the right bucket), how much MAE would that recover? This
    isolates between-bucket error (which bucket) from within-bucket error (where in the bucket).

Also evaluates the ALREADY-EXISTING bucket classifier (ADR-0009, currently gated on an absolute
obo>=60% threshold, never tested with a proper confidence interval) against a naive majority-class
baseline, bootstrapped, using the SAME ADR-0006 bar (CI excludes zero) this project holds every
other improvement to.

Zero-cost, no retraining -- pure evaluation of already-fitted, already-shipped artifacts.

Usage:
    python scripts/w6_diagnose_resolution.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from triage_iq.models.resolution import BUCKET_LABELS, ResolutionTimePredictor, engineer_features, hours_to_bucket

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPORT_PATH = Path("reports/w6_resolution_diagnosis.json")

REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]
N_BOOTSTRAP = 2000
SEED = 42


def _load_embeddings_from_index(repo: str, df: pd.DataFrame) -> np.ndarray | None:
    """Mirrors scripts/09_train_resolution.py:load_embeddings_from_index exactly."""
    import faiss
    import joblib as jl

    index_dir = MODELS_DIR / f"dup_index_{repo}_bge"
    if not (index_dir / "index.faiss").exists():
        return None
    meta = jl.load(str(index_dir / "meta.pkl"))
    index = faiss.read_index(str(index_dir / "index.faiss"))
    num_to_faiss_idx = {int(n): i for i, n in enumerate(meta["issue_numbers"])}

    dim = index.d
    embs = np.zeros((len(df), dim), dtype=np.float32)
    for row_pos, num in enumerate(df["number"]):
        faiss_idx = num_to_faiss_idx.get(int(num))
        if faiss_idx is not None:
            embs[row_pos] = index.reconstruct(int(faiss_idx))
    return embs


def _bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    """Percentile bootstrap 95% CI on the mean of `values`."""
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95_lower": float(lo), "ci95_upper": float(hi)}


def raw_bucket_accuracy_vs_naive(
    predictor: ResolutionTimePredictor, X_test: pd.DataFrame, true_bucket_idx: np.ndarray
) -> dict:
    """Evaluate the RAW trained bucket classifier against the naive majority-class prior.

    Deliberately calls predictor.model_bucket.predict(X) directly, NOT
    predictor.predict_bucket() -- that method applies the BUCKET_CLASSIFIER_TRUSTED gate
    ADR-0025 itself introduced. For an untrusted repo (e.g. vscode), predict_bucket()
    already returns the naive fallback, so calling it here would measure the naive
    prediction against itself (a tautological 0.0pp delta) and this diagnostic could
    never again re-verify the classifier the trust decision depends on.
    """
    assert predictor.model_bucket is not None, f"{predictor.repo}: no trained bucket classifier to diagnose"
    raw_proba = np.asarray(predictor.model_bucket.predict(X_test))  # shape (n, 5)
    pred_bucket_idx = raw_proba.argmax(axis=1)
    correct = (pred_bucket_idx == true_bucket_idx).astype(float)
    obo_correct = (np.abs(pred_bucket_idx - true_bucket_idx) <= 1).astype(float)

    majority_bucket = max(predictor.bucket_train_distribution, key=predictor.bucket_train_distribution.get)
    majority_idx = BUCKET_LABELS.index(majority_bucket)
    naive_correct = (true_bucket_idx == majority_idx).astype(float)
    naive_obo_correct = (np.abs(majority_idx - true_bucket_idx) <= 1).astype(float)

    acc_delta = correct - naive_correct  # paired per-issue delta -> bootstrap this directly
    obo_delta = obo_correct - naive_obo_correct

    return {
        "n": len(true_bucket_idx),
        "trained_accuracy": round(float(correct.mean()), 4),
        "naive_accuracy": round(float(naive_correct.mean()), 4),
        "accuracy_delta_bootstrap": _bootstrap_ci(acc_delta),
        "trained_obo": round(float(obo_correct.mean()), 4),
        "naive_obo": round(float(naive_obo_correct.mean()), 4),
        "obo_delta_bootstrap": _bootstrap_ci(obo_delta),
        "naive_majority_bucket": majority_bucket,
    }


def diagnose_repo(repo: str) -> dict:
    predictor = ResolutionTimePredictor.load(str(MODELS_DIR / f"resolution_predictor_{repo}.pkl"))

    train = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_test.parquet")
    train = train[train["resolution_hours"] > 0]
    test = test[test["resolution_hours"] > 0].reset_index(drop=True)

    emb_test = _load_embeddings_from_index(repo, test)
    X_test, _ = engineer_features(test, train_df=train, embeddings=emb_test, pca=predictor.pca)
    # Align to the exact feature set/order the model was trained on.
    X_test = X_test[predictor.feature_names]

    y_true_hrs = test["resolution_hours"].values
    y_pred_hrs = predictor.predict(X_test)

    y_true_days = y_true_hrs / 24.0
    y_pred_days = y_pred_hrs / 24.0
    residual_days = y_pred_days - y_true_days  # signed: positive = overestimate
    abs_residual_days = np.abs(residual_days)

    log_true = np.log1p(y_true_hrs)
    log_pred = np.log1p(y_pred_hrs)
    log_residual = log_pred - log_true

    true_bucket_idx = hours_to_bucket(y_true_hrs)

    # ── Per-bucket bias/variance decomposition ──────────────────────
    per_bucket: dict[str, Any] = {}
    for i, label in enumerate(BUCKET_LABELS):
        mask = true_bucket_idx == i
        n = int(mask.sum())
        if n == 0:
            per_bucket[label] = {"n": 0}
            continue
        per_bucket[label] = {
            "n": n,
            "mean_bias_days": round(float(residual_days[mask].mean()), 2),
            "std_days": round(float(residual_days[mask].std()), 2),
            "mae_days": round(float(abs_residual_days[mask].mean()), 2),
            "mean_bias_log": round(float(log_residual[mask].mean()), 4),
            "std_log": round(float(log_residual[mask].std()), 4),
            "actual_median_days": round(float(np.median(y_true_days[mask])), 2),
            "actual_p10_days": round(float(np.percentile(y_true_days[mask], 10)), 2),
            "actual_p90_days": round(float(np.percentile(y_true_days[mask], 90)), 2),
        }

    # ── Oracle-bucket comparison: how much MAE would knowing the TRUE bucket recover? ──
    # Oracle predicts the true bucket's own median (cheating -- it "knows" the bucket).
    # If oracle MAE << model MAE, most error is BETWEEN buckets (bias, fixable by better
    # bucket-level features). If oracle MAE is still large (close to model MAE), most error
    # is WITHIN the bucket (variance, irreducible from bucket-level information).
    bucket_medians_days = {
        i: float(np.median(y_true_days[true_bucket_idx == i])) if (true_bucket_idx == i).any() else np.nan
        for i in range(len(BUCKET_LABELS))
    }
    oracle_pred_days = np.array([bucket_medians_days[b] for b in true_bucket_idx])
    oracle_mae_days = float(np.mean(np.abs(oracle_pred_days - y_true_days)))
    model_mae_days = float(np.mean(abs_residual_days))
    naive_pred_days = float(np.median(train["resolution_hours"].values) / 24.0)
    naive_mae_days = float(np.mean(np.abs(naive_pred_days - y_true_days)))

    bucket_eval = raw_bucket_accuracy_vs_naive(predictor, X_test, true_bucket_idx)

    return {
        "repo": repo,
        "n_test": len(test),
        "point_regression": {
            "model_mae_days": round(model_mae_days, 2),
            "naive_mae_days": round(naive_mae_days, 2),
            "oracle_bucket_mae_days": round(oracle_mae_days, 2),
            "oracle_recovers_pct_of_naive_error": round(
                100 * (naive_mae_days - oracle_mae_days) / naive_mae_days, 1
            ) if naive_mae_days else None,
            "model_recovers_pct_of_naive_error": round(
                100 * (naive_mae_days - model_mae_days) / naive_mae_days, 1
            ) if naive_mae_days else None,
        },
        "per_bucket_residuals": per_bucket,
        "bucket_classifier_vs_naive": bucket_eval,
    }


def main() -> None:
    result = {"repos": {}}
    for repo in REPOS:
        print(f"\n=== {repo} ===")
        r = diagnose_repo(repo)
        result["repos"][repo] = r

        pr = r["point_regression"]
        print(f"  n_test={r['n_test']}")
        print(f"  Point regression MAE: model={pr['model_mae_days']}d naive={pr['naive_mae_days']}d "
              f"oracle(true-bucket-median)={pr['oracle_bucket_mae_days']}d")
        print(f"  Model recovers {pr['model_recovers_pct_of_naive_error']}% of naive's error; "
              f"oracle-bucket-median would recover {pr['oracle_recovers_pct_of_naive_error']}%")

        print("  Per-bucket residuals (mean_bias_days = mean(pred-actual), std_days = spread):")
        for label, b in r["per_bucket_residuals"].items():
            if b["n"] == 0:
                continue
            print(f"    {label:>7}: n={b['n']:>4}  mean_bias={b['mean_bias_days']:>8.2f}d  "
                  f"std={b['std_days']:>8.2f}d  mae={b['mae_days']:>8.2f}d  "
                  f"actual_median={b['actual_median_days']:.1f}d "
                  f"[{b['actual_p10_days']:.1f}, {b['actual_p90_days']:.1f}]")

        bc = r["bucket_classifier_vs_naive"]
        print(f"  Bucket classifier: trained_acc={bc['trained_accuracy']} vs "
              f"naive_acc={bc['naive_accuracy']} (majority={bc['naive_majority_bucket']})")
        print(f"    accuracy delta: {bc['accuracy_delta_bootstrap']}")
        print(f"  Bucket obo: trained={bc['trained_obo']} vs naive={bc['naive_obo']}")
        print(f"    obo delta: {bc['obo_delta_bootstrap']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStructured report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
