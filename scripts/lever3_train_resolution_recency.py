"""LEVER 3, step 3: test recency weighting on top of the re-split (scripts/lever3_resplit_
resolution.py) after the plain re-split retrain (scripts/lever3_train_resolution.py) showed a
NEW problem the re-split alone doesn't fix: vscode's re-split test window (Feb 2025-Apr 2026)
is 83.8% "hours"-bucket issues (median resolution ~1.2h) -- a much sharper recency-driven skew
than train's overall mix (median 2.5d), and the untuned re-split model actually underperforms
the naive baseline on both point MAE (-202%) and bucket accuracy (-1.06pp) as a result. The
model's tree splits reflect the broad historical mix, not the recent fast-resolution regime the
test window actually lives in -- exactly what recency weighting is for.

Exponential recency weight: weight = exp(-ln(2) * age_days / half_life_days), age_days measured
from each training example's created_at to the LATEST created_at in train (not to "now" or to
the test window -- that would leak test-period knowledge into the weighting scheme itself).
Sweeps a few half-lives, selects by validation-set point MAE, reports the selected model's full
metrics (point + bucket vs naive) on the held-out test set -- selection and final reporting are
on different splits, same discipline as the Optuna tuning in scripts/09_train_resolution.py.

Bypasses ResolutionTimePredictor.fit() (no sample_weight support there) and trains directly via
lgb.Dataset(..., weight=...) -- this file is prototyping an experimental lever, not a change to
the shared production class; the class only gets touched if this proves worth shipping.

Reads:  data/processed/{repo}_temporal_{train,val,test}_lever3.parquet
        data/models/d1_full_corpus_index_{repo}_bge/
Writes: reports/lever3_recency_weighting_results.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.resolution import (  # noqa: E402
    BUCKET_LABELS,
    engineer_features,
    hours_to_bucket,
)

sys.path.insert(0, str(Path(__file__).parent))
from lever3_train_resolution import bootstrap_ci, load_embeddings_from_d1_index  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SEED = 42
PROCESSED_DIR = Path("data/processed")
REPORTS = Path("reports")
REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]
HALF_LIVES_DAYS = [90, 180, 365, 730]  # 3mo, 6mo, 1yr, 2yr -- swept, selected on val

BASE_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "feature_pre_filter": False,
    "verbose": -1,
    "n_jobs": -1,
}
BUCKET_PARAMS = {
    "objective": "multiclass",
    "num_class": len(BUCKET_LABELS),
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 30,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l2": 0.1,
    "feature_pre_filter": False,
    "verbose": -1,
    "n_jobs": -1,
    "is_unbalance": True,
}


def recency_weight(
    created_at: pd.Series, anchor: pd.Timestamp, half_life_days: float
) -> np.ndarray:
    age_days = (anchor - created_at).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0)
    return np.exp(-np.log(2) * age_days / half_life_days).values


def train_weighted(X_train, y_train_hrs, w_train, X_val, y_val_hrs):
    log_y_train = np.log1p(y_train_hrs)
    log_y_val = np.log1p(y_val_hrs)
    dtrain = lgb.Dataset(X_train, label=log_y_train, weight=w_train)
    dval = lgb.Dataset(X_val, label=log_y_val, reference=dtrain)
    model = lgb.train(
        BASE_PARAMS,
        dtrain,
        num_boost_round=2000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def train_weighted_bucket(X_train, y_train_hrs, w_train, X_val, y_val_hrs):
    y_train_b = hours_to_bucket(y_train_hrs)
    y_val_b = hours_to_bucket(y_val_hrs)
    dtrain = lgb.Dataset(X_train, label=y_train_b, weight=w_train)
    dval = lgb.Dataset(X_val, label=y_val_b, reference=dtrain)
    model = lgb.train(
        BUCKET_PARAMS,
        dtrain,
        num_boost_round=500,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(0)],
    )
    return model


def run_repo(repo: str) -> dict:
    log.info("=" * 60)
    log.info("[%s] LEVER 3 recency-weighting sweep", repo)

    train = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_train_lever3.parquet")
    val = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_val_lever3.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_test_lever3.parquet")
    train = train[train["resolution_hours"] > 0].reset_index(drop=True)
    val = val[val["resolution_hours"] > 0].reset_index(drop=True)
    test = test[test["resolution_hours"] > 0].reset_index(drop=True)

    emb_train = load_embeddings_from_d1_index(repo, train)
    emb_val = load_embeddings_from_d1_index(repo, val)
    emb_test = load_embeddings_from_d1_index(repo, test)
    X_train, pca = engineer_features(train, train_df=train, embeddings=emb_train, pca=None)
    X_val, _ = engineer_features(val, train_df=train, embeddings=emb_val, pca=pca)
    X_test, _ = engineer_features(test, train_df=train, embeddings=emb_test, pca=pca)

    y_train_hrs, y_val_hrs, y_test_hrs = (
        train["resolution_hours"].values,
        val["resolution_hours"].values,
        test["resolution_hours"].values,
    )
    # Anchor to train's own latest created_at -- weighting must not know about the val/test window.
    anchor = pd.to_datetime(train["created_at"], utc=True).max()

    naive_pred_days = float(np.median(y_train_hrs)) / 24.0
    naive_mae_days = float(np.mean(np.abs(naive_pred_days - (y_test_hrs / 24.0))))

    sweep = []
    for hl in HALF_LIVES_DAYS:
        w_train = recency_weight(pd.to_datetime(train["created_at"], utc=True), anchor, hl)
        model = train_weighted(X_train, y_train_hrs, w_train, X_val, y_val_hrs)
        val_pred = np.expm1(model.predict(X_val))
        val_mae_days = float(np.mean(np.abs(val_pred - y_val_hrs))) / 24.0
        sweep.append({"half_life_days": hl, "val_mae_days": round(val_mae_days, 3)})
        log.info("[%s] half_life=%dd  val_mae=%.3fd", repo, hl, val_mae_days)

    best = min(sweep, key=lambda r: r["val_mae_days"])
    best_hl = best["half_life_days"]
    log.info("[%s] selected half_life=%dd (lowest val MAE)", repo, best_hl)

    w_train_final = recency_weight(pd.to_datetime(train["created_at"], utc=True), anchor, best_hl)
    point_model = train_weighted(X_train, y_train_hrs, w_train_final, X_val, y_val_hrs)
    bucket_model = train_weighted_bucket(X_train, y_train_hrs, w_train_final, X_val, y_val_hrs)

    y_pred_hrs = np.expm1(point_model.predict(X_test))
    model_mae_days = float(np.mean(np.abs(y_pred_hrs - y_test_hrs))) / 24.0
    improvement_pct = 100 * (naive_mae_days - model_mae_days) / naive_mae_days

    true_bucket_idx = hours_to_bucket(y_test_hrs)
    raw_proba = np.asarray(bucket_model.predict(X_test))
    pred_bucket_idx = raw_proba.argmax(axis=1)
    correct = (pred_bucket_idx == true_bucket_idx).astype(float)

    train_bucket_counts = pd.Series(hours_to_bucket(y_train_hrs)).value_counts().to_dict()
    majority_idx = max(train_bucket_counts, key=train_bucket_counts.get)
    naive_correct = (true_bucket_idx == majority_idx).astype(float)
    acc_delta = correct - naive_correct

    result = {
        "repo": repo,
        "half_life_sweep": sweep,
        "selected_half_life_days": best_hl,
        "point_regression": {
            "naive_mae_days": round(naive_mae_days, 2),
            "model_mae_days": round(model_mae_days, 2),
            "improvement_pct": round(improvement_pct, 2),
        },
        "bucket_classifier_vs_naive": {
            "n": len(true_bucket_idx),
            "trained_accuracy": round(float(correct.mean()), 4),
            "naive_accuracy": round(float(naive_correct.mean()), 4),
            "naive_majority_bucket": BUCKET_LABELS[majority_idx],
            "accuracy_delta_bootstrap": bootstrap_ci(acc_delta),
        },
    }
    log.info(
        "[%s] FINAL (half_life=%dd): point model=%.2fd naive=%.2fd (%.1f%%)  bucket trained=%.4f "
        "naive=%.4f delta=%s",
        repo,
        best_hl,
        model_mae_days,
        naive_mae_days,
        improvement_pct,
        result["bucket_classifier_vs_naive"]["trained_accuracy"],
        result["bucket_classifier_vs_naive"]["naive_accuracy"],
        result["bucket_classifier_vs_naive"]["accuracy_delta_bootstrap"],
    )
    return result


def main() -> None:
    results = {repo: run_repo(repo) for repo in REPOS}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "lever3_recency_weighting_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    log.info("Wrote reports/lever3_recency_weighting_results.json")


if __name__ == "__main__":
    main()
