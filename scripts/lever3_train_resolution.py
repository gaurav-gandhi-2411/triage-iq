"""LEVER 3, step 2: retrain the resolution predictor (point + bucket) on the re-split data
from scripts/lever3_resplit_resolution.py, and measure bucket accuracy vs naive with the same
bootstrapped-CI discipline as ADR-0025/scripts/w6_diagnose_resolution.py.

Skips Optuna hyperparameter tuning (uses ResolutionTimePredictor.fit()'s built-in sensible
defaults) -- this is a first-pass "does re-splitting help at all" measurement; tuning is a
follow-up if the untuned result looks promising. Embeddings come from the D1 full-corpus BGE
index (data/models/d1_full_corpus_index_{repo}_bge, built by scripts/d1_build_full_corpus_index.py)
rather than the stale served dup_index_*_bge, since the new splits reach issues that index
doesn't cover.

Saves to data/models/resolution_predictor_{repo}_lever3.pkl -- does NOT touch the currently
shipped resolution_predictor_{repo}.pkl.

Reads:  data/processed/{repo}_temporal_{train,val,test}_lever3.parquet
        data/models/d1_full_corpus_index_{repo}_bge/
Writes: data/models/resolution_predictor_{repo}_lever3.pkl
        reports/lever3_resolution_results.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import faiss
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.resolution import (  # noqa: E402
    BUCKET_LABELS,
    ResolutionTimePredictor,
    engineer_features,
    hours_to_bucket,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 2000
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPORTS = Path("reports")
REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]


def load_embeddings_from_d1_index(repo: str, df: pd.DataFrame) -> np.ndarray | None:
    index_dir = MODELS_DIR / f"d1_full_corpus_index_{repo}_bge"
    if not (index_dir / "index.faiss").exists():
        log.warning(
            "D1 full-corpus BGE index not found for %s -- skipping embedding features", repo
        )
        return None
    meta = joblib.load(str(index_dir / "meta.pkl"))
    index = faiss.read_index(str(index_dir / "index.faiss"))
    num_to_idx = {int(n): i for i, n in enumerate(meta["issue_numbers"])}

    dim = index.d
    embs = np.zeros((len(df), dim), dtype=np.float32)
    missing = 0
    for row_pos, num in enumerate(df["number"]):
        idx = num_to_idx.get(int(num))
        if idx is not None:
            embs[row_pos] = index.reconstruct(int(idx))
        else:
            missing += 1
    if missing:
        log.warning(
            "[%s] %d/%d issues missing from D1 index (zero embedding fallback)",
            repo,
            missing,
            len(df),
        )
    return embs


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95_lower": float(lo), "ci95_upper": float(hi)}


def bucket_accuracy_vs_naive(
    predictor: ResolutionTimePredictor, X_test: pd.DataFrame, true_bucket_idx: np.ndarray
) -> dict:
    assert predictor.model_bucket is not None
    raw_proba = np.asarray(predictor.model_bucket.predict(X_test))
    pred_bucket_idx = raw_proba.argmax(axis=1)
    correct = (pred_bucket_idx == true_bucket_idx).astype(float)
    obo_correct = (np.abs(pred_bucket_idx - true_bucket_idx) <= 1).astype(float)

    majority_bucket = max(
        predictor.bucket_train_distribution, key=predictor.bucket_train_distribution.get
    )
    majority_idx = BUCKET_LABELS.index(majority_bucket)
    naive_correct = (true_bucket_idx == majority_idx).astype(float)
    naive_obo_correct = (np.abs(majority_idx - true_bucket_idx) <= 1).astype(float)

    acc_delta = correct - naive_correct
    obo_delta = obo_correct - naive_obo_correct

    return {
        "n": len(true_bucket_idx),
        "trained_accuracy": round(float(correct.mean()), 4),
        "naive_accuracy": round(float(naive_correct.mean()), 4),
        "accuracy_delta_bootstrap": bootstrap_ci(acc_delta),
        "trained_obo": round(float(obo_correct.mean()), 4),
        "naive_obo": round(float(naive_obo_correct.mean()), 4),
        "obo_delta_bootstrap": bootstrap_ci(obo_delta),
        "naive_majority_bucket": majority_bucket,
    }


def run_repo(repo: str) -> dict:
    log.info("=" * 60)
    log.info("[%s] LEVER 3 retrain (re-split, untuned defaults)", repo)

    train = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_train_lever3.parquet")
    val = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_val_lever3.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_test_lever3.parquet")
    train = train[train["resolution_hours"] > 0]
    val = val[val["resolution_hours"] > 0]
    test = test[test["resolution_hours"] > 0].reset_index(drop=True)

    y_train, y_val, y_test = (
        train["resolution_hours"],
        val["resolution_hours"],
        test["resolution_hours"],
    )
    log.info("[%s] sizes: train=%d val=%d test=%d", repo, len(train), len(val), len(test))

    emb_train = load_embeddings_from_d1_index(repo, train)
    emb_val = load_embeddings_from_d1_index(repo, val)
    emb_test = load_embeddings_from_d1_index(repo, test)

    X_train, pca = engineer_features(train, train_df=train, embeddings=emb_train, pca=None)
    X_val, _ = engineer_features(val, train_df=train, embeddings=emb_val, pca=pca)
    X_test, _ = engineer_features(test, train_df=train, embeddings=emb_test, pca=pca)

    naive_pred_days = float(y_train.median()) / 24.0
    naive_mae_days = float(np.mean(np.abs(naive_pred_days - (y_test.values / 24.0))))

    predictor = ResolutionTimePredictor(repo=repo)
    predictor.pca = pca
    predictor.fit(X_train, y_train, X_val, y_val, lgbm_params=None)

    y_pred_hrs = predictor.predict(X_test)
    model_mae_days = float(mean_absolute_error(y_test.values, y_pred_hrs)) / 24.0
    improvement_pct = 100 * (naive_mae_days - model_mae_days) / naive_mae_days

    true_bucket_idx = hours_to_bucket(y_test.values)
    bucket_eval = bucket_accuracy_vs_naive(predictor, X_test, true_bucket_idx)

    model_path = MODELS_DIR / f"resolution_predictor_{repo}_lever3.pkl"
    predictor.save(str(model_path))
    log.info("[%s] saved to %s", repo, model_path)

    result = {
        "repo": repo,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "point_regression": {
            "naive_mae_days": round(naive_mae_days, 2),
            "model_mae_days": round(model_mae_days, 2),
            "improvement_pct": round(improvement_pct, 2),
        },
        "bucket_classifier_vs_naive": bucket_eval,
    }
    log.info(
        "[%s] point: model=%.2fd naive=%.2fd (%.1f%% improvement)  bucket: trained_acc=%.4f "
        "naive_acc=%.4f delta=%s",
        repo,
        model_mae_days,
        naive_mae_days,
        improvement_pct,
        bucket_eval["trained_accuracy"],
        bucket_eval["naive_accuracy"],
        bucket_eval["accuracy_delta_bootstrap"],
    )
    return result


def main() -> None:
    results = {repo: run_repo(repo) for repo in REPOS}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "lever3_resolution_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    log.info("Wrote reports/lever3_resolution_results.json")


if __name__ == "__main__":
    main()
