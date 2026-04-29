"""Train LightGBM resolution time predictor per repo.

Pipeline:
  1. Load temporal splits (closed issues only)
  2. Reconstruct BGE embeddings from Day 5 FAISS index
  3. Engineer features (text length, temporal, author history, PCA embeddings)
  4. Naive baseline (predict median)
  5. Optuna-tuned LightGBM point predictor + Q10/Q90 quantile models
  6. Evaluation: MAE, RMSE, R², calibration, per-component bias
  7. Charts + JSON results

Usage:
    python scripts/09_train_resolution.py
    python scripts/09_train_resolution.py --repos microsoft_vscode --trials 20
"""

import argparse
import json
import logging
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
warnings.filterwarnings("ignore", category=UserWarning)

import faiss
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from triage_iq.models.resolution import ResolutionTimePredictor, engineer_features

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
RESULTS_PATH = Path("reports/resolution_results.json")


# ── Embedding loading from Day 5 FAISS cache ─────────────────────────────────

def load_embeddings_from_index(repo: str, df: pd.DataFrame) -> np.ndarray | None:
    """Reconstruct BGE embeddings from Day 5 FAISS index for issues in df."""
    index_dir = MODELS_DIR / f"dup_index_{repo}_bge"
    if not (index_dir / "index.faiss").exists():
        log.warning("BGE index not found for %s — skipping embedding features", repo)
        return None

    meta = joblib.load(str(index_dir / "meta.pkl"))
    index = faiss.read_index(str(index_dir / "index.faiss"))
    num_to_faiss_idx = {int(n): i for i, n in enumerate(meta["issue_numbers"])}

    dim = index.d
    embs = np.zeros((len(df), dim), dtype=np.float32)
    missing = 0
    for row_pos, num in enumerate(df["number"]):
        faiss_idx = num_to_faiss_idx.get(int(num))
        if faiss_idx is not None:
            embs[row_pos] = index.reconstruct(int(faiss_idx))
        else:
            missing += 1

    if missing > 0:
        log.warning("[%s] %d issues not in FAISS index (will use zero embedding)", repo, missing)
    log.info("[%s] Loaded %d BGE embeddings (dim=%d)", repo, len(df) - missing, dim)
    return embs


# ── Evaluation helpers ────────────────────────────────────────────────────────

def evaluate_predictions(y_true_hrs: np.ndarray, y_pred_hrs: np.ndarray, tag: str = "") -> dict:
    log_true = np.log1p(y_true_hrs)
    log_pred = np.log1p(y_pred_hrs)

    mae_log = mean_absolute_error(log_true, log_pred)
    mae_hrs = mean_absolute_error(y_true_hrs, y_pred_hrs)
    rmse_hrs = np.sqrt(mean_squared_error(y_true_hrs, y_pred_hrs))
    r2_log = r2_score(log_true, log_pred)
    mae_days = mae_hrs / 24

    result = {
        "mae_log": round(mae_log, 4),
        "mae_hrs": round(mae_hrs, 1),
        "mae_days": round(mae_days, 2),
        "rmse_hrs": round(rmse_hrs, 1),
        "r2_log": round(r2_log, 4),
    }
    log.info("[%s] MAE=%.4f(log) / %.1fh / %.2fd  RMSE=%.1fh  R²=%.4f",
             tag, mae_log, mae_hrs, mae_days, rmse_hrs, r2_log)
    return result


def interval_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


# ── Visualizations ────────────────────────────────────────────────────────────

def plot_calibration(y_true: np.ndarray, y_pred: np.ndarray, repo: str, out: str) -> None:
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)

    bins = np.percentile(log_pred, np.linspace(0, 100, 11))
    bin_means_pred, bin_means_true = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (log_pred >= lo) & (log_pred < hi)
        if mask.sum() > 5:
            bin_means_pred.append(log_pred[mask].mean())
            bin_means_true.append(log_true[mask].mean())

    fig, ax = plt.subplots(figsize=(6, 5))
    lims = [min(bin_means_pred + bin_means_true), max(bin_means_pred + bin_means_true)]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect calibration")
    ax.scatter(bin_means_pred, bin_means_true, color="steelblue", s=60, zorder=3)
    ax.plot(bin_means_pred, bin_means_true, "steelblue", alpha=0.7)
    ax.set(xlabel="Mean predicted log1p(hours)", ylabel="Mean actual log1p(hours)",
           title=f"Calibration — {repo.replace('_', '/')}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved calibration chart to %s", out)


def plot_per_component_mae(
    y_true: np.ndarray, y_pred: np.ndarray,
    components: pd.Series, repo: str, out: str,
) -> dict:
    comp_mae = {}
    for comp in components.dropna().unique():
        mask = components == comp
        if mask.sum() >= 5:
            comp_mae[comp] = mean_absolute_error(y_true[mask], y_pred[mask]) / 24
    if not comp_mae:
        return {}

    ordered = sorted(comp_mae.items(), key=lambda x: -x[1])[:20]
    labels = [x[0] for x in ordered]
    values = [x[1] for x in ordered]
    overall_mae = mean_absolute_error(y_true, y_pred) / 24

    fig, ax = plt.subplots(figsize=(8, max(4, len(labels) * 0.3)))
    ax.barh(range(len(labels)), values, color="steelblue")
    ax.axvline(overall_mae, color="red", linestyle="--", alpha=0.7, label=f"Overall MAE={overall_mae:.1f}d")
    ax.set(yticks=range(len(labels)), yticklabels=labels,
           xlabel="MAE (days)", title=f"Per-component MAE — {repo.replace('_', '/')}")
    ax.legend(fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved per-component MAE chart to %s", out)
    return dict(ordered[:5])


# ── Optuna tuning ─────────────────────────────────────────────────────────────

def tune_lgbm(X_train, y_train_log, X_val, y_val_log, n_trials: int = 30) -> dict:
    import lightgbm as lgb

    dtrain = lgb.Dataset(X_train, label=y_train_log)
    dval = lgb.Dataset(X_val, label=y_val_log, reference=dtrain)

    def objective(trial):
        params = {
            "objective": "regression_l1",
            "metric": "mae",
            "verbose": -1,
            "n_jobs": -1,
            "feature_pre_filter": False,  # required when tuning min_data_in_leaf
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
        }
        model = lgb.train(
            params, dtrain,
            num_boost_round=500,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(99999)],
        )
        return model.best_score["valid_0"]["l1"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False, catch=(Exception,))
    log.info("Optuna best MAE=%.4f  params=%s", study.best_value, study.best_params)
    return study.best_params


# ── Per-repo pipeline ─────────────────────────────────────────────────────────

def run_repo(repo: str, n_trials: int = 30) -> dict:
    log.info("=" * 60)
    log.info("Repo: %s", repo)

    train = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_val.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_test.parquet")

    # Filter to positive resolution hours
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        n_before = len(split_df)
        split_df = split_df[split_df["resolution_hours"] > 0]
        if n_before != len(split_df):
            log.warning("[%s] %s: dropped %d zero/negative resolution rows", repo, split_name, n_before - len(split_df))

    train = train[train["resolution_hours"] > 0]
    val   = val[val["resolution_hours"] > 0]
    test  = test[test["resolution_hours"] > 0]

    y_train = train["resolution_hours"]
    y_val   = val["resolution_hours"]
    y_test  = test["resolution_hours"]

    log.info("Sizes — train=%d  val=%d  test=%d", len(train), len(val), len(test))
    log.info("y_train: median=%.1fh (%.1fd)  p95=%.1fh (%.1fd)",
             y_train.median(), y_train.median()/24, y_train.quantile(.95), y_train.quantile(.95)/24)

    # ── Load BGE embeddings ───────────────────────────────────────
    emb_train = load_embeddings_from_index(repo, train)
    emb_val   = load_embeddings_from_index(repo, val)
    emb_test  = load_embeddings_from_index(repo, test)

    # ── Feature engineering ───────────────────────────────────────
    log.info("[%s] Engineering features...", repo)
    X_train, pca = engineer_features(train, train_df=train, embeddings=emb_train, pca=None)
    X_val,   _   = engineer_features(val,   train_df=train, embeddings=emb_val,   pca=pca)
    X_test,  _   = engineer_features(test,  train_df=train, embeddings=emb_test,  pca=pca)

    log.info("[%s] Feature matrix: train=%s  features=%d", repo, X_train.shape, X_train.shape[1])

    # Save features parquet (train only, for inspection)
    feat_path = PROCESSED_DIR / f"{repo}_resolution_features.parquet"
    X_train.to_parquet(feat_path)
    log.info("[%s] Saved train features to %s", repo, feat_path)

    # ── Naive baseline: predict train median ──────────────────────
    train_median_hrs = float(y_train.median())
    naive_pred = np.full(len(y_test), train_median_hrs)
    naive_metrics = evaluate_predictions(y_test.values, naive_pred, tag=f"{repo}/naive")

    # ── Optuna tuning ─────────────────────────────────────────────
    log.info("[%s] Optuna: %d trials...", repo, n_trials)
    log_y_train = np.log1p(y_train.values)
    log_y_val   = np.log1p(y_val.values)
    best_params = tune_lgbm(X_train, log_y_train, X_val, log_y_val, n_trials=n_trials)

    # ── Train final model ─────────────────────────────────────────
    predictor = ResolutionTimePredictor(repo=repo)
    predictor.pca = pca
    predictor.fit(X_train, y_train, X_val, y_val, lgbm_params=best_params)

    # ── Evaluate ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    y_pred_hrs = predictor.predict(X_test)
    eval_time = time.perf_counter() - t0
    lgbm_metrics = evaluate_predictions(y_test.values, y_pred_hrs, tag=f"{repo}/lgbm")

    improvement_pct = (naive_metrics["mae_days"] - lgbm_metrics["mae_days"]) / naive_metrics["mae_days"] * 100
    log.info("[%s] Improvement over naive: %.1f%%", repo, improvement_pct)

    # ── Confidence intervals ──────────────────────────────────────
    lower_hrs, upper_hrs = predictor.predict_intervals(X_test)
    coverage = interval_coverage(y_test.values, lower_hrs, upper_hrs)
    log.info("[%s] 80%% CI coverage on test: %.3f (target=0.80)", repo, coverage)

    # ── Per-component coverage ────────────────────────────────────
    comp_coverage = {}
    if "component" in test.columns:
        for comp in test["component"].dropna().unique():
            mask = test["component"].values == comp
            if mask.sum() >= 5:
                cov = interval_coverage(y_test.values[mask], lower_hrs[mask], upper_hrs[mask])
                comp_coverage[comp] = round(cov, 3)

    # ── Latency benchmark ─────────────────────────────────────────
    lat_times = []
    sample = X_test.iloc[:100]
    for i in range(200):
        row = sample.iloc[i % len(sample):i % len(sample) + 1]
        t0 = time.perf_counter()
        predictor.predict(row)
        lat_times.append((time.perf_counter() - t0) * 1000)
    lat_arr = np.array(lat_times)
    lat_p50 = float(np.percentile(lat_arr, 50))
    lat_p95 = float(np.percentile(lat_arr, 95))
    log.info("[%s] Latency p50=%.2fms  p95=%.2fms", repo, lat_p50, lat_p95)

    # ── Feature importance ────────────────────────────────────────
    fi = predictor.feature_importance("gain").head(10)
    log.info("[%s] Top-10 features: %s", repo, dict(fi.round(1)))

    # ── Charts ────────────────────────────────────────────────────
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_calibration(
        y_test.values, y_pred_hrs, repo,
        str(CHARTS_DIR / f"resolution_calibration_{repo}.png"),
    )
    worst5 = plot_per_component_mae(
        y_test.values, y_pred_hrs,
        test["component"] if "component" in test.columns else pd.Series(dtype=str),
        repo, str(CHARTS_DIR / f"resolution_per_component_mae_{repo}.png"),
    )

    # ── Save model ────────────────────────────────────────────────
    model_path = MODELS_DIR / f"resolution_predictor_{repo}.pkl"
    predictor.save(str(model_path))

    return {
        "repo": repo,
        "train_size": len(train),
        "val_size": len(val),
        "test_size": len(test),
        "n_features": X_train.shape[1],
        "train_median_hrs": round(train_median_hrs, 1),
        "naive_metrics": naive_metrics,
        "lgbm_metrics": lgbm_metrics,
        "improvement_pct": round(improvement_pct, 1),
        "best_lgbm_params": best_params,
        "ci_coverage": round(coverage, 4),
        "comp_coverage_sample": comp_coverage,
        "latency_p50_ms": round(lat_p50, 3),
        "latency_p95_ms": round(lat_p95, 3),
        "top5_features": fi.head(5).round(2).to_dict(),
        "worst5_components_by_mae": worst5,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS
    all_results = {}
    total_t = time.perf_counter()

    for repo in repos:
        if not (PROCESSED_DIR / f"{repo}_temporal_train.parquet").exists():
            log.warning("Skipping %s — temporal splits not found", repo)
            continue
        all_results[repo] = run_repo(repo, n_trials=args.trials)

    total_elapsed = time.perf_counter() - total_t

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info("Saved results to %s", RESULTS_PATH)

    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("%-30s  %10s  %10s  %10s  %8s  %8s", "Repo", "Naive MAE", "LGBM MAE", "Improvement", "CI Cover", "p50(ms)")
    for repo, r in all_results.items():
        log.info("%-30s  %10.2fd  %10.2fd  %9.1f%%  %8.3f  %8.3f",
                 repo,
                 r["naive_metrics"]["mae_days"],
                 r["lgbm_metrics"]["mae_days"],
                 r["improvement_pct"],
                 r["ci_coverage"],
                 r["latency_p50_ms"])
    log.info("Total time: %.1fs (%.1f min)", total_elapsed, total_elapsed / 60)


if __name__ == "__main__":
    main()
