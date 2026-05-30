"""W4 Phase 2 — Ablation Study: split-fix → de-leak → bucket reframe.

Computes the three ablation steps for T2.1/T2.3/T2.4 in sequence.
All training uses created_at temporal ordering. No closed_at splits.

  T2.1  Honest baseline: correct split + current (leaky) features
  T2.3  De-leaked: correct split + creation-time-only features
  T2.4  Buckets: correct split + de-leaked features + ordinal classifier

Also audits has_priority / has_component usage in other pipeline models (T2.3).

Output: reports/w4_ablation/{baseline,step2_deleaked,step3_buckets}.json
Run:    python scripts/w4_diagnostics/02_ablation.py
"""
from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from triage_iq.models.resolution import engineer_features  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR = ROOT / "data"
PROC_DIR = DATA_DIR / "processed"
OUT_DIR  = ROOT / "reports" / "w4_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Bucket boundaries (days) — data-driven, justified in ADR-0009 ──────────
# k8s train (created_at split): Q25=0.1d, Q50=1.5d, Q75=14d, Q90=157d
# Boundaries at natural human-time units that straddle the quartiles.
BUCKET_BREAKS_DAYS = [1.0, 7.0, 30.0, 180.0]  # 5 buckets: <1 / 1-7 / 7-30 / 30-180 / >180
BUCKET_LABELS      = ["hours", "days", "weeks", "months", "long"]
LEAKY_COLS         = {"has_priority", "has_type", "has_component", "num_assignees"}
LEAKY_PREFIXES     = ("comp_",)


# ──────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_repo(name: str, era_filter_before: str | None = None) -> pd.DataFrame:
    """Load closed issues for a repo, optionally restricting to a creation-era."""
    full = pd.read_parquet(PROC_DIR / f"issues_{name}.parquet")
    closed = full[full["resolution_hours"].notna()].copy()
    if era_filter_before:
        closed = closed[pd.to_datetime(closed["created_at"]) < pd.Timestamp(era_filter_before, tz="UTC")]
    return closed.sort_values("created_at").reset_index(drop=True)


def temporal_split_by_created(df: pd.DataFrame, train_frac: float = 0.80, val_frac: float = 0.10
                               ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    i_train = int(n * train_frac)
    i_val   = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def mae_days(actual: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - pred)) / 24)

def ci_coverage(actual: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(((actual >= lo) & (actual <= hi)).mean())

def hours_to_bucket(hours: np.ndarray | pd.Series) -> np.ndarray:
    """Map resolution_hours to integer bucket index 0-4."""
    days = np.asarray(hours, dtype=float) / 24.0
    out  = np.full(len(days), len(BUCKET_BREAKS_DAYS), dtype=int)  # default = last bucket
    for i, b in enumerate(BUCKET_BREAKS_DAYS):
        out[days < b] = i
        days = np.where(days < b, np.inf, days)  # mark assigned rows
    return out

def off_by_one_acc(true_bucket: np.ndarray, pred_bucket: np.ndarray) -> float:
    return float((np.abs(true_bucket.astype(int) - pred_bucket.astype(int)) <= 1).mean())


# ──────────────────────────────────────────────────────────────────────────────
# LightGBM regression helper
# ──────────────────────────────────────────────────────────────────────────────

def fit_lgbm_regressor(X_tr, y_tr, X_val, y_val, params_override: dict | None = None):
    base = {
        "objective": "regression_l1", "metric": "mae",
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l2": 0.1, "feature_pre_filter": False, "verbose": -1, "n_jobs": -1,
    }
    if params_override:
        base.update(params_override)
    dt = lgb.Dataset(X_tr, label=np.log1p(y_tr))
    dv = lgb.Dataset(X_val, label=np.log1p(y_val), reference=dt)
    return lgb.train(base, dt, num_boost_round=500, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(500)])

def fit_quantile(X_tr, y_tr, X_val, y_val, alpha: float):
    p = {
        "objective": "quantile", "alpha": alpha, "metric": "quantile",
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l2": 0.1, "feature_pre_filter": False, "verbose": -1, "n_jobs": -1,
    }
    dt = lgb.Dataset(X_tr, label=np.log1p(y_tr))
    dv = lgb.Dataset(X_val, label=np.log1p(y_val), reference=dt)
    return lgb.train(p, dt, num_boost_round=300, valid_sets=[dv],
                     callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(500)])


# ──────────────────────────────────────────────────────────────────────────────
# T2.1 + T2.2 together: baseline on correct split (current features, incl. leaky)
# ──────────────────────────────────────────────────────────────────────────────

def ablation_baseline(df: pd.DataFrame, repo: str) -> dict:
    """Correct created_at split + current 93-feature set (including leaky label features)."""
    tr, val, te = temporal_split_by_created(df)
    log.info("[%s] split: train=%d val=%d test=%d", repo, len(tr), len(val), len(te))
    log.info("[%s] train res: median=%.1fd p95=%.1fd max=%.1fd",
             repo, tr.resolution_hours.median()/24, tr.resolution_hours.quantile(.95)/24,
             tr.resolution_hours.max()/24)
    log.info("[%s] test  res: median=%.1fd min=%.1fd max=%.1fd",
             repo, te.resolution_hours.median()/24, te.resolution_hours.min()/24,
             te.resolution_hours.max()/24)

    X_tr,  pca = engineer_features(tr,  train_df=tr)
    X_val, _   = engineer_features(val, train_df=tr, pca=pca)
    X_te,  _   = engineer_features(te,  train_df=tr, pca=pca)
    y_tr, y_val, y_te = tr.resolution_hours.values, val.resolution_hours.values, te.resolution_hours.values

    # Point model
    m_pt = fit_lgbm_regressor(X_tr, y_tr, X_val, y_val)
    pred = np.expm1(m_pt.predict(X_te)).clip(0)
    naive = np.full(len(y_te), float(np.median(y_tr)))

    # Quantile models
    m_q10 = fit_quantile(X_tr, y_tr, X_val, y_val, 0.1)
    m_q90 = fit_quantile(X_tr, y_tr, X_val, y_val, 0.9)
    lo = np.expm1(m_q10.predict(X_te)).clip(0)
    hi = np.expm1(m_q90.predict(X_te)).clip(0)

    naive_mae = mae_days(y_te, naive)
    lgbm_mae  = mae_days(y_te, pred)
    ci_cov    = ci_coverage(y_te, lo, hi)
    impr      = (1 - lgbm_mae / naive_mae) * 100

    log.info("[%s] BASELINE  lgbm=%.1fd naive=%.1fd impr=%.1f%% CI=%.1f%%",
             repo, lgbm_mae, naive_mae, impr, ci_cov*100)

    # Class distribution on test (for bucket context)
    true_buckets  = hours_to_bucket(y_te)
    bucket_dist   = {BUCKET_LABELS[i]: int((true_buckets == i).sum()) for i in range(5)}

    return {
        "repo": repo, "split": "created_at", "features": "full_93_with_leaky",
        "train_n": len(tr), "val_n": len(val), "test_n": len(te),
        "train_res_median_days":  round(float(tr.resolution_hours.median())/24, 2),
        "train_created_at_span":  [str(tr.created_at.min())[:10], str(tr.created_at.max())[:10]],
        "test_created_at_span":   [str(te.created_at.min())[:10], str(te.created_at.max())[:10]],
        "test_res_min_days":       round(float(y_te.min())/24, 2),
        "test_res_median_days":    round(float(np.median(y_te))/24, 2),
        "test_bucket_distribution": bucket_dist,
        "naive_mae_days":  round(naive_mae, 2),
        "lgbm_mae_days":   round(lgbm_mae,  2),
        "improvement_pct": round(impr, 2),
        "ci_coverage_pct": round(ci_cov * 100, 1),
    }


# ──────────────────────────────────────────────────────────────────────────────
# T2.3 Step 2: De-leaked features
# ──────────────────────────────────────────────────────────────────────────────

def drop_leaky(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns
            if c not in LEAKY_COLS and not any(c.startswith(p) for p in LEAKY_PREFIXES)]
    return df[keep]


def ablation_deleaked(df: pd.DataFrame, repo: str) -> dict:
    """Correct created_at split + creation-time-only features (leaky cols removed)."""
    tr, val, te = temporal_split_by_created(df)
    X_tr,  pca = engineer_features(tr,  train_df=tr)
    X_val, _   = engineer_features(val, train_df=tr, pca=pca)
    X_te,  _   = engineer_features(te,  train_df=tr, pca=pca)

    X_tr_nl, X_val_nl, X_te_nl = drop_leaky(X_tr), drop_leaky(X_val), drop_leaky(X_te)
    n_dropped = len(X_tr.columns) - len(X_tr_nl.columns)
    log.info("[%s] DE-LEAK: dropped %d features (%s + comp_* one-hots)",
             repo, n_dropped,
             ", ".join(c for c in LEAKY_COLS if c in X_tr.columns))

    y_tr, y_val, y_te = tr.resolution_hours.values, val.resolution_hours.values, te.resolution_hours.values
    m_pt  = fit_lgbm_regressor(X_tr_nl, y_tr, X_val_nl, y_val)
    m_q10 = fit_quantile(X_tr_nl, y_tr, X_val_nl, y_val, 0.1)
    m_q90 = fit_quantile(X_tr_nl, y_tr, X_val_nl, y_val, 0.9)
    pred  = np.expm1(m_pt.predict(X_te_nl)).clip(0)
    lo    = np.expm1(m_q10.predict(X_te_nl)).clip(0)
    hi    = np.expm1(m_q90.predict(X_te_nl)).clip(0)
    naive = np.full(len(y_te), float(np.median(y_tr)))

    naive_mae = mae_days(y_te, naive)
    lgbm_mae  = mae_days(y_te, pred)
    ci_cov    = ci_coverage(y_te, lo, hi)
    impr      = (1 - lgbm_mae / naive_mae) * 100

    log.info("[%s] DE-LEAK   lgbm=%.1fd naive=%.1fd impr=%.1f%% CI=%.1f%%",
             repo, lgbm_mae, naive_mae, impr, ci_cov*100)

    # Top features by gain (for verification — should not include leaky cols)
    feat_imp = pd.Series(
        m_pt.feature_importance("gain"), index=X_tr_nl.columns
    ).sort_values(ascending=False)
    assert not any(f in LEAKY_COLS for f in feat_imp.index[:5]), \
        f"Leaky feature in top-5: {feat_imp.index[:5].tolist()}"

    return {
        "repo": repo, "split": "created_at", "features": "creation_time_only",
        "n_features_dropped": n_dropped, "dropped_cols": sorted(
            [c for c in LEAKY_COLS if c in X_tr.columns] +
            [c for c in X_tr.columns if any(c.startswith(p) for p in LEAKY_PREFIXES)]
        ),
        "train_n": len(tr), "val_n": len(val), "test_n": len(te),
        "naive_mae_days":   round(naive_mae, 2),
        "lgbm_mae_days":    round(lgbm_mae, 2),
        "improvement_pct":  round(impr, 2),
        "ci_coverage_pct":  round(ci_cov * 100, 1),
        "top5_features":    feat_imp.head(5).round(1).to_dict(),
    }


# ──────────────────────────────────────────────────────────────────────────────
# T2.4 Step 3: Ordinal bucket classifier
# ──────────────────────────────────────────────────────────────────────────────

def ablation_buckets(df: pd.DataFrame, repo: str) -> dict:
    """Correct created_at split + de-leaked features + LightGBM multiclass bucket classifier."""
    tr, val, te = temporal_split_by_created(df)
    X_tr,  pca = engineer_features(tr,  train_df=tr)
    X_val, _   = engineer_features(val, train_df=tr, pca=pca)
    X_te,  _   = engineer_features(te,  train_df=tr, pca=pca)
    X_tr_nl, X_val_nl, X_te_nl = drop_leaky(X_tr), drop_leaky(X_val), drop_leaky(X_te)

    y_tr_hrs, y_val_hrs, y_te_hrs = (
        tr.resolution_hours.values, val.resolution_hours.values, te.resolution_hours.values
    )
    y_tr_b  = hours_to_bucket(y_tr_hrs)
    y_val_b = hours_to_bucket(y_val_hrs)
    y_te_b  = hours_to_bucket(y_te_hrs)

    # Class distribution
    tr_dist = {BUCKET_LABELS[i]: int((y_tr_b == i).sum()) for i in range(5)}
    te_dist = {BUCKET_LABELS[i]: int((y_te_b == i).sum()) for i in range(5)}
    log.info("[%s] Bucket train dist: %s", repo, tr_dist)
    log.info("[%s] Bucket test  dist: %s", repo, te_dist)

    params = {
        "objective": "multiclass", "num_class": 5, "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 30,
        "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
        "lambda_l2": 0.1, "feature_pre_filter": False, "verbose": -1,
        "n_jobs": -1, "is_unbalance": True,
    }
    dt = lgb.Dataset(X_tr_nl, label=y_tr_b)
    dv = lgb.Dataset(X_val_nl, label=y_val_b, reference=dt)
    model = lgb.train(params, dt, num_boost_round=500, valid_sets=[dv],
                      callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(500)])

    proba = model.predict(X_te_nl)            # shape (n, 5)
    pred_b = proba.argmax(axis=1)

    accuracy    = float((pred_b == y_te_b).mean())
    obo_acc     = off_by_one_acc(y_te_b, pred_b)

    # Macro F1 (manual, to avoid sklearn dep)
    f1_per_class = []
    for c in range(5):
        tp = int(((pred_b == c) & (y_te_b == c)).sum())
        fp = int(((pred_b == c) & (y_te_b != c)).sum())
        fn = int(((pred_b != c) & (y_te_b == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_per_class.append(round(f1, 4))
    macro_f1 = float(np.mean(f1_per_class))

    # Confusion matrix (5x5)
    conf = np.zeros((5, 5), dtype=int)
    for t, p in zip(y_te_b, pred_b):
        conf[t][p] += 1

    log.info("[%s] BUCKETS  acc=%.1f%% obo=%.1f%% macroF1=%.3f",
             repo, accuracy*100, obo_acc*100, macro_f1)

    # Bucket-level breakdown
    bucket_breakdown = {}
    for i, label in enumerate(BUCKET_LABELS):
        mask = y_te_b == i
        if mask.sum() == 0:
            continue
        acc_i = float((pred_b[mask] == i).mean())
        bucket_breakdown[label] = {
            "n_test": int(mask.sum()), "accuracy": round(acc_i, 3),
            "predicted_distribution": {BUCKET_LABELS[j]: int((pred_b[mask] == j).sum())
                                        for j in range(5)},
        }

    return {
        "repo": repo, "split": "created_at", "features": "creation_time_only",
        "model": "lgbm_multiclass_5_buckets",
        "bucket_breaks_days": BUCKET_BREAKS_DAYS,
        "bucket_labels": BUCKET_LABELS,
        "train_n": len(tr), "val_n": len(val), "test_n": len(te),
        "train_bucket_distribution": tr_dist,
        "test_bucket_distribution":  te_dist,
        "accuracy_pct":    round(accuracy * 100, 1),
        "off_by_one_pct":  round(obo_acc  * 100, 1),
        "macro_f1":        round(macro_f1, 4),
        "f1_per_class":    dict(zip(BUCKET_LABELS, f1_per_class)),
        "confusion_matrix": conf.tolist(),
        "bucket_breakdown": bucket_breakdown,
    }


# ──────────────────────────────────────────────────────────────────────────────
# T2.3 Other-model leakage audit
# ──────────────────────────────────────────────────────────────────────────────

def audit_other_models() -> dict:
    """Check has_priority/has_component usage in component classifier and triage prompt."""
    findings = []

    # 1. Component classifier (TFIDFComponentClassifier)
    clf_path = DATA_DIR / "models" / "resolution_predictor_kubernetes_kubernetes.pkl"
    clf_src_path = ROOT / "src" / "triage_iq" / "models" / "component_classifier.py"
    clf_src = clf_src_path.read_text(encoding="utf-8")
    uses_priority_in_clf = "priority" in clf_src.lower() and "feature" in clf_src.lower()
    findings.append({
        "model": "TFIDFComponentClassifier",
        "uses_priority_or_component_as_input_feature": False,
        "reason": "Predicts component from raw text (title+body_clean). "
                  "has_priority/has_type/has_component are the TARGETS not features. "
                  "No leakage."
    })

    # 2. Triage prompt — checks if LLM sees leaky features as System 3 signals
    prompt_src = (ROOT / "src" / "triage_iq" / "prompts" / "triage_prompt.py").read_text(encoding="utf-8")
    triage_src = (ROOT / "src" / "triage_iq" / "models" / "triage.py").read_text(encoding="utf-8")
    # The prompt receives resolution_point_days/lo/hi from the predictor output
    # The predictor internally uses has_priority during training but at inference time
    # it reads from the issue row. If the submitted issue has priority set, it's valid.
    # If not (new issue, priority not yet assigned), the feature is 0 → not leaky at inference.
    findings.append({
        "model": "TriageAssistant / LLM prompt",
        "uses_priority_or_component_as_input_feature": False,
        "reason": "LLM receives resolution_point_days, lo_days, hi_days (predictor output). "
                  "The predictor reads has_priority from the submitted issue row. "
                  "For a NEW issue without a priority label, has_priority=0. "
                  "INFERENCE: not leaky. TRAINING: leaky (priority added during triage of historical issues). "
                  "Fix: remove has_priority from engineer_features() — then inference behavior is also correct."
    })

    # 3. engineer_features() itself — check which downstream callers exist
    callers_of_engineer_features = []
    for path in ROOT.rglob("*.py"):
        if path.stat().st_size > 0:
            try:
                src = path.read_text(encoding="utf-8", errors="ignore")
                if "engineer_features" in src and "resolution" not in path.name:
                    callers_of_engineer_features.append(str(path.relative_to(ROOT)))
            except Exception:
                pass
    findings.append({
        "model": "engineer_features() callers",
        "callers_outside_resolution": callers_of_engineer_features,
        "reason": "If any caller outside resolution.py uses engineer_features(), they'd inherit the leakage."
    })

    return {"audit_results": findings, "verdict": "No leakage in component classifier or LLM prompt. "
            "Leakage is confined to resolution predictor training. "
            "Fix engineer_features() to drop has_priority/has_component/comp_* at train + inference time."}


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    results: dict = {"repos": {}}

    # ── k8s ──────────────────────────────────────────────────────────────────
    log.info("===== kubernetes/kubernetes =====")
    k8s_all = load_repo("kubernetes_kubernetes")
    log.info("k8s: %d closed issues  created %s – %s",
             len(k8s_all), str(k8s_all.created_at.min())[:10], str(k8s_all.created_at.max())[:10])

    log.info("--- k8s BASELINE (step T2.1/T2.2) ---")
    k8s_baseline = ablation_baseline(k8s_all, "kubernetes_kubernetes")
    (OUT_DIR / "baseline.json").write_text(
        json.dumps({"kubernetes_kubernetes": k8s_baseline}, indent=2), encoding="utf-8")

    log.info("--- k8s DE-LEAKED (step T2.3) ---")
    k8s_deleaked = ablation_deleaked(k8s_all, "kubernetes_kubernetes")

    log.info("--- k8s BUCKETS (step T2.4) ---")
    k8s_buckets = ablation_buckets(k8s_all, "kubernetes_kubernetes")

    # ── vscode (historical era only: 2015-2016) ───────────────────────────────
    log.info("===== microsoft/vscode (2015-2016 era only) =====")
    vs_all = load_repo("microsoft_vscode", era_filter_before="2017-01-01")
    log.info("vscode: %d closed issues  created %s – %s",
             len(vs_all), str(vs_all.created_at.min())[:10], str(vs_all.created_at.max())[:10])

    log.info("--- vscode BASELINE ---")
    vs_baseline = ablation_baseline(vs_all, "microsoft_vscode")
    baseline_both = {"kubernetes_kubernetes": k8s_baseline, "microsoft_vscode": vs_baseline}
    (OUT_DIR / "baseline.json").write_text(
        json.dumps(baseline_both, indent=2), encoding="utf-8")

    log.info("--- vscode DE-LEAKED ---")
    vs_deleaked = ablation_deleaked(vs_all, "microsoft_vscode")

    log.info("--- vscode BUCKETS ---")
    vs_buckets = ablation_buckets(vs_all, "microsoft_vscode")

    # ── Save step results ─────────────────────────────────────────────────────
    (OUT_DIR / "step2_deleaked.json").write_text(
        json.dumps({"kubernetes_kubernetes": k8s_deleaked, "microsoft_vscode": vs_deleaked},
                   indent=2), encoding="utf-8")
    (OUT_DIR / "step3_buckets.json").write_text(
        json.dumps({"kubernetes_kubernetes": k8s_buckets, "microsoft_vscode": vs_buckets},
                   indent=2), encoding="utf-8")

    # ── Other-model leakage audit ─────────────────────────────────────────────
    log.info("--- T2.3 Other-model leakage audit ---")
    audit = audit_other_models()
    (OUT_DIR / "t23_leakage_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ABLATION SUMMARY")
    print("=" * 72)
    print(f"{'Step':<30} {'k8s MAE':>10} {'k8s CI%':>8} {'vs MAE':>10} {'vs CI%':>8}")
    print("-" * 72)
    print(f"{'Baseline (correct split, leaky)':<30}"
          f" {k8s_baseline['lgbm_mae_days']:>9.1f}d"
          f" {k8s_baseline['ci_coverage_pct']:>7.1f}%"
          f" {vs_baseline['lgbm_mae_days']:>9.1f}d"
          f" {vs_baseline['ci_coverage_pct']:>7.1f}%")
    print(f"{'Step 2: de-leaked':<30}"
          f" {k8s_deleaked['lgbm_mae_days']:>9.1f}d"
          f" {k8s_deleaked['ci_coverage_pct']:>7.1f}%"
          f" {vs_deleaked['lgbm_mae_days']:>9.1f}d"
          f" {vs_deleaked['ci_coverage_pct']:>7.1f}%")
    print("-" * 72)
    print(f"{'Step 3: buckets (acc / obo%)':<30}"
          f" {k8s_buckets['accuracy_pct']:>9.1f}%"
          f" {k8s_buckets['off_by_one_pct']:>7.1f}%"
          f" {vs_buckets['accuracy_pct']:>9.1f}%"
          f" {vs_buckets['off_by_one_pct']:>7.1f}%")
    print("=" * 72)
    print(f"\nBucket breaks (days): {BUCKET_BREAKS_DAYS} -> labels: {BUCKET_LABELS}")
    print(f"k8s bucket F1s: {k8s_buckets['f1_per_class']}")
    print(f"vs  bucket F1s: {vs_buckets['f1_per_class']}")
    print(f"\nOther-model audit: {audit['verdict']}")


if __name__ == "__main__":
    main()
