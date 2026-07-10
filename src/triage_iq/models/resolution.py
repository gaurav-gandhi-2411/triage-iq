"""LightGBM resolution-time predictor.

Primary output: ordinal bucket classifier (hours/days/weeks/months/long).
Secondary output: point regression + Q10/Q90 intervals (kept for comparison).
All features use only information available at issue creation time. See ADR-0009.

Bucket boundaries (days): [1, 7, 30, 180]
  hours   < 1 day
  days    1–7 days
  weeks   7–30 days
  months  30–180 days
  long    > 180 days
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

# Bucket boundaries and labels — data-driven from k8s/vscode training distributions.
# Boundaries chosen at natural human-time units that straddle quartile breakpoints.
# k8s train Q50=1.5d, Q75=14d, Q90=157d. See ADR-0009 T2.4.
BUCKET_BREAKS_DAYS: list[float] = [1.0, 7.0, 30.0, 180.0]
BUCKET_LABELS: list[str] = ["hours", "days", "weeks", "months", "long"]
ResolutionBucket = Literal["hours", "days", "weeks", "months", "long"]

# Per-repo bucket-classifier trust decision (ADR-0025). Ships only where the trained
# classifier beats a naive majority-class baseline with a bootstrapped 95% CI (2000
# resamples, held-out temporal test set) that excludes zero -- see
# scripts/w6_diagnose_resolution.py / reports/w6_resolution_diagnosis.json. Replaces
# ADR-0009's original arbitrary obo>=60% threshold with this rigorous criterion.
# Repos not listed here default to trusted (unmeasured, not proven untrustworthy) --
# see predict_bucket()'s use of this dict.
BUCKET_CLASSIFIER_TRUSTED: dict[str, bool] = {
    # accuracy delta vs naive: +3.27pp, 95% CI [+1.80pp, +4.74pp] -- excludes zero, positive.
    "kubernetes_kubernetes": True,
    # accuracy delta vs naive: -22.08pp, 95% CI [-25.81pp, -18.02pp] -- excludes zero, WRONG
    # direction. The trained classifier is significantly WORSE than guessing the majority
    # class (train/test distribution shift, ADR-0009 T1.5 -- not a feature gap, see ADR-0025).
    "microsoft_vscode": False,
}


def hours_to_bucket(hours: np.ndarray | float) -> np.ndarray:
    """Map resolution_hours to integer bucket index 0–4."""
    days = np.asarray(hours, dtype=float) / 24.0
    out  = np.full(days.shape if days.ndim > 0 else (1,), len(BUCKET_BREAKS_DAYS), dtype=int)
    for i, b in enumerate(BUCKET_BREAKS_DAYS):
        mask = days < b
        out  = np.where(mask & (out == len(BUCKET_BREAKS_DAYS)), i, out)
        days = np.where(mask, np.inf, days)
    return out

logger = logging.getLogger(__name__)


@dataclass
class ConformalAdjustment:
    """CQR scalar adjustment per repo per target level.

    Q is the conformal quantile of conformity scores on a held-out calibration
    set. The conformal interval is [q_lo(x) - Q, q_hi(x) + Q].
    Conformity score: E_i = max(q_lo(x_i) - y_i, y_i - q_hi(x_i)).
    """

    target_coverage: float          # e.g. 0.80
    n_calibration: int              # number of calibration points used
    q_adjustment_hours: float       # Q in hours (applied to raw interval predictions)
    empirical_test_coverage: float  # measured on held-out test set (post-calibration)
    n_test: int                     # size of the test set used

# ------------------------------------------------------------------
# Feature engineering
# ------------------------------------------------------------------

def engineer_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    embeddings: np.ndarray | None = None,
    pca=None,
) -> pd.DataFrame:
    """Build feature matrix from issues DataFrame.

    Args:
        df: Issues to featurize (any split).
        train_df: Full training set — used to compute author history and fit
                  PCA. If None, author features fall back to zeros.
        embeddings: Pre-computed BGE embeddings for df rows, shape (n, dim).
                    If None, embedding features are omitted.
        pca: Fitted PCA instance. If None and embeddings provided, PCA is
             fitted on train embeddings (only call this with train data).

    Returns:
        Feature DataFrame aligned with df index.
    """
    feats = pd.DataFrame(index=df.index)

    # ── Text length features ──────────────────────────────────────
    title = df["title"].fillna("")
    body = df["body_clean"].fillna("")

    feats["title_len_chars"] = title.str.len()
    feats["title_len_words"] = title.str.split().str.len().fillna(0)
    feats["body_len_chars"] = body.str.len()
    feats["body_len_words"] = body.str.split().str.len().fillna(0)
    feats["body_len_lines"] = body.str.count("\n") + 1
    feats["has_code_blocks"] = df.get("code_blocks", pd.Series(0, index=df.index)).apply(
        lambda x: 1 if (isinstance(x, list) and len(x) > 0) or (isinstance(x, str) and len(x) > 2) else 0
    )

    # ── Temporal features ─────────────────────────────────────────
    if "created_at" in df.columns:
        created = pd.to_datetime(df["created_at"], utc=True)
    else:
        from datetime import datetime
        from datetime import timezone as _tz
        created = pd.Series(
            [pd.Timestamp(datetime.now(_tz.utc))] * len(df), index=df.index
        )
    feats["day_of_week"] = created.dt.dayofweek
    feats["hour_of_day"] = created.dt.hour
    feats["week_of_year"] = created.dt.isocalendar().week.astype(int)
    repo_start = created.min()
    feats["days_since_repo_start"] = (created - repo_start).dt.days

    # ── Label features excluded (triage-assigned, not creation-time) ──
    # has_priority, has_component, has_type, num_assignees, and comp_* one-hots
    # are omitted: they are assigned during triage AFTER issue creation and are
    # not available at the time a new issue arrives. Including them in training
    # creates a spurious correlation with resolution time (e.g., has_priority
    # fill: fast issues 6.9% vs slow issues 93.1%). See ADR-0009 T1.4.

    # ── Author features (leak-proof: only past info) ──────────────
    if train_df is not None:
        all_df = pd.concat([train_df, df], sort=False).drop_duplicates(subset=["number"])
        all_df = all_df.sort_values("created_at")

        author_prior_count: dict[str, int] = {}
        author_prior_resolutions: dict[str, list[float]] = {}
        author_count_at = {}
        author_median_at = {}

        for _, row in all_df.iterrows():
            num = row["number"]
            author = row.get("author", "")
            count = author_prior_count.get(author, 0)
            medians = author_prior_resolutions.get(author, [])
            author_count_at[num] = count
            author_median_at[num] = np.median(medians) if medians else np.nan

            # Update for future rows
            author_prior_count[author] = count + 1
            if pd.notna(row.get("resolution_hours")) and row.get("state") == "closed":
                author_prior_resolutions.setdefault(author, []).append(row["resolution_hours"])

        feats["author_prior_count"] = df["number"].map(author_count_at).fillna(0)
        feats["is_first_author"] = (feats["author_prior_count"] == 0).astype(int)
        feats["author_prior_median_hrs"] = df["number"].map(author_median_at).fillna(-1)
    else:
        feats["author_prior_count"] = 0
        feats["is_first_author"] = 1
        feats["author_prior_median_hrs"] = -1

    # ── Cross features ────────────────────────────────────────────
    feats["body_x_title"] = (feats["body_len_chars"] * feats["title_len_chars"]) / 1e6
    feats["code_x_body"] = feats["has_code_blocks"] * feats["body_len_chars"] / 1e4

    # ── Embedding features (BGE → PCA 64) ────────────────────────
    if embeddings is not None:
        if pca is None:
            from sklearn.decomposition import PCA
            n_components = min(64, embeddings.shape[1], embeddings.shape[0] - 1)
            pca = PCA(n_components=n_components, random_state=42)
            pca.fit(embeddings)
        emb_reduced = pca.transform(embeddings)
        for i in range(emb_reduced.shape[1]):
            feats[f"emb_{i}"] = emb_reduced[:, i]

    return feats, pca


# ------------------------------------------------------------------
# Predictor class
# ------------------------------------------------------------------

class ResolutionTimePredictor:
    """LightGBM resolution-time predictor.

    Primary output: ordinal bucket (hours/days/weeks/months/long) via predict_bucket().
    Secondary output: point estimate + Q10/Q90 interval via predict() / predict_intervals().

    The bucket classifier is the recommended production output. The point regression
    is retained for ablation comparisons and as a fallback. See ADR-0009.
    """

    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.model_point: lgb.Booster | None = None
        self.model_q10: lgb.Booster | None = None
        self.model_q90: lgb.Booster | None = None
        self.model_bucket: lgb.Booster | None = None
        # Per-repo training-distribution bucket prior (used as fallback when
        # bucket model has low confidence — e.g., vscode where obo=55%).
        self.bucket_train_distribution: dict[str, float] = {}
        self.pca = None
        self.top_components: list[str] = []
        self.feature_names: list[str] = []
        self.conformal_adjustments: dict[float, ConformalAdjustment] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        lgbm_params: dict | None = None,
    ) -> "ResolutionTimePredictor":
        self.feature_names = list(X_train.columns)
        log_y_train = np.log1p(y_train.values)
        log_y_val = np.log1p(y_val.values)

        dtrain = lgb.Dataset(X_train, label=log_y_train)
        dval = lgb.Dataset(X_val, label=log_y_val, reference=dtrain)

        base_params = {
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
        if lgbm_params:
            base_params.update(lgbm_params)

        callbacks = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]

        self.model_point = lgb.train(
            base_params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=callbacks,  # type: ignore[arg-type]
        )
        logger.info("[%s] Point model: %d rounds, val MAE=%.4f",
                    self.repo, self.model_point.best_iteration,
                    self.model_point.best_score["valid_0"]["l1"])

        # Quantile models
        for alpha, attr in [(0.1, "model_q10"), (0.9, "model_q90")]:
            qp = dict(base_params)
            qp.update({"objective": "quantile", "alpha": alpha, "metric": "quantile"})
            dtrain_q = lgb.Dataset(X_train, label=log_y_train)
            dval_q = lgb.Dataset(X_val, label=log_y_val, reference=dtrain_q)
            model_q = lgb.train(
                qp, dtrain_q,
                num_boost_round=1000,
                valid_sets=[dval_q],
                callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(200)],
            )
            setattr(self, attr, model_q)
            logger.info("[%s] Q%.0f model: %d rounds", self.repo, alpha * 100,
                        model_q.best_iteration)

        # Bucket classifier (primary production output)
        y_train_b = hours_to_bucket(y_train.values)
        y_val_b   = hours_to_bucket(y_val.values)
        bucket_params = {
            "objective": "multiclass", "num_class": len(BUCKET_LABELS),
            "metric": "multi_logloss", "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 30, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 5, "lambda_l2": 0.1,
            "feature_pre_filter": False, "verbose": -1, "n_jobs": -1,
            "is_unbalance": True,
        }
        dt_b = lgb.Dataset(X_train, label=y_train_b)
        dv_b = lgb.Dataset(X_val,   label=y_val_b,   reference=dt_b)
        self.model_bucket = lgb.train(
            bucket_params, dt_b, num_boost_round=500, valid_sets=[dv_b],
            callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(200)],
        )
        train_bucket_counts = {BUCKET_LABELS[i]: int((y_train_b == i).sum()) for i in range(5)}
        total = sum(train_bucket_counts.values())
        self.bucket_train_distribution = {k: v / total for k, v in train_bucket_counts.items()}
        logger.info("[%s] Bucket model: %d rounds  train dist=%s",
                    self.repo, self.model_bucket.best_iteration, train_bucket_counts)

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point prediction in original hours (not log)."""
        assert self.model_point is not None, "Call fit() first"
        log_pred = self.model_point.predict(X)
        return np.expm1(log_pred).clip(min=0)

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model_point is not None, "Call fit() first"
        return self.model_point.predict(X)  # type: ignore[return-value]

    def predict_intervals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower_hours, upper_hours) 80% prediction interval."""
        assert self.model_q10 is not None and self.model_q90 is not None, "Call fit() first"
        lower = np.expm1(self.model_q10.predict(X)).clip(min=0)
        upper = np.expm1(self.model_q90.predict(X)).clip(min=0)
        return lower, upper

    def calibrate_cqr(
        self,
        X_cal: pd.DataFrame,
        y_cal_hours: np.ndarray,
        target_coverage: float = 0.80,
    ) -> ConformalAdjustment:
        """Compute the CQR scalar adjustment Q from a held-out calibration set.

        Args:
            X_cal: Feature DataFrame for calibration points (out-of-sample; model
                   must never have been trained on these rows).
            y_cal_hours: True resolution_hours for each calibration row.
            target_coverage: Desired marginal coverage level (e.g. 0.80).

        Returns:
            ConformalAdjustment with q_adjustment_hours filled in.
            empirical_test_coverage and n_test are left at 0.0 / 0 — fill them
            in from an evaluation script after measuring coverage on a test set.
        """
        lower_cal, upper_cal = self.predict_intervals(X_cal)
        E = np.maximum(lower_cal - y_cal_hours, y_cal_hours - upper_cal)
        n = len(E)
        # CQR finite-sample level: ceil((n+1)*(1-alpha))/n where alpha=1-target_coverage.
        # Equivalent to ceil((n+1)*target_coverage)/n. Clips to 1.0 for small n.
        level = np.ceil((n + 1) * target_coverage) / n
        Q = float(np.quantile(E, min(level, 1.0)))
        return ConformalAdjustment(
            target_coverage=target_coverage,
            n_calibration=n,
            q_adjustment_hours=Q,
            empirical_test_coverage=0.0,
            n_test=0,
        )

    def predict_conformal_interval(
        self,
        X: pd.DataFrame,
        adjustment: ConformalAdjustment,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return CQR-adjusted prediction intervals.

        Args:
            X: Feature DataFrame for new points.
            adjustment: ConformalAdjustment from calibrate_cqr.

        Returns:
            (lower_hours, upper_hours) conformal interval. lower_hours is clipped
            to >= 0; upper_hours is not clipped (Q can be negative when the base
            intervals already over-cover, which is valid and tightens the interval).
        """
        raw_lower, raw_upper = self.predict_intervals(X)
        Q = adjustment.q_adjustment_hours
        conformal_lower = np.clip(raw_lower - Q, 0, None)
        conformal_upper = raw_upper + Q
        return conformal_lower, conformal_upper

    def predict_bucket(self, X: pd.DataFrame) -> tuple[list[str], np.ndarray]:
        """Return (bucket_labels, confidences) for each row.

        bucket_labels: list of strings from BUCKET_LABELS (hours/days/weeks/months/long).
        confidences: float array in [0, 1], the model's probability for the predicted bucket.

        Falls back to the naive majority-class prior when EITHER model_bucket was never
        trained OR this repo's classifier is not trusted (ADR-0025:
        BUCKET_CLASSIFIER_TRUSTED) -- i.e. it's been measured to lose to that same naive
        prior with a CI that excludes zero. A repo whose classifier is proven worse than
        guessing must not be served in preference to guessing.
        """
        trusted = BUCKET_CLASSIFIER_TRUSTED.get(self.repo, True)
        if self.model_bucket is not None and trusted:
            proba  = np.asarray(self.model_bucket.predict(X))  # shape (n, 5)
            idx    = proba.argmax(axis=1)
            labels = [BUCKET_LABELS[i] for i in idx]
            confs  = proba[np.arange(len(idx)), idx]
        else:
            # Naive prior fallback: most frequent bucket from training distribution
            if self.bucket_train_distribution:
                top = max(self.bucket_train_distribution, key=self.bucket_train_distribution.get)  # type: ignore[arg-type]
                top_prob = self.bucket_train_distribution[top]
            else:
                top, top_prob = "days", 0.33
            labels = [top] * len(X)
            confs  = np.full(len(X), top_prob)
        return labels, confs

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        assert self.model_point is not None, "Call fit() first"
        imp = self.model_point.feature_importance(importance_type=importance_type)
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "repo": self.repo,
            "model_point": self.model_point,
            "model_q10": self.model_q10,
            "model_q90": self.model_q90,
            "model_bucket": self.model_bucket,
            "bucket_train_distribution": self.bucket_train_distribution,
            "pca": self.pca,
            "top_components": self.top_components,
            "feature_names": self.feature_names,
            "conformal_adjustments": getattr(self, "conformal_adjustments", {}),
        }, path)
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str) -> "ResolutionTimePredictor":
        data = joblib.load(path)
        obj = cls(repo=data["repo"])
        for k, v in data.items():
            if k != "repo":
                setattr(obj, k, v)
        # Backward-compat: old pkl files lack bucket model fields
        if not hasattr(obj, "model_bucket"):
            obj.model_bucket = None
        if not hasattr(obj, "bucket_train_distribution"):
            obj.bucket_train_distribution = {}
        if not hasattr(obj, "conformal_adjustments"):
            obj.conformal_adjustments = {}
        return obj
