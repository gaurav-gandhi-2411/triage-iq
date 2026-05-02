"""LightGBM regression for issue resolution time prediction.

Target: log1p(resolution_hours). Heavy-tailed distribution requires log transform.
Quantile regression (Q10/Q90) provides 80% confidence intervals.
All features computed from information available at issue creation time only.
"""

import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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

    # ── Label features ────────────────────────────────────────────
    _component = df["component"] if "component" in df.columns else pd.Series(pd.NA, index=df.index)
    feats["has_component"] = _component.notna().astype(int)
    feats["has_type"] = df["type"].notna().astype(int) if "type" in df.columns else 0
    feats["has_priority"] = df["priority"].notna().astype(int) if "priority" in df.columns else 0
    feats["num_assignees"] = df.get("num_assignees", pd.Series(0, index=df.index)).fillna(0)

    # Component one-hot (top-10 per training set)
    if train_df is not None and "component" in train_df.columns:
        top_components = train_df["component"].value_counts().head(10).index.tolist()
    else:
        top_components = []
    for comp in top_components:
        feats[f"comp_{comp.replace('/', '_').replace('-', '_')}"] = (_component == comp).astype(int)

    # ── Author features (leak-proof: only past info) ──────────────
    if train_df is not None:
        all_df = pd.concat([train_df, df], sort=False).drop_duplicates(subset=["number"])
        all_df = all_df.sort_values("created_at")

        author_prior_count = {}
        author_prior_resolutions = {}
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
    """LightGBM regression for days-to-close prediction.

    Trains three models per repo:
    - Point predictor (MAE, log-scale target)
    - Lower quantile (Q10)
    - Upper quantile (Q90)
    """

    def __init__(self, repo: str) -> None:
        self.repo = repo
        self.model_point: lgb.Booster | None = None
        self.model_q10: lgb.Booster | None = None
        self.model_q90: lgb.Booster | None = None
        self.pca = None
        self.top_components: list[str] = []
        self.feature_names: list[str] = []

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
            callbacks=callbacks,
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

        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point prediction in original hours (not log)."""
        log_pred = self.model_point.predict(X)
        return np.expm1(log_pred).clip(min=0)

    def predict_log(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_point.predict(X)

    def predict_intervals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower_hours, upper_hours) 80% prediction interval."""
        lower = np.expm1(self.model_q10.predict(X)).clip(min=0)
        upper = np.expm1(self.model_q90.predict(X)).clip(min=0)
        return lower, upper

    # ------------------------------------------------------------------
    # Feature importance
    # ------------------------------------------------------------------

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
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
            "pca": self.pca,
            "top_components": self.top_components,
            "feature_names": self.feature_names,
        }, path)
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str) -> "ResolutionTimePredictor":
        data = joblib.load(path)
        obj = cls(repo=data["repo"])
        for k, v in data.items():
            if k != "repo":
                setattr(obj, k, v)
        return obj
