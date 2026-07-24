"""TF-IDF + Logistic Regression baseline for component classification.

Trained per-repo since label vocabularies differ across projects.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit as _sigmoid
from scipy.special import softmax
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)


class TemperatureScaler:
    """Single-parameter temperature calibrator for multi-class LR pipelines.

    Scales LR decision_function logits by 1/T before softmax.  Preserves
    argmax (test accuracy is unchanged by construction).  T < 1 sharpens
    distributions (fixes underconfidence); T > 1 flattens them.
    """

    def __init__(self, pipeline: Pipeline, T: float) -> None:
        self.pipeline = pipeline
        self.T = T

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        X_tfidf = self.pipeline["tfidf"].transform(X)
        logits = self.pipeline["lr"].decision_function(X_tfidf)
        return softmax(logits / self.T, axis=1)


def _build_text(title: pd.Series, body: pd.Series) -> pd.Series:
    """Concatenate title and body for richer signal."""
    return title.fillna("").str.strip() + ". " + body.fillna("").str.strip()


class TFIDFComponentClassifier:
    """TF-IDF + Logistic Regression baseline for component classification."""

    def __init__(
        self,
        repo: str,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
    ) -> None:
        self.repo = repo
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.pipeline: Pipeline | None = None
        self.label_encoder: LabelEncoder | None = None
        self.calibrator: TemperatureScaler | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.Series,
        y_train: pd.Series,
        X_val: pd.Series | None = None,
        y_val: pd.Series | None = None,
    ) -> TFIDFComponentClassifier:
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y_train)

        self.pipeline = Pipeline([
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=self.max_features,
                    ngram_range=self.ngram_range,
                    stop_words="english",
                    strip_accents="unicode",
                    min_df=2,
                    sublinear_tf=True,
                ),
            ),
            (
                "lr",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    n_jobs=-1,
                    C=1.0,
                    solver="saga",
                ),
            ),
        ])

        t0 = time.perf_counter()
        self.pipeline.fit(X_train, y_enc)
        elapsed = time.perf_counter() - t0
        logger.info("Trained %s in %.1fs", self.repo, elapsed)

        if X_val is not None and y_val is not None:
            from sklearn.metrics import accuracy_score, f1_score
            y_val_enc = self.label_encoder.transform(y_val)
            y_pred = self.pipeline.predict(X_val)
            acc = accuracy_score(y_val_enc, y_pred)
            f1 = f1_score(y_val_enc, y_pred, average="macro", zero_division=0)
            logger.info("Val — accuracy=%.3f  macro_f1=%.3f", acc, f1)

        return self

    def predict(self, X: pd.Series) -> np.ndarray:
        assert self.pipeline is not None, "Model not fitted"
        assert self.label_encoder is not None
        encoded = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(encoded)

    def predict_encoded(self, X: pd.Series) -> np.ndarray:
        assert self.pipeline is not None
        return self.pipeline.predict(X)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        assert self.pipeline is not None
        return self.pipeline.predict_proba(X)

    def predict_proba_calibrated(self, X: pd.Series) -> np.ndarray:
        """Return calibrated probabilities; falls back to predict_proba if no calibrator is set."""
        if self.calibrator is not None:
            return self.calibrator.predict_proba(X)
        return self.predict_proba(X)

    def classes_(self) -> np.ndarray:
        assert self.label_encoder is not None
        return self.label_encoder.classes_

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "pipeline": self.pipeline,
            "label_encoder": self.label_encoder,
            "repo": self.repo,
            "max_features": self.max_features,
            "ngram_range": self.ngram_range,
            "calibrator": self.calibrator,
        }, path)
        logger.info("Saved model to %s", path)

    @classmethod
    def load(cls, path: str) -> TFIDFComponentClassifier:
        data = joblib.load(path)
        obj = cls(repo=data["repo"], max_features=data["max_features"],
                  ngram_range=data["ngram_range"])
        obj.pipeline = data["pipeline"]
        obj.label_encoder = data["label_encoder"]
        obj.calibrator = data.get("calibrator")
        return obj


class MultiLabelTFIDFComponentClassifier:
    """TF-IDF + one-vs-rest Logistic Regression, trained on ALL valid component labels per
    issue (not just the single collapsed one -- ADR-0036). Drop-in replacement for
    TFIDFComponentClassifier: identical public interface (predict/predict_proba/
    predict_proba_calibrated/classes_/save/load), so triage.py needs zero changes to
    consume this class -- only which .pkl gets loaded changes.

    Temperature calibration here is a single shared scalar T dividing every class's
    independent decision_function logit before sigmoid (not a softmax over classes, since
    OvR has no single normalized distribution) -- a shared positive T preserves the
    across-class ranking (sigmoid is monotonic), so argmax/argsort (top-1/top-3) are
    unchanged by calibration, verified empirically in scripts/calibrate_multilabel_classifier.py,
    not just assumed. T is optimized against top-1-confidence-vs-correctness NLL (what ECE
    measures), not full-matrix multi-label BCE -- the latter is dominated by true-negative
    classes and does not target the metric production actually reads.
    """

    def __init__(
        self,
        repo: str,
        max_features: int = 50_000,
        ngram_range: tuple = (1, 2),
    ) -> None:
        self.repo = repo
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer: TfidfVectorizer | None = None
        self.ovr: OneVsRestClassifier | None = None
        self.label_encoder: LabelEncoder | None = None
        self.T: float | None = None  # calibration temperature; None = uncalibrated

    def fit(self, X_train: pd.Series, Y_train_multihot: np.ndarray, classes: list[str]) -> MultiLabelTFIDFComponentClassifier:
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(classes)

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features, ngram_range=self.ngram_range,
            stop_words="english", strip_accents="unicode", min_df=2, sublinear_tf=True,
        )
        X_tfidf = self.vectorizer.fit_transform(X_train)
        self.ovr = OneVsRestClassifier(
            LogisticRegression(class_weight="balanced", max_iter=1000, n_jobs=-1, C=1.0, solver="liblinear"),
            n_jobs=-1,
        )
        self.ovr.fit(X_tfidf, Y_train_multihot)
        return self

    def _logits(self, X: pd.Series) -> np.ndarray:
        assert self.vectorizer is not None and self.ovr is not None
        return self.ovr.decision_function(self.vectorizer.transform(X))

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        return _sigmoid(self._logits(X))

    def predict_proba_calibrated(self, X: pd.Series) -> np.ndarray:
        if self.T is None:
            return self.predict_proba(X)
        return _sigmoid(self._logits(X) / self.T)

    def predict(self, X: pd.Series) -> np.ndarray:
        proba = self.predict_proba_calibrated(X)
        assert self.label_encoder is not None
        return self.label_encoder.inverse_transform(proba.argmax(axis=1))

    def classes_(self) -> np.ndarray:
        assert self.label_encoder is not None
        return self.label_encoder.classes_

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "repo": self.repo, "max_features": self.max_features, "ngram_range": self.ngram_range,
            "vectorizer": self.vectorizer, "ovr": self.ovr, "label_encoder": self.label_encoder,
            "T": self.T, "model_kind": "multilabel_ovr",
        }, path)
        logger.info("Saved multi-label model to %s", path)

    @classmethod
    def load(cls, path: str) -> MultiLabelTFIDFComponentClassifier:
        data = joblib.load(path)
        obj = cls(repo=data["repo"], max_features=data["max_features"], ngram_range=data["ngram_range"])
        obj.vectorizer = data["vectorizer"]
        obj.ovr = data["ovr"]
        obj.label_encoder = data["label_encoder"]
        obj.T = data.get("T")
        return obj
