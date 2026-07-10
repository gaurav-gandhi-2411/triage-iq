"""Unit tests for src/triage_iq/models/resolution.py's bucket-classifier trust gate (ADR-0025).

Uses a minimal stub in place of a real LightGBM Booster so these tests isolate
predict_bucket()'s trust-gating logic from the classifier's own training/behavior.
No real model files, no training -- fast, deterministic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from triage_iq.models.resolution import BUCKET_CLASSIFIER_TRUSTED, BUCKET_LABELS, ResolutionTimePredictor


class _StubBucketModel:
    """Always predicts 'long' (index 4) with 0.9 confidence -- deliberately different from
    the naive fixture's majority class, so trust-gating is distinguishable from a coincidence."""

    def predict(self, X):
        n = len(X)
        proba = np.zeros((n, len(BUCKET_LABELS)))
        proba[:, 4] = 0.9
        proba[:, 0] = 0.1
        return proba


def _make_predictor(repo: str) -> ResolutionTimePredictor:
    p = ResolutionTimePredictor(repo=repo)
    p.model_bucket = _StubBucketModel()
    p.bucket_train_distribution = {
        "hours": 0.5, "days": 0.2, "weeks": 0.15, "months": 0.1, "long": 0.05,
    }
    return p


def test_trusted_repo_uses_trained_classifier() -> None:
    """k8s is trusted (ADR-0025: CI excludes zero, positive) -- serves the real classifier."""
    p = _make_predictor("kubernetes_kubernetes")
    X = pd.DataFrame({"f": [1, 2, 3]})
    labels, confs = p.predict_bucket(X)
    assert labels == ["long", "long", "long"]
    assert all(c == 0.9 for c in confs)


def test_untrusted_repo_falls_back_to_naive() -> None:
    """vscode is untrusted (ADR-0025: CI excludes zero, WRONG direction) -- falls back to the
    naive majority-class prior even though model_bucket is present and predicts something else."""
    p = _make_predictor("microsoft_vscode")
    X = pd.DataFrame({"f": [1, 2, 3]})
    labels, confs = p.predict_bucket(X)
    assert labels == ["hours", "hours", "hours"]
    assert all(c == 0.5 for c in confs)


def test_unmeasured_repo_defaults_to_trusted() -> None:
    """A repo not in BUCKET_CLASSIFIER_TRUSTED defaults to trusted (unmeasured, not proven
    untrustworthy) -- preserves historical behavior for anything not yet evaluated."""
    assert "some_future_repo" not in BUCKET_CLASSIFIER_TRUSTED
    p = _make_predictor("some_future_repo")
    X = pd.DataFrame({"f": [1, 2, 3]})
    labels, _confs = p.predict_bucket(X)
    assert labels == ["long", "long", "long"]


def test_current_trust_dict_matches_measured_repos() -> None:
    """Pins the exact, current ADR-0025 decision -- a regression guard: if this ever changes,
    it should be a deliberate, reviewed edit, not a silent drive-by."""
    assert BUCKET_CLASSIFIER_TRUSTED == {
        "kubernetes_kubernetes": True,
        "microsoft_vscode": False,
    }


def test_model_bucket_none_falls_back_regardless_of_trust() -> None:
    """Untrained model_bucket (None) always falls back, independent of the trust flag --
    k8s is trusted=True here but has no model_bucket, so naive still applies."""
    p = ResolutionTimePredictor(repo="kubernetes_kubernetes")
    p.bucket_train_distribution = {"hours": 0.6, "days": 0.4}
    X = pd.DataFrame({"f": [1, 2]})
    labels, confs = p.predict_bucket(X)
    assert labels == ["hours", "hours"]
    assert all(c == 0.6 for c in confs)
