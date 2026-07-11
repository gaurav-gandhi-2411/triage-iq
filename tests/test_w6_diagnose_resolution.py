"""Tests for scripts/w6_diagnose_resolution.py's bucket-classifier diagnostic (ADR-0028 Phase B4).

ADR-0025 wired BUCKET_CLASSIFIER_TRUSTED into ResolutionTimePredictor.predict_bucket(), which
now serves the naive majority-class fallback for any untrusted repo (vscode). The diagnostic
script's job is to independently re-verify the RAW trained classifier against naive -- that's
the whole basis for the trust decision. If it called predict_bucket() instead of the raw
model_bucket.predict(), it would measure the naive prediction against itself for an untrusted
repo (a tautological 0.0pp delta) and could never again re-verify the decision it justifies.
This file proves raw_bucket_accuracy_vs_naive() uses the raw classifier, not the gated one.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage_iq.models.resolution import BUCKET_LABELS, ResolutionTimePredictor  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "w6_diagnose_resolution", Path(__file__).parent.parent / "scripts" / "w6_diagnose_resolution.py"
)
w6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(w6)


class _StubBucketModel:
    """Always predicts 'long' (index 4) -- deliberately different from the naive fixture's
    majority class ('hours'), so raw-vs-gated behavior is distinguishable, not coincidental."""

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


class TestRawBucketAccuracyVsNaive:

    def test_untrusted_repo_still_measures_raw_classifier(self):
        """vscode is untrusted (ADR-0025) -- predict_bucket() would return naive for every
        row, but this diagnostic must measure the RAW classifier's real accuracy instead."""
        p = _make_predictor("microsoft_vscode")
        X = pd.DataFrame({"f": [1, 2, 3, 4]})
        # All 4 true buckets are 'hours' (idx 0) -- naive majority ('hours') would score
        # 100%, but the stub classifier always predicts 'long' (idx 4) -- 0% raw accuracy.
        true_bucket_idx = np.array([0, 0, 0, 0])

        result = w6.raw_bucket_accuracy_vs_naive(p, X, true_bucket_idx)

        assert result["trained_accuracy"] == 0.0  # raw classifier: always wrong here
        assert result["naive_accuracy"] == 1.0  # naive majority ('hours') always right here
        assert result["accuracy_delta_bootstrap"]["mean"] == pytest.approx(-1.0)

    def test_raw_accuracy_diverges_from_gated_predict_bucket(self):
        """Proves the regression this fix guards against: predict_bucket() (gated) and
        raw_bucket_accuracy_vs_naive() (raw) must disagree for an untrusted repo with a
        classifier that predicts something other than the naive majority class."""
        p = _make_predictor("microsoft_vscode")
        X = pd.DataFrame({"f": [1, 2, 3]})
        true_bucket_idx = np.array([4, 4, 4])  # true bucket is 'long' -- matches the stub

        gated_labels, _ = p.predict_bucket(X)
        assert gated_labels == ["hours", "hours", "hours"]  # gated: naive fallback, all wrong

        raw_result = w6.raw_bucket_accuracy_vs_naive(p, X, true_bucket_idx)
        assert raw_result["trained_accuracy"] == 1.0  # raw: stub always predicts 'long', all right

    def test_trusted_repo_raw_and_gated_agree(self):
        """k8s is trusted -- predict_bucket() serves the real classifier too, so raw and
        gated should agree (sanity check that the fix doesn't change trusted-repo behavior)."""
        p = _make_predictor("kubernetes_kubernetes")
        X = pd.DataFrame({"f": [1, 2, 3]})
        true_bucket_idx = np.array([4, 4, 4])

        gated_labels, _ = p.predict_bucket(X)
        assert gated_labels == ["long", "long", "long"]

        raw_result = w6.raw_bucket_accuracy_vs_naive(p, X, true_bucket_idx)
        assert raw_result["trained_accuracy"] == 1.0

    def test_missing_model_bucket_raises(self):
        p = ResolutionTimePredictor(repo="microsoft_vscode")
        p.bucket_train_distribution = {"hours": 1.0}
        X = pd.DataFrame({"f": [1]})
        with pytest.raises(AssertionError, match="no trained bucket classifier"):
            w6.raw_bucket_accuracy_vs_naive(p, X, np.array([0]))
