"""Tests for the classifier eval top-3 / multi-label-credit fix (ADR-0028 Phase B2).

classifier_eval.py previously reported only top-1 accuracy, even though the product
never surfaces a single label -- src/triage_iq/models/triage.py builds classifier_top3
and src/triage_iq/models/grounding.py defines "correct" as top-3 membership. This file
covers the new top_k_accuracy, all_matching_component_labels, and the multi-label
credit metric now computed by evaluate_classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from triage_iq.evaluation.classifier_eval import (
    all_matching_component_labels,
    evaluate_classifier,
    top_k_accuracy,
    wilson_ci,
)


class TestWilsonCi:

    def test_returns_lower_le_p_le_upper(self):
        lo, hi = wilson_ci(0.7, 100)
        assert lo < 0.7 < hi

    def test_narrower_with_more_n(self):
        lo_small, hi_small = wilson_ci(0.7, 20)
        lo_big, hi_big = wilson_ci(0.7, 2000)
        assert (hi_big - lo_big) < (hi_small - lo_small)

    def test_zero_n_returns_zero_zero(self):
        assert wilson_ci(0.5, 0) == (0.0, 0.0)


class TestTopKAccuracy:

    def test_top1_equivalent_when_k_is_1(self):
        y_test = pd.Series(["a", "b", "c"])
        classes = np.array(["a", "b", "c"])
        # Row-wise argmax matches true label for rows 0, 2 only
        y_proba = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ])
        acc = top_k_accuracy(y_test, y_proba, classes, k=1)
        assert acc == pytest.approx(2 / 3)

    def test_top3_credits_true_label_anywhere_in_topk(self):
        y_test = pd.Series(["a", "b"])
        classes = np.array(["a", "b", "c", "d"])
        y_proba = np.array([
            [0.1, 0.1, 0.4, 0.4],  # true='a' ranked 4th (lowest) -> miss at k=3
            [0.1, 0.3, 0.3, 0.3],  # true='b' ranked 2nd -> hit at k=3
        ])
        acc_k3 = top_k_accuracy(y_test, y_proba, classes, k=3)
        assert acc_k3 == pytest.approx(0.5)

    def test_top3_never_less_than_top1(self):
        rng = np.random.default_rng(42)
        y_proba = rng.dirichlet(np.ones(5), size=50)
        classes = np.array(["a", "b", "c", "d", "e"])
        y_test = pd.Series(classes[y_proba.argmax(axis=1)])  # all top-1 hits by construction
        acc1 = top_k_accuracy(y_test, y_proba, classes, k=1)
        acc3 = top_k_accuracy(y_test, y_proba, classes, k=3)
        assert acc3 >= acc1


class TestAllMatchingComponentLabels:

    def test_regex_pattern_single_match(self):
        result = all_matching_component_labels(
            "kubernetes_kubernetes", ["area/kubectl", "kind/bug"]
        )
        assert result == {"kubectl"}

    def test_regex_pattern_multiple_matches(self):
        result = all_matching_component_labels(
            "kubernetes_kubernetes", ["area/kubectl", "area/apiserver", "kind/bug"]
        )
        assert result == {"kubectl", "apiserver"}

    def test_list_pattern_case_insensitive(self):
        result = all_matching_component_labels("microsoft_vscode", ["Terminal", "bug"])
        assert result == {"Terminal"}

    def test_no_match_returns_empty_set(self):
        result = all_matching_component_labels("kubernetes_kubernetes", ["kind/bug"])
        assert result == set()

    def test_unknown_repo_returns_empty_set(self):
        result = all_matching_component_labels("unknown_repo", ["area/foo"])
        assert result == set()


class _StubModel:
    """Minimal stand-in for TFIDFComponentClassifier — fixed predictions/probabilities."""

    def __init__(self, y_pred, y_proba, classes):
        self._y_pred = np.asarray(y_pred)
        self._y_proba = y_proba
        self._classes = classes

        class _LE:
            def transform(_self, labels):
                return np.array([list(classes).index(x) for x in labels])

        self.label_encoder = _LE()

    def predict(self, X):
        return self._y_pred

    def predict_proba(self, X):
        return self._y_proba

    def classes_(self):
        return self._classes


class TestEvaluateClassifierTop3AndMultiLabel:

    def test_top3_reported_and_gte_top1(self):
        classes = np.array(["a", "b", "c"])
        y_test = pd.Series(["a", "b", "c", "a"])
        y_pred = np.array(["b", "b", "c", "c"])  # 2/4 top-1 hits
        y_proba = np.array([
            [0.3, 0.35, 0.35],  # true 'a' rank 3rd -> top1 miss, top3 hit
            [0.1, 0.8, 0.1],    # true 'b' rank 1st -> hit
            [0.1, 0.1, 0.8],    # true 'c' rank 1st -> hit
            [0.2, 0.3, 0.5],    # true 'a' rank 3rd -> top1 miss, top3 hit
        ])
        model = _StubModel(y_pred, y_proba, classes)
        X_test = pd.Series(["x"] * 4)
        result = evaluate_classifier(model, X_test, y_test)

        assert result["top1_accuracy"] == pytest.approx(0.5)
        assert result["top3_accuracy"] == pytest.approx(1.0)
        assert result["top3_accuracy"] >= result["top1_accuracy"]
        assert "multi_label_credit_accuracy" not in result  # no repo/labels_raw given

    def test_multi_label_credit_recovers_collapsed_alternate_label(self):
        classes = np.array(["kubectl", "apiserver"])
        y_test = pd.Series(["kubectl"])  # collapsed gold: first-match-wins kept "kubectl"
        y_pred = np.array(["apiserver"])  # predicted the OTHER valid label -> top-1 "miss"
        y_proba = np.array([[0.4, 0.6]])
        model = _StubModel(y_pred, y_proba, classes)
        X_test = pd.Series(["x"])
        labels_raw = pd.Series([["area/kubectl", "area/apiserver", "kind/bug"]])

        result = evaluate_classifier(
            model, X_test, y_test, repo="kubernetes_kubernetes", labels_raw=labels_raw
        )

        assert result["top1_accuracy"] == pytest.approx(0.0)  # naive top-1 says wrong
        assert result["multi_label_credit_accuracy"] == pytest.approx(1.0)  # actually valid
        assert result["n_multi_label_test_rows"] == 1
        assert result["multi_label_test_row_rate"] == pytest.approx(1.0)

    def test_single_label_row_not_counted_as_multi_label(self):
        classes = np.array(["kubectl", "apiserver"])
        y_test = pd.Series(["kubectl"])
        y_pred = np.array(["kubectl"])
        y_proba = np.array([[0.9, 0.1]])
        model = _StubModel(y_pred, y_proba, classes)
        X_test = pd.Series(["x"])
        labels_raw = pd.Series([["area/kubectl", "kind/bug"]])

        result = evaluate_classifier(
            model, X_test, y_test, repo="kubernetes_kubernetes", labels_raw=labels_raw
        )

        assert result["n_multi_label_test_rows"] == 0
        assert result["multi_label_credit_accuracy"] == pytest.approx(1.0)
