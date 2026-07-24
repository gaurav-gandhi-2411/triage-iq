"""CPU-only test of GG's supervision-fix hypothesis: does fixing the label-collapse defect
(normalize_labels() keeps only the first valid component label; 30.4% of k8s test issues /
8.0% of vscode's have >1) matter more than architecture?

Retrains the CURRENT TF-IDF+LR baseline as MULTI-LABEL: one-vs-rest logistic regression over
ALL valid component labels per issue (multi-hot targets built from labels_raw via
classifier_eval.py::all_matching_component_labels -- the same function that measured the
collapse and that the DeBERTa ARM 2 multi-label arm uses), instead of the single collapsed
`component` column. SAME TF-IDF feature extraction as the shipped baseline
(component_classifier.py::TFIDFComponentClassifier) -- only the supervision changes.

Leakage guard: scripts/classifier_assert_leakage_guard.py asserted as a hard pre-flight gate.
Evaluated with the exact same evaluate_classifier() methodology as the TF-IDF baseline,
DistilBERT re-eval, and the DeBERTa arms -- top-3 primary (ship bar: beat 82.5% k8s / 90.4%
vscode, CI clearly excluding zero), top-1, any-valid-label top-1, macro-F1, per-class recall
for tail classes (<15 train examples).

Usage:
  python scripts/tfidf_multilabel.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "src")
from triage_iq.evaluation.classifier_eval import evaluate_classifier  # noqa: E402
from triage_iq.models.component_classifier import TFIDFComponentClassifier, _build_text  # noqa: E402

sys.path.insert(0, "scripts")
from _retrieval_eval_common import N_BOOTSTRAP, SEED, paired_bootstrap_ci  # noqa: E402

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPORTS = Path("reports")
TAIL_THRESHOLD = 15
REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]

# Identical TF-IDF settings to the shipped TFIDFComponentClassifier -- only supervision changes.
TFIDF_KW = dict(
    max_features=50_000, ngram_range=(1, 2), stop_words="english",
    strip_accents="unicode", min_df=2, sublinear_tf=True,
)
# NOTE: solver differs from the shipped single-label baseline's 'saga' -- diagnosed, not
# arbitrary. 'saga' never converges on the per-class OvR binary sub-problems (confirmed even
# at max_iter=5000: identical degenerate collapse to ~15/35 classes ever predicted). 'saga' is
# tuned for the ORIGINAL single 35-way multi-class softmax problem; wrapped in OneVsRestClassifier
# it solves 35 independent, far-more-imbalanced binary problems, a different optimization
# landscape 'saga' doesn't handle here. 'liblinear' (sklearn's standard recommendation for
# per-class L2-regularized binary logistic regression) converges cleanly and uses the full
# label space (27-30/35 classes actually predicted, vs saga's 15/35).
LR_KW = dict(class_weight="balanced", max_iter=1000, n_jobs=-1, C=1.0, solver="liblinear")


def assert_leakage_guard_passed() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/classifier_assert_leakage_guard.py"],
        capture_output=True, text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("Leakage guard FAILED -- refusing to train.")


def build_multi_hot(repo: str, labels_raw_series: pd.Series, classes: list[str]) -> np.ndarray:
    from triage_iq.evaluation.classifier_eval import all_matching_component_labels
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(labels_raw_series), len(classes)), dtype=np.float32)
    for i, labels_raw in enumerate(labels_raw_series):
        for m in all_matching_component_labels(repo, labels_raw):
            if m in class_to_idx:
                Y[i, class_to_idx[m]] = 1.0
    return Y


class _MultiLabelTFIDFAdapter:
    """Adapter satisfying evaluate_classifier()'s interface for the OvR multi-label model."""

    def __init__(self, vectorizer: TfidfVectorizer, ovr: OneVsRestClassifier, label_encoder: LabelEncoder) -> None:
        self.vectorizer = vectorizer
        self.ovr = ovr
        self.label_encoder = label_encoder

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        X_tfidf = self.vectorizer.transform(X)
        return self.ovr.predict_proba(X_tfidf)

    def predict(self, X: pd.Series) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.label_encoder.inverse_transform(proba.argmax(axis=1))

    def classes_(self) -> np.ndarray:
        return self.label_encoder.classes_


def hit_vectors(proba: np.ndarray, classes: np.ndarray, y_test: pd.Series, k: int) -> np.ndarray:
    top_k_idx = np.argsort(-proba, axis=1)[:, :k]
    top_k_labels = classes[top_k_idx]
    y_arr = np.asarray(y_test)
    return np.array([float(y_arr[i] in top_k_labels[i]) for i in range(len(y_arr))])


def run_repo(repo: str) -> dict:
    train = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_test.parquet")

    X_train = _build_text(train["title"], train["body_clean"])
    X_test = _build_text(test["title"], test["body_clean"])
    y_test = test["component"]

    # Load the single-label baseline for a paired comparison -- never retrain it here
    # (LogisticRegression's saga solver has no fixed seed; a past incident overwrote the
    # shipped artifact non-reproducibly by retraining directly instead of loading). Post-ADR-0036
    # cutover, component_classifier_{repo}.pkl IS the multi-label model -- the single-label
    # baseline now only exists at the archived _PRE_MULTILABEL.pkl path.
    baseline = TFIDFComponentClassifier.load(str(MODELS_DIR / f"component_classifier_{repo}_PRE_MULTILABEL.pkl"))
    base_proba = baseline.predict_proba(X_test)
    base_classes = baseline.classes_()
    base_top3 = hit_vectors(base_proba, base_classes, y_test, k=3)
    base_top1 = hit_vectors(base_proba, base_classes, y_test, k=1)

    le = LabelEncoder()
    le.fit(train["component"])
    classes = list(le.classes_)

    Y_train = build_multi_hot(repo, train["labels_raw"], classes)
    n_multi_train = int((Y_train.sum(axis=1) > 1).sum())
    print(f"[{repo}] train multi-label rows: {n_multi_train}/{len(train)} "
          f"({100*n_multi_train/len(train):.1f}%)")

    vectorizer = TfidfVectorizer(**TFIDF_KW)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    ovr = OneVsRestClassifier(LogisticRegression(**LR_KW), n_jobs=-1)
    ovr.fit(X_train_tfidf, Y_train)

    clf = _MultiLabelTFIDFAdapter(vectorizer, ovr, le)
    r = evaluate_classifier(clf, X_test, y_test, repo=repo, labels_raw=test["labels_raw"])

    # Paired bootstrap vs the EXISTING shipped single-label baseline, same test set, same
    # resample indices (ADR-0027/ADR-0035's established method) -- ship bar: meaningful top-3
    # lift with CI clearly excluding zero.
    multi_proba = clf.predict_proba(X_test)
    multi_classes = clf.classes_()
    multi_top3 = hit_vectors(multi_proba, multi_classes, y_test, k=3)
    multi_top1 = hit_vectors(multi_proba, multi_classes, y_test, k=1)
    top3_lo, top3_hi, top3_delta = paired_bootstrap_ci(base_top3, multi_top3)
    top1_lo, top1_hi, top1_delta = paired_bootstrap_ci(base_top1, multi_top1)
    top3_ships = top3_lo > 0

    train_support = train["component"].value_counts()
    tail_classes = train_support[train_support < TAIL_THRESHOLD].index.tolist()
    tail_recall = {
        c: r["per_class_metrics"][c]["recall"]
        for c in tail_classes
        if c in r["per_class_metrics"]
    }

    print(f"=== {repo} (TF-IDF+LR, MULTI-LABEL OvR vs shipped single-label baseline) ===")
    print(f"  baseline (single-label, shipped): top1={base_top1.mean():.4f}  top3={base_top3.mean():.4f}")
    print(f"  multi-label: top1={r['top1_accuracy']:.4f} {r['top1_accuracy_ci95']}")
    print(f"  multi-label: top3={r['top3_accuracy']:.4f} {r['top3_accuracy_ci95']}  <- PRIMARY (ship bar)")
    print(f"  PAIRED top3 delta={top3_delta:+.4f}  CI95=[{top3_lo:+.4f},{top3_hi:+.4f}]  ships={top3_ships}")
    print(f"  PAIRED top1 delta={top1_delta:+.4f}  CI95=[{top1_lo:+.4f},{top1_hi:+.4f}]")
    print(f"  macro_f1={r['macro_f1']:.4f}  weighted_f1={r['weighted_f1']:.4f}")
    print(f"  any_valid_label_top1={r['multi_label_credit_accuracy']:.4f} {r['multi_label_credit_accuracy_ci95']}")
    print(f"  n_multi_label_test_rows={r['n_multi_label_test_rows']}/{len(test)}")
    print(f"  tail classes (<{TAIL_THRESHOLD} train examples): {len(tail_classes)}")
    for c, rec in sorted(tail_recall.items(), key=lambda kv: kv[1]):
        print(f"    {c} (train_n={train_support[c]}): recall={rec:.3f}")
    print()

    return {
        "repo": repo, "n_train": len(train), "n_test": len(test),
        "n_multi_label_train_rows": n_multi_train,
        "baseline_top1_accuracy": float(base_top1.mean()),
        "baseline_top3_accuracy": float(base_top3.mean()),
        "top1_accuracy": r["top1_accuracy"], "top1_accuracy_ci95": r["top1_accuracy_ci95"],
        "top3_accuracy": r["top3_accuracy"], "top3_accuracy_ci95": r["top3_accuracy_ci95"],
        "top3_delta_vs_baseline_paired": round(top3_delta, 4),
        "top3_delta_ci95_paired": [round(top3_lo, 4), round(top3_hi, 4)],
        "top3_ships": bool(top3_ships),
        "top1_delta_vs_baseline_paired": round(top1_delta, 4),
        "top1_delta_ci95_paired": [round(top1_lo, 4), round(top1_hi, 4)],
        "macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"],
        "any_valid_label_top1_accuracy": r["multi_label_credit_accuracy"],
        "any_valid_label_top1_accuracy_ci95": r["multi_label_credit_accuracy_ci95"],
        "n_multi_label_test_rows": r["n_multi_label_test_rows"],
        "tail_classes_recall": {c: round(rec, 4) for c, rec in tail_recall.items()},
        "tail_threshold": TAIL_THRESHOLD,
        "bootstrap": {"n_resamples": N_BOOTSTRAP, "seed": SEED, "method": "paired percentile"},
    }


def main() -> None:
    assert_leakage_guard_passed()
    results = {repo: run_repo(repo) for repo in REPOS}
    out_path = REPORTS / "tfidf_multilabel_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
