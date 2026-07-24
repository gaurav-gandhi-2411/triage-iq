"""ADR-0036 cutover step 1: temperature-calibrate the multi-label OvR classifier, then
report how the confidence-keyed abstention threshold's firing rate shifts (GG's addition).

Temperature scaling generalized from ADR-0004's single-softmax method to OvR's independent
per-class binary logits: ONE shared scalar T divides every class's raw decision_function logit
before sigmoid. A shared positive T preserves the across-class ranking (sigmoid is monotonic,
z_A > z_B  =>  z_A/T > z_B/T for T > 0), so argmax/argsort (top-1/top-3) are unchanged by
construction -- same "argmax preserved" property ADR-0004 relied on, generalized correctly, not
assumed. T is optimized by minimizing standard multi-label binary cross-entropy NLL on the val
split's full multi-hot target (every valid label, not just the single collapsed one) -- the
natural objective for what these independent probabilities represent.

Also checks: does the abstention gate's per-repo COMPONENT_CONFIDENCE_THRESHOLD
(kubernetes/kubernetes=0.45, microsoft/vscode=0.29 -- currently gated OFF by default,
ADR-0021) fire at a materially different rate under the OLD (shipped, calibrated single-label)
vs NEW (multi-label, newly calibrated) confidence stream? Reports both firing rates -- does not
retune the threshold.

Usage:
  python scripts/calibrate_multilabel_classifier.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, "src")
from triage_iq.evaluation.classifier_eval import all_matching_component_labels, calibration_analysis  # noqa: E402
from triage_iq.models.abstention import COMPONENT_CONFIDENCE_THRESHOLD  # noqa: E402
from triage_iq.models.component_classifier import TFIDFComponentClassifier, _build_text  # noqa: E402

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPORTS = Path("reports")
TFIDF_KW = dict(max_features=50_000, ngram_range=(1, 2), stop_words="english",
                strip_accents="unicode", min_df=2, sublinear_tf=True)
LR_KW = dict(class_weight="balanced", max_iter=1000, n_jobs=-1, C=1.0, solver="liblinear")
ECE_THRESHOLD = 0.18  # same hard-stop as 12_calibrate_classifier.py / ADR-0004
REPO_KEY = {"microsoft_vscode": "microsoft/vscode", "kubernetes_kubernetes": "kubernetes/kubernetes"}


def build_multi_hot(repo: str, labels_raw_series: pd.Series, classes: list[str]) -> np.ndarray:
    class_to_idx = {c: i for i, c in enumerate(classes)}
    Y = np.zeros((len(labels_raw_series), len(classes)), dtype=np.float32)
    for i, labels_raw in enumerate(labels_raw_series):
        for m in all_matching_component_labels(repo, labels_raw):
            if m in class_to_idx:
                Y[i, class_to_idx[m]] = 1.0
    return Y


def run_repo(repo: str) -> dict:
    train = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_train.parquet")
    val = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_val.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_test.parquet")

    X_train = _build_text(train["title"], train["body_clean"])
    X_val = _build_text(val["title"], val["body_clean"])
    X_test = _build_text(test["title"], test["body_clean"])
    y_test = test["component"]

    le = LabelEncoder()
    le.fit(train["component"])
    classes = list(le.classes_)
    Y_train = build_multi_hot(repo, train["labels_raw"], classes)
    Y_val = build_multi_hot(repo, val["labels_raw"], classes)

    vectorizer = TfidfVectorizer(**TFIDF_KW)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    ovr = OneVsRestClassifier(LogisticRegression(**LR_KW), n_jobs=-1)
    ovr.fit(X_train_tfidf, Y_train)

    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    logits_val = ovr.decision_function(X_val_tfidf)
    logits_test = ovr.decision_function(X_test_tfidf)

    proba_raw_test = sigmoid(logits_test)
    cal_raw = calibration_analysis(y_test, proba_raw_test, np.array(classes), le)

    # Objective corrected: minimizing full-matrix multi-label BCE (all 35 classes x all
    # samples) is dominated by the ~34 true-negative classes per example -- a different,
    # much-easier objective than "is the TOP-1/max confidence reliable," which is the only
    # thing production reads (component_confidence = top-1 value). Directly minimize the
    # binary NLL of (argmax-correct vs reported top-1 confidence) instead -- this is exactly
    # what ECE approximates, so optimizing it directly targets the metric that matters.
    y_val_enc = le.transform(val["component"])

    def top1_confidence_nll(T: float) -> float:
        p = sigmoid(logits_val / T)
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == y_val_enc).astype(float)
        conf = np.clip(conf, 1e-15, 1 - 1e-15)
        return float(-np.mean(correct * np.log(conf) + (1 - correct) * np.log(1 - conf)))

    opt = minimize_scalar(top1_confidence_nll, bounds=(0.05, 20.0), method="bounded")
    T_opt = float(opt.x)

    proba_cal_test = sigmoid(logits_test / T_opt)
    cal_after = calibration_analysis(y_test, proba_cal_test, np.array(classes), le)

    # Argmax-preservation check (must hold by construction -- verify, don't just assert).
    argmax_raw = proba_raw_test.argmax(axis=1)
    argmax_cal = proba_cal_test.argmax(axis=1)
    argmax_preserved = bool(np.array_equal(argmax_raw, argmax_cal))

    print(f"=== {repo} ===")
    print(f"  T_opt={T_opt:.4f}")
    print(f"  RAW (uncalibrated):       ECE={cal_raw['ece']:.4f}  mean_conf={cal_raw['mean_confidence']:.3f}  "
          f"mean_acc={cal_raw['mean_accuracy']:.3f}  overconfident={cal_raw['overconfident']}")
    print(f"  CALIBRATED (T={T_opt:.3f}): ECE={cal_after['ece']:.4f}  mean_conf={cal_after['mean_confidence']:.3f}  "
          f"mean_acc={cal_after['mean_accuracy']:.3f}  overconfident={cal_after['overconfident']}")
    print(f"  argmax preserved by calibration: {argmax_preserved}")

    # Hard stop, same threshold as 12_calibrate_classifier.py
    hard_stop = cal_after["ece"] >= ECE_THRESHOLD
    if hard_stop:
        print(f"  HARD STOP: ECE {cal_after['ece']:.4f} >= threshold {ECE_THRESHOLD}")

    # --- Abstention threshold firing-rate check (GG's addition) ---
    repo_key = REPO_KEY[repo]
    threshold = COMPONENT_CONFIDENCE_THRESHOLD.get(repo_key)
    baseline = TFIDFComponentClassifier.load(str(MODELS_DIR / f"component_classifier_{repo}.pkl"))
    base_proba = baseline.predict_proba_calibrated(X_test)
    base_top1_conf = base_proba.max(axis=1)
    new_top1_conf = proba_cal_test.max(axis=1)

    old_fire_rate = float((base_top1_conf < threshold).mean()) if threshold is not None else None
    new_fire_rate = float((new_top1_conf < threshold).mean()) if threshold is not None else None
    print(f"  abstention threshold ({repo_key}={threshold}): "
          f"OLD fires {old_fire_rate*100:.1f}%  NEW fires {new_fire_rate*100:.1f}%  "
          f"(gate is OFF by default in prod, ADR-0021)")
    print()

    return {
        "repo": repo, "T_opt": round(T_opt, 4),
        "ece_raw_test": round(cal_raw["ece"], 4), "ece_calibrated_test": round(cal_after["ece"], 4),
        "mean_conf_calibrated": round(cal_after["mean_confidence"], 4),
        "mean_acc_calibrated": round(cal_after["mean_accuracy"], 4),
        "overconfident_calibrated": cal_after["overconfident"],
        "argmax_preserved": argmax_preserved,
        "hard_stop_triggered": hard_stop,
        "abstention_threshold": threshold,
        "abstention_fire_rate_old_baseline": old_fire_rate,
        "abstention_fire_rate_new_multilabel": new_fire_rate,
        "abstention_gate_live_in_prod": False,
    }


def main() -> None:
    results = {}
    for repo in ["microsoft_vscode", "kubernetes_kubernetes"]:
        results[repo] = run_repo(repo)
    out_path = REPORTS / "tfidf_multilabel_calibration_and_threshold_check.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
