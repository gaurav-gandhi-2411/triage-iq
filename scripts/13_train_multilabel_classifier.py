"""ADR-0036 cutover: train + calibrate the final multi-label component classifier artifacts.

Leakage guard first (hard-fail). Trains MultiLabelTFIDFComponentClassifier (same TF-IDF config
as the shipped single-label baseline, one-vs-rest logistic regression over ALL valid component
labels per issue). Calibrates via a single shared temperature T (scripts/
calibrate_multilabel_classifier.py's corrected top-1-confidence-vs-correctness objective).
Hard stops (same thresholds as 12_calibrate_classifier.py): ECE >= 0.18 -> abort; argmax not
preserved by calibration -> abort (would violate the drop-in-replacement guarantee).

Saves to a STAGING path (component_classifier_{repo}_multilabel_staged.pkl), NOT the production
path -- the swap to the production path (component_classifier_{repo}.pkl, with the old artifact
archived first) is a separate, explicit step, not automatic.

Usage:
  python scripts/13_train_multilabel_classifier.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit as sigmoid

sys.path.insert(0, "src")
from triage_iq.evaluation.classifier_eval import all_matching_component_labels, calibration_analysis  # noqa: E402
from triage_iq.models.component_classifier import MultiLabelTFIDFComponentClassifier, _build_text  # noqa: E402

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPORTS = Path("reports")
ECE_THRESHOLD = 0.18
REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]


def assert_leakage_guard_passed() -> None:
    result = subprocess.run([sys.executable, "scripts/classifier_assert_leakage_guard.py"],
                             capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)
        raise SystemExit("Leakage guard FAILED -- refusing to train.")


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

    classes = sorted(train["component"].unique().tolist())
    Y_train = build_multi_hot(repo, train["labels_raw"], classes)

    clf = MultiLabelTFIDFComponentClassifier(repo=repo)
    clf.fit(X_train, Y_train, classes)

    # Calibrate: single shared T, top-1-confidence-vs-correctness NLL (corrected objective).
    y_val_enc = clf.label_encoder.transform(val["component"])
    logits_val = clf._logits(X_val)

    def top1_confidence_nll(T: float) -> float:
        p = sigmoid(logits_val / T)
        conf = p.max(axis=1)
        correct = (p.argmax(axis=1) == y_val_enc).astype(float)
        conf = np.clip(conf, 1e-15, 1 - 1e-15)
        return float(-np.mean(correct * np.log(conf) + (1 - correct) * np.log(1 - conf)))

    opt = minimize_scalar(top1_confidence_nll, bounds=(0.05, 20.0), method="bounded")
    T_opt = float(opt.x)

    proba_raw_test = clf.predict_proba(X_test)
    proba_cal_test = sigmoid(clf._logits(X_test) / T_opt)

    argmax_preserved = bool(np.array_equal(proba_raw_test.argmax(axis=1), proba_cal_test.argmax(axis=1)))
    cal_after = calibration_analysis(y_test, proba_cal_test, np.array(classes), clf.label_encoder)

    print(f"=== {repo} ===")
    print(f"  T_opt={T_opt:.4f}  argmax_preserved={argmax_preserved}")
    print(f"  post-calibration test ECE={cal_after['ece']:.4f}  mean_conf={cal_after['mean_confidence']:.3f}  "
          f"mean_acc={cal_after['mean_accuracy']:.3f}  overconfident={cal_after['overconfident']}")

    if cal_after["ece"] >= ECE_THRESHOLD:
        raise SystemExit(f"[{repo}] HARD STOP: ECE {cal_after['ece']:.4f} >= {ECE_THRESHOLD}")
    if not argmax_preserved:
        raise SystemExit(f"[{repo}] HARD STOP: calibration did not preserve argmax -- "
                          f"violates drop-in-replacement guarantee.")

    clf.T = T_opt
    staged_path = MODELS_DIR / f"component_classifier_{repo}_multilabel_staged.pkl"
    clf.save(str(staged_path))
    print(f"  Saved staged artifact -> {staged_path}\n")

    return {
        "repo": repo, "T_opt": round(T_opt, 4), "argmax_preserved": argmax_preserved,
        "ece_test": round(cal_after["ece"], 4),
        "mean_conf_test": round(cal_after["mean_confidence"], 4),
        "mean_acc_test": round(cal_after["mean_accuracy"], 4),
        "overconfident": cal_after["overconfident"],
        "staged_path": str(staged_path),
    }


def main() -> None:
    assert_leakage_guard_passed()
    results = {repo: run_repo(repo) for repo in REPOS}
    out_path = REPORTS / "multilabel_classifier_final_training.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
