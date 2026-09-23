from __future__ import annotations
"""Paired old-vs-new component classifier comparison on the SAME fresh test rows (ADR-0057).

ADR-0057 reported the new classifier's top-3 and the old classifier's top-3 on the fresh split
from two separate runs. This script scores both models on identical rows and adds what a
published claim needs: top-1 as well as top-3, macro-F1, and a paired test (exact McNemar on
the discordant rows) with a paired 95% CI on the difference. Every number the README's
classifier rows cite comes from the JSON this writes.

Zero live calls, zero training -- local inference against the deployed pkls and the archived
pre-retrain pkls.

Usage:
    .venv/Scripts/python.exe scripts/measure_classifier_old_vs_new_paired.py
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score

sys.path.insert(0, "src")
from triage_iq.evaluation.classifier_eval import wilson_ci  # noqa: E402
from triage_iq.models.component_classifier import (  # noqa: E402
    MultiLabelTFIDFComponentClassifier,
    _build_text,
)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
OUT = Path("reports/classifier_old_vs_new_paired_2026-09-23.json")
REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]


def _hits(clf: MultiLabelTFIDFComponentClassifier, X: pd.Series, y: np.ndarray) -> dict:
    proba = clf.predict_proba_calibrated(X)
    classes = np.asarray(clf.classes_())
    order = np.argsort(-proba, axis=1)
    top1 = classes[order[:, 0]]
    top3 = classes[order[:, :3]]
    return {
        "top1_pred": top1,
        "top1_hit": top1 == y,
        "top3_hit": (top3 == y[:, None]).any(axis=1),
        "n_classes": len(classes),
    }


def _paired(new_hit: np.ndarray, old_hit: np.ndarray) -> dict:
    n = len(new_hit)
    b = int((new_hit & ~old_hit).sum())  # new right, old wrong
    c = int((~new_hit & old_hit).sum())  # old right, new wrong
    diff = (b - c) / n
    se = math.sqrt(max((b + c) - (b - c) ** 2 / n, 0.0)) / n
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {
        "new_minus_old_pp": round(100 * diff, 2),
        "ci95_pp": [round(100 * (diff - 1.96 * se), 2), round(100 * (diff + 1.96 * se), 2)],
        "discordant_new_only": b,
        "discordant_old_only": c,
        "mcnemar_exact_p": round(float(p), 6),
    }


def run_repo(repo: str) -> dict:
    new = MultiLabelTFIDFComponentClassifier.load(str(MODELS_DIR / f"component_classifier_{repo}.pkl"))
    old = MultiLabelTFIDFComponentClassifier.load(
        str(MODELS_DIR / f"component_classifier_{repo}_PRE_RETRAIN_2026-09-04.pkl")
    )
    test = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_test.parquet")
    X = _build_text(test["title"], test["body_clean"])
    y = test["component"].to_numpy()
    n = len(y)
    hn, ho = _hits(new, X, y), _hits(old, X, y)

    def summary(h: dict) -> dict:
        t1, t3 = float(h["top1_hit"].mean()), float(h["top3_hit"].mean())
        return {
            "n_classes": h["n_classes"],
            "top1": round(t1, 4), "top1_ci95": wilson_ci(t1, n),
            "top3": round(t3, 4), "top3_ci95": wilson_ci(t3, n),
            "macro_f1_top1": round(float(f1_score(y, h["top1_pred"], average="macro", zero_division=0)), 4),
        }

    out_of_old_taxonomy = int((~pd.Series(y).isin(set(old.classes_()))).sum())
    return {
        "test_rows": n,
        "gold_labels_outside_old_classifier_classes": out_of_old_taxonomy,
        "gold_labels_outside_old_classifier_classes_pct": round(100 * out_of_old_taxonomy / n, 2),
        "gold_labels_outside_new_classifier_classes": int((~pd.Series(y).isin(set(new.classes_()))).sum()),
        "new": summary(hn),
        "old": summary(ho),
        "paired_top1": _paired(hn["top1_hit"], ho["top1_hit"]),
        "paired_top3": _paired(hn["top3_hit"], ho["top3_hit"]),
    }


def main() -> None:
    results = {
        "population": "data/processed/{repo}_classifier_test.parquet (fresh split, ADR-0056/0057)",
        "new_model": "data/models/component_classifier_{repo}.pkl (47-class retrain, MANIFEST.sha256)",
        "old_model": "data/models/component_classifier_{repo}_PRE_RETRAIN_2026-09-04.pkl",
        "per_repo": {repo: run_repo(repo) for repo in REPOS},
    }
    OUT.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
