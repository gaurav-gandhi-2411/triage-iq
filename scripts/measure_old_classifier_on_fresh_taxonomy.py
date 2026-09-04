from __future__ import annotations
"""Re-measure the OLD (pre-retrain, 28/35-class) classifier's top-3 accuracy against the SAME
fresh test split (data/processed/{repo}_classifier_test.parquet, 423/671 rows) that scored the
new 47-class retrained classifier -- see ADR-0057. The old classifier's own reported top-3
(89.84%/87.06%, reports/classifier_results.json) was measured on its own stale n=187/n=286 test
set, which is not the same population as the new classifier's n=423/n=671 test set -- the two
numbers are not comparable as reported. This script produces the missing, comparable half of
that comparison: same model, same fresh test rows the new classifier was scored on.

Zero live calls, zero training -- pure local inference against archived pkls.

Usage:
    .venv/Scripts/python.exe scripts/measure_old_classifier_on_fresh_taxonomy.py
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, "src")
from triage_iq.evaluation.classifier_eval import top_k_accuracy, wilson_ci  # noqa: E402
from triage_iq.models.component_classifier import MultiLabelTFIDFComponentClassifier, _build_text  # noqa: E402

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPORTS = Path("reports")
REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]


def run_repo(repo: str) -> dict:
    old_path = MODELS_DIR / f"component_classifier_{repo}_PRE_RETRAIN_2026-09-04.pkl"
    clf = MultiLabelTFIDFComponentClassifier.load(str(old_path))
    old_classes = clf.classes_()

    test = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_test.parquet")
    X_test = _build_text(test["title"], test["body_clean"])
    y_test = test["component"]

    proba = clf.predict_proba_calibrated(X_test)
    top1_acc = float((clf.label_encoder.inverse_transform(proba.argmax(axis=1)) == y_test.to_numpy()).mean())
    top3_acc = top_k_accuracy(y_test, proba, old_classes, k=3)

    n = len(y_test)
    in_taxonomy_mask = y_test.isin(set(old_classes))
    out_of_taxonomy = int((~in_taxonomy_mask).sum())

    # In-taxonomy-only subset: the old classifier's top-3 restricted to rows it could
    # structurally ever get right (gold label within its own class list) -- isolates
    # "did the retrain also change in-taxonomy discrimination quality" from "the old
    # classifier can't score on labels it was never trained to emit."
    in_tax_top3 = top_k_accuracy(y_test[in_taxonomy_mask], proba[in_taxonomy_mask.to_numpy()],
                                  old_classes, k=3)
    n_in_tax = int(in_taxonomy_mask.sum())

    return {
        "repo": repo,
        "old_classifier_path": str(old_path),
        "old_classifier_n_classes": len(old_classes),
        "fresh_test_n": n,
        "fresh_test_out_of_taxonomy_gold_count": out_of_taxonomy,
        "fresh_test_out_of_taxonomy_gold_pct": round(100 * out_of_taxonomy / n, 2),
        "old_classifier_top1_on_fresh_test": round(top1_acc, 4),
        "old_classifier_top1_on_fresh_test_ci95": wilson_ci(top1_acc, n),
        "old_classifier_top3_on_fresh_test": round(top3_acc, 4),
        "old_classifier_top3_on_fresh_test_ci95": wilson_ci(top3_acc, n),
        "old_classifier_top3_on_fresh_test_in_taxonomy_subset_only": round(in_tax_top3, 4),
        "old_classifier_top3_on_fresh_test_in_taxonomy_subset_only_ci95": wilson_ci(in_tax_top3, n_in_tax),
        "in_taxonomy_subset_n": n_in_tax,
    }


def main() -> None:
    results = {repo: run_repo(repo) for repo in REPOS}
    for repo, r in results.items():
        print(f"=== {repo} ===")
        print(f"  fresh test n={r['fresh_test_n']}, "
              f"out-of-taxonomy for OLD classifier: {r['fresh_test_out_of_taxonomy_gold_count']} "
              f"({r['fresh_test_out_of_taxonomy_gold_pct']}%)")
        print(f"  OLD classifier on fresh test: top1={r['old_classifier_top1_on_fresh_test']:.4f}  "
              f"top3={r['old_classifier_top3_on_fresh_test']:.4f}")
        print(f"  OLD classifier top3, in-taxonomy subset only (n={r['in_taxonomy_subset_n']}): "
              f"{r['old_classifier_top3_on_fresh_test_in_taxonomy_subset_only']:.4f}")
    out_path = REPORTS / "old_classifier_on_fresh_taxonomy.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
