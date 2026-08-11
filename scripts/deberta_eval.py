"""Phase B: evaluate a trained DeBERTa arm (single-label or multi-label) against the test set,
using the SAME evaluate_classifier() methodology as the TF-IDF baseline and the DistilBERT
re-eval -- top-3 (primary), top-1, any-valid-label top-1 credit, macro-F1, per-class recall.

Usage:
  python scripts/deberta_eval.py --arm single --repo microsoft_vscode
  python scripts/deberta_eval.py --arm multi --repo microsoft_vscode
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, "src")
from triage_iq.evaluation.classifier_eval import evaluate_classifier  # noqa: E402
from triage_iq.models.component_classifier import _build_text  # noqa: E402

MAX_LEN = 512
TAIL_THRESHOLD = 15  # classes with fewer than this many TRAIN examples are "tail"
PROCESSED_DIR = Path("data/processed")
REPORTS = Path("reports")


class _DebertaAdapter:
    def __init__(self, model_dir: str, multi_label: bool) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.multi_label = multi_label
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device).eval()
        self.label_encoder = joblib.load(Path(model_dir) / "label_encoder.pkl")

    def _proba(self, X: pd.Series) -> np.ndarray:
        texts = X.tolist()
        all_probs = []
        batch_size = 16
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                enc = self.tokenizer(
                    batch, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt"
                ).to(self.device)
                logits = self.model(**enc).logits
                if self.multi_label:
                    probs = torch.sigmoid(logits).cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                all_probs.append(probs)
        return np.concatenate(all_probs, axis=0)

    def predict_proba(self, X: pd.Series) -> np.ndarray:
        return self._proba(X)

    def predict(self, X: pd.Series) -> np.ndarray:
        proba = self._proba(X)
        return self.label_encoder.inverse_transform(proba.argmax(axis=1))

    def classes_(self) -> np.ndarray:
        return self.label_encoder.classes_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["single", "multi"])
    ap.add_argument("--repo", required=True, choices=["microsoft_vscode", "kubernetes_kubernetes"])
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    suffix = f"_{args.run_name}" if args.run_name else ""
    model_dir = f"data/models/deberta_{args.arm}_{args.repo}{suffix}"
    train_df = pd.read_parquet(PROCESSED_DIR / f"{args.repo}_classifier_train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / f"{args.repo}_classifier_test.parquet")

    X_test = _build_text(test_df["title"], test_df["body_clean"])
    y_test = test_df["component"]

    clf = _DebertaAdapter(model_dir, multi_label=(args.arm == "multi"))
    r = evaluate_classifier(clf, X_test, y_test, repo=args.repo, labels_raw=test_df["labels_raw"])

    train_support = train_df["component"].value_counts()
    tail_classes = train_support[train_support < TAIL_THRESHOLD].index.tolist()
    tail_recall = {
        c: r["per_class_metrics"][c]["recall"]
        for c in tail_classes
        if c in r["per_class_metrics"]
    }

    print(f"=== {args.arm} / {args.repo} ===")
    print(f"  top1={r['top1_accuracy']:.4f} {r['top1_accuracy_ci95']}")
    print(f"  top3={r['top3_accuracy']:.4f} {r['top3_accuracy_ci95']}  <- PRIMARY (ship bar)")
    print(f"  macro_f1={r['macro_f1']:.4f}  weighted_f1={r['weighted_f1']:.4f}")
    print(f"  any_valid_label_top1={r['multi_label_credit_accuracy']:.4f} {r['multi_label_credit_accuracy_ci95']}")
    print(f"  n_multi_label_test_rows={r['n_multi_label_test_rows']}/{len(test_df)}")
    print(f"  tail classes (<{TAIL_THRESHOLD} train examples): {len(tail_classes)}")
    for c, rec in sorted(tail_recall.items(), key=lambda kv: kv[1]):
        print(f"    {c} (train_n={train_support[c]}): recall={rec:.3f}")

    out = {
        "arm": args.arm, "repo": args.repo, "n_test": len(test_df),
        "top1_accuracy": r["top1_accuracy"], "top1_accuracy_ci95": r["top1_accuracy_ci95"],
        "top3_accuracy": r["top3_accuracy"], "top3_accuracy_ci95": r["top3_accuracy_ci95"],
        "macro_f1": r["macro_f1"], "weighted_f1": r["weighted_f1"],
        "any_valid_label_top1_accuracy": r["multi_label_credit_accuracy"],
        "any_valid_label_top1_accuracy_ci95": r["multi_label_credit_accuracy_ci95"],
        "n_multi_label_test_rows": r["n_multi_label_test_rows"],
        "tail_classes_recall": {c: round(rec, 4) for c, rec in tail_recall.items()},
        "tail_threshold": TAIL_THRESHOLD,
    }
    out_path = REPORTS / f"deberta_eval_{args.arm}_{args.repo}{suffix}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
