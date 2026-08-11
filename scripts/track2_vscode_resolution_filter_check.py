"""One-off investigation script (2026-08-11 session, not wired into CI/eval-gate).

Follow-up to ADR-0041's disclosed-but-not-actioned finding: the vscode lever3 re-split test
window (Feb 2025-Apr 2026) is 83.8% "hours"-bucket, driven partly by a duplicate-report wave
(near-identical "Terminal not working"-style reports, #240070-240145) and bot-authored churn
(vs-code-engineering[bot]). This script quantifies, rather than assumes:

  1. What fraction of the test window's "hours"-bucket skew survives once (a) bot-authored
     issues and (b) a rule-based duplicate-wave filter are removed.
  2. Whether the ALREADY-TRAINED lever3 model's accuracy-vs-naive delta changes once measured
     on the filtered population, using the exact same bucket_accuracy_vs_naive() methodology
     as scripts/lever3_train_resolution.py (bootstrapped CI, same bucket boundaries).

Duplicate-wave filter (disclosed, rule-based, not hand-picked per-issue): title (lowercased)
contains "terminal" AND matches a short list of generic-complaint phrases ("not working",
"not responding", "doesn't work", "won't work", "issue", "problem", "crash", "freeze",
"blur", or is empty/near-empty after stripping "terminal"), AND resolution_hours < 24. This
is a precision-over-recall filter: it will miss some duplicate-wave issues with idiosyncratic
titles and may drop a few genuine fast-fixed terminal bugs, but every dropped row is logged
for inspection, not silently discarded.

Reads:  data/processed/microsoft_vscode_temporal_{train,test}_lever3.parquet
        data/models/resolution_predictor_microsoft_vscode_lever3.pkl
        data/models/d1_full_corpus_index_microsoft_vscode_bge/
Writes: reports/track2_vscode_resolution_filter_check.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.resolution import (  # noqa: E402
    BUCKET_LABELS,
    ResolutionTimePredictor,
    engineer_features,
    hours_to_bucket,
)

SEED = 42
N_BOOTSTRAP = 2000
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPORTS = Path("reports")
REPO = "microsoft_vscode"

BOT_AUTHORS = {"vs-code-engineering[bot]"}
DUP_PHRASES = [
    "not working",
    "not responding",
    "doesn't work",
    "does not work",
    "won't work",
    "wont work",
    "issue",
    "problem",
    "crash",
    "freeze",
    "blur",
    "can't used",
    "cant used",
]


def is_duplicate_wave(title: str, resolution_hours: float) -> bool:
    if resolution_hours >= 24:
        return False
    t = str(title).lower()
    if "terminal" not in t:
        return False
    return any(p in t for p in DUP_PHRASES)


def bootstrap_ci(values: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_means[i] = sample.mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"mean": float(values.mean()), "ci95_lower": float(lo), "ci95_upper": float(hi)}


def bucket_accuracy_vs_naive(
    predictor: ResolutionTimePredictor, X_test: pd.DataFrame, true_bucket_idx: np.ndarray
) -> dict:
    assert predictor.model_bucket is not None
    raw_proba = np.asarray(predictor.model_bucket.predict(X_test))
    pred_bucket_idx = raw_proba.argmax(axis=1)
    correct = (pred_bucket_idx == true_bucket_idx).astype(float)

    majority_bucket = max(
        predictor.bucket_train_distribution, key=predictor.bucket_train_distribution.get
    )
    majority_idx = BUCKET_LABELS.index(majority_bucket)
    naive_correct = (true_bucket_idx == majority_idx).astype(float)
    acc_delta = correct - naive_correct

    return {
        "n": len(true_bucket_idx),
        "trained_accuracy": round(float(correct.mean()), 4),
        "naive_accuracy": round(float(naive_correct.mean()), 4),
        "accuracy_delta_bootstrap": bootstrap_ci(acc_delta),
        "naive_majority_bucket": majority_bucket,
    }


def load_embeddings_from_d1_index(df: pd.DataFrame) -> np.ndarray | None:
    index_dir = MODELS_DIR / f"d1_full_corpus_index_{REPO}_bge"
    if not (index_dir / "index.faiss").exists():
        return None
    import faiss
    import joblib

    meta = joblib.load(str(index_dir / "meta.pkl"))
    index = faiss.read_index(str(index_dir / "index.faiss"))
    num_to_idx = {int(n): i for i, n in enumerate(meta["issue_numbers"])}
    dim = index.d
    embs = np.zeros((len(df), dim), dtype=np.float32)
    for row_pos, num in enumerate(df["number"]):
        idx = num_to_idx.get(int(num))
        if idx is not None:
            embs[row_pos] = index.reconstruct(int(idx))
    return embs


def main() -> None:
    train = pd.read_parquet(PROCESSED_DIR / f"{REPO}_temporal_train_lever3.parquet")
    test = pd.read_parquet(PROCESSED_DIR / f"{REPO}_temporal_test_lever3.parquet")
    train = train[train["resolution_hours"] > 0]
    test = test[test["resolution_hours"] > 0].reset_index(drop=True)

    test["is_bot"] = test["author"].isin(BOT_AUTHORS)
    test["is_dup_wave"] = test.apply(
        lambda r: is_duplicate_wave(r["title"], r["resolution_hours"]), axis=1
    )
    test["bucket_idx"] = hours_to_bucket(test["resolution_hours"].values)
    test["bucket_label"] = [BUCKET_LABELS[i] for i in test["bucket_idx"]]

    n_total = len(test)
    n_bot = int(test["is_bot"].sum())
    n_dup = int(test["is_dup_wave"].sum())
    n_either = int((test["is_bot"] | test["is_dup_wave"]).sum())

    dist_unfiltered = test["bucket_label"].value_counts(normalize=True).round(4).to_dict()
    filtered = test[~(test["is_bot"] | test["is_dup_wave"])].reset_index(drop=True)
    dist_filtered = filtered["bucket_label"].value_counts(normalize=True).round(4).to_dict()

    print(f"Test set: {n_total} rows")
    print(f"  bot-authored: {n_bot} ({n_bot/n_total:.1%})")
    print(f"  duplicate-wave (rule-based): {n_dup} ({n_dup/n_total:.1%})")
    print(f"  either (union): {n_either} ({n_either/n_total:.1%})")
    print(f"  remaining after filter: {len(filtered)}")
    print(f"Unfiltered bucket distribution: {dist_unfiltered}")
    print(f"Filtered bucket distribution:   {dist_filtered}")

    dup_wave_sample = test[test["is_dup_wave"]][["number", "title", "author", "resolution_hours"]]
    dup_wave_sample_records = dup_wave_sample.head(30).to_dict(orient="records")

    predictor = ResolutionTimePredictor.load(
        str(MODELS_DIR / f"resolution_predictor_{REPO}_lever3.pkl")
    )
    emb_train = load_embeddings_from_d1_index(train)
    emb_test_full = load_embeddings_from_d1_index(test)
    X_train, pca = engineer_features(train, train_df=train, embeddings=emb_train, pca=None)
    X_test_full, _ = engineer_features(test, train_df=train, embeddings=emb_test_full, pca=pca)

    bucket_idx_full = hours_to_bucket(test["resolution_hours"].values)
    eval_unfiltered = bucket_accuracy_vs_naive(predictor, X_test_full, bucket_idx_full)

    keep_mask = ~(test["is_bot"] | test["is_dup_wave"]).values
    X_test_filtered = X_test_full[keep_mask].reset_index(drop=True)
    bucket_idx_filtered = bucket_idx_full[keep_mask]
    eval_filtered = bucket_accuracy_vs_naive(predictor, X_test_filtered, bucket_idx_filtered)

    print("\n=== Bucket accuracy vs naive, UNFILTERED ===")
    print(json.dumps(eval_unfiltered, indent=2))
    print("\n=== Bucket accuracy vs naive, FILTERED (bot + dup-wave removed) ===")
    print(json.dumps(eval_filtered, indent=2))

    out = {
        "n_total": n_total,
        "n_bot": n_bot,
        "n_dup_wave": n_dup,
        "n_either_filtered_out": n_either,
        "n_remaining": len(filtered),
        "bucket_distribution_unfiltered": dist_unfiltered,
        "bucket_distribution_filtered": dist_filtered,
        "dup_wave_sample": dup_wave_sample_records,
        "eval_unfiltered": eval_unfiltered,
        "eval_filtered": eval_filtered,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "track2_vscode_resolution_filter_check.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8"
    )
    print("\nWrote reports/track2_vscode_resolution_filter_check.json")


if __name__ == "__main__":
    main()
