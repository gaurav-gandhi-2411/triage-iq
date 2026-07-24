"""Phase B (classifier improvement): hard pre-flight leakage guard.

Asserts issue-level disjointness across the classifier's train/val/test splits for both repos --
non-negotiable, fails hard on violation. Same discipline as scripts/d2_assert_leakage_guard.py.

Usage:
  python scripts/classifier_assert_leakage_guard.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]


def assert_repo_disjoint(repo: str) -> None:
    splits = {}
    for name in ("train", "val", "test"):
        path = PROCESSED_DIR / f"{repo}_classifier_{name}.parquet"
        if not path.exists():
            raise SystemExit(f"[{repo}] missing split file: {path}")
        splits[name] = set(pd.read_parquet(path)["number"].astype(int))

    train_val = splits["train"] & splits["val"]
    train_test = splits["train"] & splits["test"]
    val_test = splits["val"] & splits["test"]

    if train_val or train_test or val_test:
        raise SystemExit(
            f"[{repo}] LEAKAGE DETECTED -- train/val overlap={len(train_val)}, "
            f"train/test overlap={len(train_test)}, val/test overlap={len(val_test)}. "
            f"Contaminated split, refusing to train."
        )

    n_train, n_val, n_test = len(splits["train"]), len(splits["val"]), len(splits["test"])
    n_union = len(splits["train"] | splits["val"] | splits["test"])
    print(
        f"[{repo}] DISJOINT -- train={n_train} val={n_val} test={n_test} "
        f"issues, union={n_union} (== sum: {n_union == n_train + n_val + n_test})"
    )


def main() -> None:
    for repo in REPOS:
        assert_repo_disjoint(repo)
    print("\nLeakage guard PASSED for all repos.")


if __name__ == "__main__":
    main()
