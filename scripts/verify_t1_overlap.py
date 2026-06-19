"""T1 verification: check eval sample overlap with training pairs."""
from __future__ import annotations

import random
import pandas as pd


def sample_gold(gold: pd.DataFrame, repo: str, n: int, seed: int) -> pd.DataFrame:
    """Exact same sampling as T5 sample_gold / W1.3 fast benchmark."""
    repo_gold = gold[gold["repo"] == repo].copy()
    rng = random.Random(seed)
    idxs = rng.sample(range(len(repo_gold)), min(n, len(repo_gold)))
    return repo_gold.iloc[idxs]


def main() -> None:
    gold = pd.read_parquet("data/gold_related.parquet")
    split_df = pd.read_parquet("data/w3_split.parquet")

    print("=== DATA SHAPES ===")
    print(f"gold: {len(gold)} rows, cols: {list(gold.columns)}")
    print(f"split_df: {len(split_df)} rows, cols: {list(split_df.columns)}")
    print()

    train_pairs = split_df[split_df["split"] == "train"]
    val_pairs = split_df[split_df["split"] == "val"]
    test_pairs = split_df[split_df["split"] == "test"]

    print("=== SPLIT COUNTS ===")
    for repo in ["kubernetes_kubernetes", "microsoft_vscode"]:
        t = len(train_pairs[train_pairs["repo"] == repo])
        v = len(val_pairs[val_pairs["repo"] == repo])
        te = len(test_pairs[test_pairs["repo"] == repo])
        g = len(gold[gold["repo"] == repo])
        print(f"  {repo}: gold={g}, train={t}, val={v}, test={te}")
    print()

    # Build keyed sets for each split
    def pair_keys(df: pd.DataFrame) -> set:
        return set(
            zip(df["repo"], df["query_number"].astype(int), df["original_number"].astype(int))
        )

    train_keys = pair_keys(train_pairs)
    val_keys = pair_keys(val_pairs)
    test_keys = pair_keys(test_pairs)
    all_split_keys = train_keys | val_keys | test_keys

    print(f"Total training pair keys: {len(train_keys)}")
    print(f"Total val pair keys: {len(val_keys)}")
    print(f"Total test pair keys: {len(test_keys)}")
    print()

    # Check n=100 seed=42 sample
    for repo in ["kubernetes_kubernetes", "microsoft_vscode"]:
        sample = sample_gold(gold, repo, 100, 42)
        sample_keys = pair_keys(sample)

        overlap_train = sample_keys & train_keys
        overlap_val = sample_keys & val_keys
        overlap_test = sample_keys & test_keys
        not_in_any_split = sample_keys - all_split_keys

        print(f"=== {repo} n=100 seed=42 ===")
        print(f"  Sample size: {len(sample)}")
        print(f"  In train split:          {len(overlap_train)} / 100")
        print(f"  In val split:            {len(overlap_val)} / 100")
        print(f"  In test split:           {len(overlap_test)} / 100")
        print(f"  NOT in any split:        {len(not_in_any_split)} / 100")
        print()

        if overlap_train:
            print(f"  *** TRAIN CONTAMINATION: {len(overlap_train)} pairs ***")
            for k in sorted(overlap_train)[:20]:
                print(f"    repo={k[0]}, query={k[1]}, positive={k[2]}")
        else:
            print("  CLEAN: zero overlap with training pairs")
        print()

    # Also verify test-split leakage (T3c)
    print("=== T3c: TEST SPLIT LEAKAGE CHECK ===")
    print("Confirm test pairs have zero overlap with train pairs")
    test_train_overlap = test_keys & train_keys
    print(f"  test & train: {len(test_train_overlap)} pairs")
    if test_train_overlap:
        print("  *** LEAKAGE DETECTED ***")
        for k in sorted(test_train_overlap)[:20]:
            print(f"    {k}")
    else:
        print("  CLEAN: test and train are disjoint")

    test_val_overlap = test_keys & val_keys
    print(f"  test & val: {len(test_val_overlap)} pairs")
    if test_val_overlap:
        print("  *** VAL/TEST OVERLAP ***")
    else:
        print("  CLEAN: test and val are disjoint")
    print()

    # Print the full n=100 sample for transparency
    print("=== FULL n=100 SAMPLE LISTING (query_number, original_number) ===")
    for repo in ["kubernetes_kubernetes", "microsoft_vscode"]:
        sample = sample_gold(gold, repo, 100, 42)
        sample_keys = pair_keys(sample)
        print(f"\n--- {repo} ---")
        for k in sorted(sample_keys, key=lambda x: x[1]):
            in_split = "train" if k in train_keys else ("val" if k in val_keys else ("test" if k in test_keys else "NONE"))
            print(f"  query={k[1]:6d}, positive={k[2]:6d}  split={in_split}")


if __name__ == "__main__":
    main()
