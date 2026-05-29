"""W3 T1 — Data audit for gold_duplicates.parquet.

Reports schema, per-repo counts, column inspection, chain detection,
and connected-component analysis for train/val/test splitting.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
import numpy as np

ROOT = Path(__file__).parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

DATA_DIR = ROOT / "data"


def main() -> None:
    df = pd.read_parquet(DATA_DIR / "gold_duplicates.parquet")

    print("=== SCHEMA ===")
    print(df.dtypes.to_string())
    print(f"\nTotal rows: {len(df)}")
    print(f"\nColumns: {list(df.columns)}")

    print("\n=== PER-REPO COUNTS ===")
    print(df["repo"].value_counts().to_string())

    print("\n=== SAMPLE ROWS (first 3) ===")
    pd.set_option("display.max_colwidth", 80)
    print(df.head(3).to_string())

    print("\n=== NULL COUNTS ===")
    print(df.isnull().sum().to_string())

    # Check for query_number / original_number columns
    if "query_number" in df.columns and "original_number" in df.columns:
        print("\n=== PAIR STRUCTURE (query_number → original_number) ===")
        print(f"Unique query issues: {df['query_number'].nunique()}")
        print(f"Unique original issues: {df['original_number'].nunique()}")
        print(f"Unique repos: {df['repo'].nunique()}")

        # Multi-positive detection (one query → multiple originals)
        query_counts = df.groupby(["repo", "query_number"]).size()
        multi_pos = query_counts[query_counts > 1]
        print(f"\nQueries with >1 positive (multi-positive): {len(multi_pos)}")
        if len(multi_pos) > 0:
            print(multi_pos.head(5))

        # Chain detection: A↔B and B↔C patterns
        print("\n=== DUPLICATE CHAIN DETECTION ===")
        all_numbers = set(df["query_number"].unique()) | set(df["original_number"].unique())
        print(f"Total unique issue numbers involved: {len(all_numbers)}")

        # Build adjacency for connected components
        from collections import defaultdict
        adj: dict = defaultdict(set)
        for _, row in df.iterrows():
            q, o = int(row["query_number"]), int(row["original_number"])
            adj[q].add(o)
            adj[o].add(q)

        # BFS to find connected components
        visited: set = set()
        components: list = []
        for node in sorted(all_numbers):
            if node not in visited:
                comp: list = []
                queue = [node]
                while queue:
                    n = queue.pop(0)
                    if n in visited:
                        continue
                    visited.add(n)
                    comp.append(n)
                    queue.extend(adj[n] - visited)
                components.append(comp)

        sizes = [len(c) for c in components]
        print(f"Connected components: {len(components)}")
        print(f"Component size distribution:")
        for s in sorted(set(sizes)):
            count = sizes.count(s)
            print(f"  size={s}: {count} components ({count * s} issues)")

        # Check if chains exist (components > 2)
        large_comps = [c for c in components if len(c) > 2]
        print(f"\nComponents with >2 issues (chains): {len(large_comps)}")
        if large_comps:
            print(f"  Largest chain size: {max(len(c) for c in large_comps)}")
            print(f"  Sample chain (first): {large_comps[0][:5]}")

        # Per-repo component analysis
        print("\n=== PER-REPO COMPONENT ANALYSIS ===")
        for repo in df["repo"].unique():
            repo_df = df[df["repo"] == repo]
            repo_adj: dict = defaultdict(set)
            for _, row in repo_df.iterrows():
                q, o = int(row["query_number"]), int(row["original_number"])
                repo_adj[q].add(o)
                repo_adj[o].add(q)
            repo_numbers = set(repo_df["query_number"].unique()) | set(repo_df["original_number"].unique())
            visited2: set = set()
            repo_comps: list = []
            for node in sorted(repo_numbers):
                if node not in visited2:
                    comp = []
                    queue = [node]
                    while queue:
                        n = queue.pop(0)
                        if n in visited2:
                            continue
                        visited2.add(n)
                        comp.append(n)
                        queue.extend(repo_adj[n] - visited2)
                    repo_comps.append(comp)
            comp_sizes = [len(c) for c in repo_comps]
            print(f"\n  {repo}:")
            print(f"    Pairs: {len(repo_df)}")
            print(f"    Unique issues: {len(repo_numbers)}")
            print(f"    Components: {len(repo_comps)}")
            max_size = max(comp_sizes) if comp_sizes else 0
            print(f"    Max component size: {max_size}")
            print(f"    Components with chains (>2): {sum(1 for s in comp_sizes if s > 2)}")

    # Check for text columns
    text_cols = [c for c in df.columns if "title" in c.lower() or "body" in c.lower() or "text" in c.lower()]
    if text_cols:
        print(f"\n=== TEXT COLUMN LENGTHS ===")
        for col in text_cols:
            lengths = df[col].fillna("").str.len()
            print(f"  {col}: mean={lengths.mean():.0f}, p50={lengths.median():.0f}, p95={lengths.quantile(0.95):.0f}, max={lengths.max()}")

    print("\n=== DONE ===")


if __name__ == "__main__":
    main()
