"""Regression test: W3 eval/train disjointness.

Guards against reintroduction of the ADR-0010 contamination bug where sample_gold()
drew evaluation pairs from the full gold corpus instead of the held-out test split,
resulting in 66-71% of eval pairs being training data.
"""
from __future__ import annotations

from pathlib import Path

import pytest


SPLIT_PATH = Path("data/w3_split.parquet")


@pytest.mark.skipif(not SPLIT_PATH.exists(), reason="W3 split artifact not present")
def test_w3_test_split_disjoint_from_train() -> None:
    """Test-split pairs must have zero overlap with training pairs.

    If this fails, the split artifact has been corrupted or the test-pair selection
    logic has been changed to include training data — both require immediate investigation.
    """
    import pandas as pd

    split_df = pd.read_parquet(str(SPLIT_PATH))

    def pair_keys(df: pd.DataFrame) -> frozenset:
        return frozenset(
            zip(df["repo"], df["query_number"].astype(int), df["original_number"].astype(int))
        )

    train_keys = pair_keys(split_df[split_df["split"] == "train"])
    test_keys = pair_keys(split_df[split_df["split"] == "test"])

    overlap = train_keys & test_keys
    assert len(overlap) == 0, (
        f"EVAL/TRAIN LEAK: {len(overlap)} test pairs found in training set. "
        f"Sample: {sorted(overlap)[:5]}. "
        "See ADR-0010 correction note."
    )
