"""Train/validation/test split utilities for TriageIQ datasets.

Two split strategies:
- time_based_split: split by closed_at timestamp to prevent leakage where
  test issues were opened before training cutoff.
- stratified_classifier_split: preserve label distribution across splits,
  used for the component classifier where class balance matters.
"""

import logging

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def time_based_split(
    df: pd.DataFrame,
    train_pct: float = 0.8,
    val_pct: float = 0.1,
    test_pct: float = 0.1,
    timestamp_col: str = "closed_at",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by timestamp to avoid leakage.

    Issues are sorted by `timestamp_col` ascending. The earliest 80% become
    training, the next 10% validation, the last 10% test. Open issues (null
    closed_at) are excluded since they have no resolution time target.

    Args:
        df: Input DataFrame. Must contain `timestamp_col`.
        train_pct: Fraction for training set.
        val_pct: Fraction for validation set.
        test_pct: Fraction for test set.
        timestamp_col: Column to sort by (default: "closed_at").

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-9, "Percentages must sum to 1.0"

    closed = df[df[timestamp_col].notna()].copy()
    closed = closed.sort_values(timestamp_col).reset_index(drop=True)
    n = len(closed)

    train_end = int(n * train_pct)
    val_end = train_end + int(n * val_pct)

    train_df = closed.iloc[:train_end].copy()
    val_df = closed.iloc[train_end:val_end].copy()
    test_df = closed.iloc[val_end:].copy()

    logger.info(
        "Time-based split on %d closed issues: train=%d  val=%d  test=%d",
        n, len(train_df), len(val_df), len(test_df),
    )
    logger.info(
        "Cutoffs — train ends: %s  val ends: %s",
        train_df[timestamp_col].max(),
        val_df[timestamp_col].max(),
    )
    return train_df, val_df, test_df


def stratified_classifier_split(
    df: pd.DataFrame,
    label_col: str,
    train_pct: float = 0.8,
    val_pct: float = 0.1,
    test_pct: float = 0.1,
    min_class_samples: int = 10,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split preserving label distribution for the component classifier.

    Rows where `label_col` is null are excluded. Classes with fewer than
    `min_class_samples` examples are dropped to ensure stratification works.

    Args:
        df: Input DataFrame.
        label_col: Column containing the classification target.
        train_pct: Fraction for training set.
        val_pct: Fraction for validation set.
        test_pct: Fraction for test set.
        min_class_samples: Minimum instances per class; smaller classes dropped.
        random_state: Reproducibility seed.

    Returns:
        (train_df, val_df, test_df)
    """
    assert abs(train_pct + val_pct + test_pct - 1.0) < 1e-9, "Percentages must sum to 1.0"

    labeled = df[df[label_col].notna()].copy()

    # Drop classes too small to split reliably
    counts = labeled[label_col].value_counts()
    valid_classes = counts[counts >= min_class_samples].index
    dropped = counts[counts < min_class_samples]
    if not dropped.empty:
        logger.warning(
            "Dropping %d classes with < %d samples: %s",
            len(dropped), min_class_samples, list(dropped.index),
        )
    labeled = labeled[labeled[label_col].isin(valid_classes)].copy()

    # First split: (train+val) vs test
    test_size = test_pct
    train_val, test_df = train_test_split(
        labeled,
        test_size=test_size,
        stratify=labeled[label_col],
        random_state=random_state,
    )

    # Second split: train vs val
    val_size = val_pct / (train_pct + val_pct)
    train_df, val_df = train_test_split(
        train_val,
        test_size=val_size,
        stratify=train_val[label_col],
        random_state=random_state,
    )

    logger.info(
        "Stratified split on %d labeled issues (col=%s): train=%d  val=%d  test=%d",
        len(labeled), label_col, len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def save_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    prefix: str,
    out_dir: str = "data/processed",
) -> None:
    """Save split DataFrames as parquet files.

    Files written: {out_dir}/{prefix}_train.parquet, _val.parquet, _test.parquet
    """
    import pathlib
    base = pathlib.Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
        path = base / f"{prefix}_{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        logger.info("Saved %d rows to %s", len(split_df), path)
