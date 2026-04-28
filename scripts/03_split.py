"""Generate train/val/test splits for all processed repos.

Usage:
    python scripts/03_split.py
    python scripts/03_split.py --repos microsoft_vscode kubernetes_kubernetes
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from triage_iq.data.splits import (
    save_splits,
    stratified_classifier_split,
    time_based_split,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

DEFAULT_REPOS = [
    "microsoft_vscode",
    "kubernetes_kubernetes",
    "tensorflow_tensorflow",
    "pytorch_pytorch",
    "apache_airflow",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate train/val/test splits.")
    parser.add_argument("--repos", nargs="+", default=None)
    parser.add_argument("--processed-dir", default="data/processed")
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS

    for repo in repos:
        parquet = Path(args.processed_dir) / f"issues_{repo}.parquet"
        if not parquet.exists():
            logging.warning("Skipping %s — parquet not found", repo)
            continue

        df = pd.read_parquet(parquet)
        logging.info("Loaded %s: %d rows", repo, len(df))

        # ── Time-based split for resolution time predictor ──
        train, val, test = time_based_split(df)
        save_splits(train, val, test, prefix=f"{repo}_temporal", out_dir=args.processed_dir)

        # ── Stratified split for component classifier ──
        if df["component"].notna().sum() >= 30:
            train_c, val_c, test_c = stratified_classifier_split(df, label_col="component")
            save_splits(
                train_c, val_c, test_c,
                prefix=f"{repo}_classifier",
                out_dir=args.processed_dir,
            )
        else:
            logging.warning(
                "%s: fewer than 30 labeled rows for component classifier split — skipping", repo
            )

        logging.info(
            "%s splits done. Temporal: %d/%d/%d",
            repo, len(train), len(val), len(test),
        )


if __name__ == "__main__":
    main()
