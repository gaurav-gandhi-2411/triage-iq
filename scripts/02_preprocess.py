"""Run preprocessing for all scraped repos and save parquet files.

Usage:
    python scripts/02_preprocess.py
    python scripts/02_preprocess.py --repos microsoft_vscode kubernetes_kubernetes
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage_iq.data.preprocess import (
    build_processed_df,
    load_raw_issues,
    save_processed,
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
    parser = argparse.ArgumentParser(description="Preprocess raw GitHub Issues to parquet.")
    parser.add_argument("--repos", nargs="+", default=None, help="Repo dirs to process")
    parser.add_argument("--cache-dir", default="data/raw")
    parser.add_argument("--out-dir", default="data/processed")
    args = parser.parse_args()

    repos = args.repos or DEFAULT_REPOS

    for repo in repos:
        raw_dir = Path(args.cache_dir) / repo
        if not raw_dir.exists():
            logging.warning("Skipping %s — raw directory not found", repo)
            continue
        try:
            df = load_raw_issues(repo, cache_dir=args.cache_dir)
            df = build_processed_df(df, repo)
            out_path = save_processed(df, repo, out_dir=args.out_dir)
            logging.info(
                "%s: %d total issues, %d with resolution_hours",
                repo,
                len(df),
                df["resolution_hours"].notna().sum(),
            )
        except Exception as exc:
            logging.error("Failed to process %s: %s", repo, exc)


if __name__ == "__main__":
    main()
