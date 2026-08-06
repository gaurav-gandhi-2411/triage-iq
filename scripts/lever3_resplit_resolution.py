"""LEVER 3, step 1: regenerate the resolution model's temporal splits from the CURRENT corpus.

Finding: `{repo}_temporal_{train,val,test}.parquet` were last generated 2026-05-30, but
`data/processed/issues_{repo}.parquet` was regenerated 2026-07-11 (Phase 2b corpus growth) and
grew substantially since. Both repos' resolution splits are stale relative to their own source
data:

  - vscode: train pinned to Oct 2015-Apr 2016 only; corpus now has ~650 closed issues/year
    available through 2025 plus a partial 2026 batch (12,242 closed total vs ~6,150 used).
  - k8s: train/val/test together span only Jun 2014-Oct 2015 (16 months), 14,968 rows used out
    of 29,911 closed issues now available in the corpus -- less than half.

This is the exact "distribution shift" root cause diagnosed in ADR-0009 T1.5 (train = old era,
test = a narrow recent window) -- except it's not fixed by the created_at split alone if the
split itself is running against stale, narrow source data.

Deliberately calls time_based_split() directly, NOT scripts/03_split.py's main() -- that script
also regenerates the component classifier's stratified splits in the same call, which are OUT OF
SCOPE here (the classifier is a separate, already-shipped, ADR-0036-verified system; touching its
training data is not part of this fix and would be uncontrolled scope creep).

Writes to a `_lever3` suffix, NOT the canonical `{repo}_temporal_*.parquet` path scripts/
09_train_resolution.py reads -- candidate first, verify, then an explicit, separate step promotes
it, same discipline as the retrieval index rebuild (ADR-0040).

Reads:  data/processed/issues_{repo}.parquet
Writes: data/processed/{repo}_temporal_{train,val,test}_lever3.parquet
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.data.splits import time_based_split  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]


def main() -> None:
    for repo in REPOS:
        df = pd.read_parquet(PROCESSED_DIR / f"issues_{repo}.parquet")
        closed = df[df["resolution_hours"].notna() & (df["resolution_hours"] > 0)].copy()

        old_train = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_train.parquet")
        old_val = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_val.parquet")
        old_test = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_test.parquet")
        old_total = len(old_train) + len(old_val) + len(old_test)

        train, val, test = time_based_split(closed, timestamp_col="created_at")

        log.info(
            "[%s] OLD (05-30, stale): n=%d, train ends %s, test spans [%s, %s]",
            repo,
            old_total,
            old_train["created_at"].max(),
            old_test["created_at"].min(),
            old_test["created_at"].max(),
        )
        log.info(
            "[%s] NEW (lever3, current corpus): n=%d (%.0f%% more data), train ends %s, "
            "test spans [%s, %s]",
            repo,
            len(closed),
            100 * (len(closed) - old_total) / old_total,
            train["created_at"].max(),
            test["created_at"].min(),
            test["created_at"].max(),
        )

        for split_name, split_df in [("train", train), ("val", val), ("test", test)]:
            out_path = PROCESSED_DIR / f"{repo}_temporal_{split_name}_lever3.parquet"
            split_df.to_parquet(out_path, index=False)
            log.info("[%s] wrote %d rows to %s", repo, len(split_df), out_path)


if __name__ == "__main__":
    main()
