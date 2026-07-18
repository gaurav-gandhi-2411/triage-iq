"""D1 task 2: draw the additional hand-judging sample needed to CONSTRUCT the held-out eval set.

The eval set is the trustworthy scoreboard (spec.md hard rule: >=90% verified precision). The
clean-pool channels measured in checkpoint 1 sit at 83-86% -- a plain random draw from them
would NOT clear 90%. The only rigorous way to guarantee >=90% is to build the eval set entirely
from INDIVIDUALLY hand-verified genuine pairs (same principle ADR-0032 used for its "clean
subset" recall computation) -- by construction, an eval set containing only verified-genuine
pairs is ~100% precision, limited only by judge error, not sampling noise.

This script draws NEW samples (excluding every pair already hand-judged in checkpoint 1, so no
duplicate review work) sized to net enough genuine pairs after judging:

  k8s (related task):
    - k8s_forward_scrape/body_related_ext: NEW sample of 90 (of 424 remaining, 30 already judged)
    - legacy_gold_v1/body_related: full census of the 34 remaining (3 already judged of 37)
    (k8s_forward_scrape/body_related is already fully census-judged, 45/45 -- nothing left there)
    Existing bank: 66 genuine (38+25+3). Target: ~150 genuine for eval -> need ~84 more; this
    draw (90+34=124 raw) should net ~100+ more at the ~83-84% observed hit rate, comfortable
    margin over the 84-needed floor.

  vscode (duplicate task):
    - vscode_dup_scrape/dup_comment: NEW sample of 220 (of 2202 remaining, 40 already judged)
    Existing bank: 34 genuine. Target: ~200 genuine for eval -> need ~166 more; 220 raw at the
    observed 85% hit rate nets ~187, comfortable margin.

  vscode (related task): no new draw -- the entire 22-pair population is already fully
  census-judged (19 genuine); GG's decision was to use it as-is, eval-only, directional.

Reads:  data/gold_related_v2.parquet
        reports/d1_pair_quality_review.json      (D1 checkpoint-1 judged pairs, to exclude)
        reports/phaseC_pair_quality_review.json  (ADR-0032 judged pairs, to exclude)
Writes: reports/d1_eval_carve_sample_for_review.json
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SEED = 43  # distinct from the checkpoint-1 draw (SEED=42) -- avoids any accidental correlation
GOLD_PATH = Path("data/gold_related_v2.parquet")
D1_REVIEW = Path("reports/d1_pair_quality_review.json")
ADR0032_REVIEW = Path("reports/phaseC_pair_quality_review.json")
OUT = Path("reports/d1_eval_carve_sample_for_review.json")

MAX_BODY_FOR_REVIEW = 1500

TARGETS = [
    ("kubernetes_kubernetes", "k8s_forward_scrape", "body_related_ext", "product", 90),
    (
        "kubernetes_kubernetes",
        "legacy_gold_v1",
        "body_related",
        "product",
        None,
    ),  # full census remainder
    ("microsoft_vscode", "vscode_dup_scrape", "dup_comment", "gate", 220),
]


def already_judged_keys() -> set[tuple[str, int, int]]:
    keys = set()
    for path in (D1_REVIEW, ADR0032_REVIEW):
        for r in json.loads(path.read_text(encoding="utf-8")):
            keys.add((r["repo"], int(r["query_number"]), int(r["original_number"])))
    return keys


def build_sample() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    exclude = already_judged_keys()
    log.info("excluding %d already-judged pairs", len(exclude))

    rows = []
    for repo, channel, source, stratum, n in TARGETS:
        subset = gold[
            (gold["repo"] == repo)
            & (gold["channel"] == channel)
            & (gold["source"] == source)
            & (gold["stratum"] == stratum)
        ]
        not_judged = subset[
            ~subset.apply(
                lambda r: (r["repo"], int(r["query_number"]), int(r["original_number"])) in exclude,
                axis=1,
            )
        ]
        if n is None or len(not_judged) <= n:
            sample = not_judged
            note = "full_census_remainder"
        else:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(not_judged), size=n, replace=False)
            sample = not_judged.iloc[sorted(idx)]
            note = f"sample_{n}_of_{len(not_judged)}_remaining"
        log.info(
            "[%s/%s/%s/%s] %d available (excl. judged), %s -> %d drawn",
            repo,
            channel,
            source,
            stratum,
            len(not_judged),
            note,
            len(sample),
        )
        for _, row in sample.iterrows():
            rows.append(
                {
                    "repo": repo,
                    "query_number": int(row["query_number"]),
                    "original_number": int(row["original_number"]),
                    "query_title": row["query_title"],
                    "query_body": str(row["query_body"])[:MAX_BODY_FOR_REVIEW],
                    "original_title": row["original_title"],
                    "original_body": str(row["original_body"])[:MAX_BODY_FOR_REVIEW],
                    "channel": row["channel"],
                    "source": row["source"],
                    "stratum": row["stratum"],
                    "sample_note": note,
                    "purpose": "eval_carve",
                    "verdict": None,
                    "verdict_reason": None,
                }
            )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log.info("Wrote %d-pair sample to %s", len(rows), OUT)


if __name__ == "__main__":
    build_sample()
