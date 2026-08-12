"""Mining precision: strict (retrieval-validity) audit of channels never measured under the
VALID/EXCLUDE_UMBRELLA/EXCLUDE_CAUSAL_ONLY/EXCLUDE_OTHER rubric.

Context: the 2026-08-11 k8s clean-eval investigation pre-registered a stricter validity rubric
than D1's original genuine/incidental audit (scripts/d1_channel_precision_audit.py) -- one that
specifically asks "is this pair a fair test of single-vector content-similarity retrieval,"
not just "is there a real relationship." Applying it to the 150-pair k8s eval set (which draws
from k8s_forward_scrape/body_related(_ext) and legacy_gold_v1/body_related) found only 42-45%
VALID, vs. 83-84% "genuine" under D1's looser rubric for the same channels
(reports/d1_channel_precision_audit.json). That gap is the actual finding: roughly half of
even genuinely-related pairs are structurally unfit for training a single-embedding retriever
(umbrella/tracking issues, causal-only citations) -- a shape D1's rubric never screened for.

This script draws blind (no channel/source/stratum visible), fixed-seed samples from the
channels that have NEVER been measured under the strict rubric, for hand/agent labeling:
  1. vscode_dup_scrape / dup_comment / gate (2242 pairs) -- vscode's largest channel by far,
     currently siloed into the "gate" (duplicate-detection) stratum only. Structurally immune
     to the umbrella-issue failure mode (a duplicate-closure names exactly one target), so a
     high strict-rubric score here would make it the standout product-task training candidate.
  2. k8s_extended_mine / body_related_ext / product (200 pairs) -- only a partial n=20 loose
     read exists (ADR-0032). This is the OTHER half (with k8s_forward_scrape) of k8s's product
     stratum body-text-regex mining.
  3. vscode_body_refs / body_related_ext / product (206 pairs) -- only body_related (n=11,
     the stronger-pattern sibling) has ever been read; body_related_ext (weaker patterns,
     "related to #N" style) never has.

Reads:  data/gold_related_v2.parquet
Writes: reports/mining_precision_strict_sample.json  -- blind pairs for labeling
         (batched: field "batch" 1..N, ~30/batch, matching the D1/2026-08-11 precedent)

Reproduce: python scripts/mining_precision_strict_audit.py --sample
After labeling (fill "label": "VALID"|"EXCLUDE_UMBRELLA"|"EXCLUDE_CAUSAL_ONLY"|"EXCLUDE_OTHER"
+ "reason") into reports/mining_precision_strict_labeled.json:
           python scripts/mining_precision_strict_audit.py --analyze
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SEED = 42
GOLD_PATH = Path("data/gold_related_v2.parquet")
SAMPLE_PATH = Path("reports/mining_precision_strict_sample.json")
LABELED_PATH = Path("reports/mining_precision_strict_labeled.json")
OUTPUT_PATH = Path("reports/mining_precision_strict_audit.json")

MAX_BODY = 1800
BATCH_SIZE = 30

# (repo, channel, source, stratum, n)
TARGETS = [
    ("microsoft_vscode", "vscode_dup_scrape", "dup_comment", "gate", 60),
    ("kubernetes_kubernetes", "k8s_extended_mine", "body_related_ext", "product", 50),
    ("microsoft_vscode", "vscode_body_refs", "body_related_ext", "product", 50),
]


def build_sample() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    rows: list[dict] = []
    pid = 0
    for repo, channel, source, stratum, n in TARGETS:
        subset = gold[
            (gold["repo"] == repo)
            & (gold["channel"] == channel)
            & (gold["source"] == source)
            & (gold["stratum"] == stratum)
        ]
        rng = np.random.default_rng(SEED + pid)  # vary seed per target, still reproducible
        if len(subset) <= n:
            sample = subset
            note = "full_census"
        else:
            idx = rng.choice(len(subset), size=n, replace=False)
            sample = subset.iloc[sorted(idx)]
            note = f"sample_{n}_of_{len(subset)}"
        log.info("[%s/%s/%s/%s] %s -> %d rows", repo, channel, source, stratum, note, len(sample))
        for _, row in sample.iterrows():
            pid += 1
            rows.append(
                {
                    "pair_id": f"mp{pid:04d}",
                    "_repo": repo,
                    "_channel": channel,
                    "_source": source,
                    "_stratum": stratum,
                    "_sample_note": note,
                    "_query_number": int(row["query_number"]),
                    "_target_number": int(row["original_number"]),
                    "query_title": row["query_title"],
                    "query_body": str(row["query_body"])[:MAX_BODY],
                    "target_title": row["original_title"],
                    "target_body": str(row["original_body"])[:MAX_BODY],
                    "label": None,
                    "reason": None,
                }
            )

    # Batch assignment for parallel independent labeling, blind fields stripped per-batch on write
    for i, row in enumerate(rows):
        row["batch"] = i // BATCH_SIZE + 1

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    n_batches = rows[-1]["batch"] if rows else 0
    log.info("Wrote %d pairs across %d batches to %s", len(rows), n_batches, SAMPLE_PATH)


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (center - adj) / denom, (center + adj) / denom


def analyze() -> None:
    labeled = json.loads(LABELED_PATH.read_text(encoding="utf-8"))
    df = pd.DataFrame(labeled)
    valid_labels = {"VALID", "EXCLUDE_UMBRELLA", "EXCLUDE_CAUSAL_ONLY", "EXCLUDE_OTHER"}
    assert df["label"].isin(valid_labels).all(), "unjudged or mislabeled rows remain"
    assert df["pair_id"].is_unique, "duplicate pair_id in labeled set"

    result: dict = {"n_judged": len(df), "by_channel": {}}
    for (repo, channel, source, stratum), g in df.groupby(
        ["_repo", "_channel", "_source", "_stratum"]
    ):
        n = len(g)
        x = int((g["label"] == "VALID").sum())
        lo, hi = wilson_ci(x, n)
        key = f"{repo}::{channel}::{source}::{stratum}"
        result["by_channel"][key] = {
            "n_judged": n,
            "n_valid": x,
            "precision": round(x / n, 4),
            "wilson95": [round(lo, 4), round(hi, 4)],
            "label_counts": g["label"].value_counts().to_dict(),
            "sample_note": g["_sample_note"].iloc[0],
        }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Wrote %s", OUTPUT_PATH)
    for k, v in result["by_channel"].items():
        log.info(
            "%s: %d/%d VALID (%.1f%%) CI%s  %s",
            k,
            v["n_valid"],
            v["n_judged"],
            v["precision"] * 100,
            v["wilson95"],
            v["label_counts"],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true")
    group.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.sample:
        build_sample()
    else:
        analyze()
