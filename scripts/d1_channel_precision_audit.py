"""D1: precision audit for retrieval clean-pool channels not yet hand-verified.

ADR-0032 hand-judged a 50-pair sample (25/repo) of the *live product-eval* pair sets and
established title_sim precision (k8s ~mixed, vscode 20%) plus a partial k8s_extended_mine
read (n=20, 65%). ADR-0030 separately measured vscode's extended-body channel (Channel A /
`vscode_body_refs`+`body_related_ext`) at 30-43% (n=30) -- noisy, not a clean channel.

Neither prior audit ever sampled two channels that matter for the D1 clean TRAINING pool
(not just the live-index eval subset):
  1. k8s_forward_scrape / body_related(_ext) / stratum=product (499 pairs) -- issues #15003+,
     outside the live index at ADR-0032 time, so never eligible for the live-eval sample.
  2. vscode_dup_scrape / dup_comment / stratum=gate (2242 pairs) -- excluded from ADR-0032's
     "live product eval" sample by construction (that sample only covered stratum=="product").
     This is vscode's single largest pair channel and the deciding factor in its clean-pool
     verdict.

This script draws fixed-seed (SEED=42) samples from those two channels, plus a full census
(no sampling error) of two tiny candidate pools where n is small enough to just read all of
them: k8s legacy body_ref/product (2 pairs) and vscode's "narrow" reference pool -- legacy
body_ref/product (1) + legacy body_related/product (10) + vscode_body_refs body_related/product
(11, NOT body_related_ext -- that's the noisy Channel A already excluded) = 22 pairs.

Reads:  data/gold_related_v2.parquet
Writes: reports/d1_pair_sample_for_review.json   -- pairs to hand-judge (verdict: null)

Reproduce: python scripts/d1_channel_precision_audit.py --sample
After hand-judging (fill "verdict": "genuine"|"incidental" + "verdict_reason"):
           python scripts/d1_channel_precision_audit.py --analyze
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
SAMPLE_PATH = Path("reports/d1_pair_sample_for_review.json")
REVIEW_PATH = Path("reports/d1_pair_quality_review.json")
AUDIT_OUTPUT = Path("reports/d1_channel_precision_audit.json")

MAX_BODY_FOR_REVIEW = 1500

# (repo, channel, source, stratum, sample_size or None for full census)
TARGETS = [
    ("kubernetes_kubernetes", "k8s_forward_scrape", "body_related", "product", None),  # 45, census
    ("kubernetes_kubernetes", "k8s_forward_scrape", "body_related_ext", "product", 30),  # of 454
    ("microsoft_vscode", "vscode_dup_scrape", "dup_comment", "gate", 40),  # of 2242
    ("kubernetes_kubernetes", "legacy_gold_v1", "body_ref", "product", None),  # 2, census
    ("microsoft_vscode", "legacy_gold_v1", "body_ref", "product", None),  # 1, census
    ("microsoft_vscode", "legacy_gold_v1", "body_related", "product", None),  # 10, census
    ("microsoft_vscode", "vscode_body_refs", "body_related", "product", None),  # 11, census
]


def build_sample() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    rows = []
    for repo, channel, source, stratum, n in TARGETS:
        subset = gold[
            (gold["repo"] == repo)
            & (gold["channel"] == channel)
            & (gold["source"] == source)
            & (gold["stratum"] == stratum)
        ]
        if n is None or len(subset) <= n:
            sample = subset
            note = "full_census"
        else:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(subset), size=n, replace=False)
            sample = subset.iloc[sorted(idx)]
            note = f"sample_{n}_of_{len(subset)}"
        log.info(
            "[%s/%s/%s/%s] %s -> %d rows drawn", repo, channel, source, stratum, note, len(sample)
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
                    "verdict": None,
                    "verdict_reason": None,
                }
            )
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    log.info("Wrote %d-pair sample to %s", len(rows), SAMPLE_PATH)


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (center - adj) / denom, (center + adj) / denom


def analyze() -> None:
    review = pd.DataFrame(json.loads(REVIEW_PATH.read_text(encoding="utf-8")))
    assert review["verdict"].isin(["genuine", "incidental"]).all(), "unjudged rows remain"

    result: dict = {"n_judged": len(review), "by_channel": {}}
    for (repo, channel, source, stratum), g in review.groupby(
        ["repo", "channel", "source", "stratum"]
    ):
        n = len(g)
        x = int((g["verdict"] == "genuine").sum())
        lo, hi = wilson_ci(x, n)
        key = f"{repo}::{channel}::{source}::{stratum}"
        result["by_channel"][key] = {
            "n_judged": n,
            "n_genuine": x,
            "precision": round(x / n, 4),
            "wilson95": [round(lo, 4), round(hi, 4)],
            "sample_note": g["sample_note"].iloc[0],
        }
    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Wrote %s", AUDIT_OUTPUT)
    for k, v in result["by_channel"].items():
        log.info(
            "%s: %d/%d genuine (%.1f%%) CI%s",
            k,
            v["n_genuine"],
            v["n_judged"],
            v["precision"] * 100,
            v["wilson95"],
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
