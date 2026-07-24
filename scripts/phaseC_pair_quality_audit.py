"""Quality audit of the product-task gold pairs (ADR-0031 follow-up).

Three retrieval-quality levers (ADR-0031) all failed to move the ~23-27% product-task
Recall@5 base rate. Before accepting that as a ceiling, this script checks a cheaper
alternative explanation first: are the gold pairs themselves noisy? The product-task
pairs are mined from "related to #N"-style signals (channels A/B/D, ADR-0030) -- if a
meaningful fraction aren't genuinely related, low R@5 reflects noisy ground truth, not
retriever failure.

This script does NOT judge relatedness itself (that requires reading issue text, done
by hand -- see reports/phaseC_pair_quality_review.json, populated after manual review).
It only:
  1. Reproduces the exact live product-stratum pair sets (same method as
     phaseC_{k8s,vscode}_live_product_eval.py) and draws a fixed-seed sample for hand
     judging, writing the full pair text to reports/phaseC_pair_sample_for_review.json.
  2. Given a completed hand-judging file (repo, query_number, original_number, verdict),
     computes: pair-set precision, clean-subset (genuine-only) R@5 vs full-set R@5, and
     query/target vocabulary (token) overlap for genuine pairs the retriever misses.

Reads:
  data/gold_related_v2.parquet
  data/models/dup_index_kubernetes_kubernetes_bge/
  data/models/dup_index_microsoft_vscode_bge/

Outputs:
  reports/phaseC_pair_sample_for_review.json   -- sample to hand-judge (step 1)
  reports/phaseC_pair_quality_audit.json       -- precision + clean R@5 + miss analysis (step 2)

Reproduce:
  python scripts/phaseC_pair_quality_audit.py --sample     # writes the sample to judge
  python scripts/phaseC_pair_quality_audit.py --analyze    # after reports/phaseC_pair_quality_review.json exists
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from _retrieval_eval_common import (  # noqa: E402
    N_BOOTSTRAP,
    SEED,
    query_text,
    select_live_product_pairs,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

GOLD_PATH = Path("data/gold_related_v2.parquet")
INDEX_DIRS = {
    "kubernetes_kubernetes": Path("data/models/dup_index_kubernetes_kubernetes_bge"),
    "microsoft_vscode": Path("data/models/dup_index_microsoft_vscode_bge"),
}
SAMPLE_PATH = Path("reports/phaseC_pair_sample_for_review.json")
REVIEW_PATH = Path("reports/phaseC_pair_quality_review.json")
AUDIT_OUTPUT = Path("reports/phaseC_pair_quality_audit.json")

SAMPLE_PER_REPO = 25  # 50 total, ~40-50 target
MAX_BODY_FOR_REVIEW = 1500

TOKEN_RE = re.compile(r"[a-z0-9_]{3,}")
STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "have",
    "not",
    "are",
    "was",
    "but",
    "when",
    "would",
    "should",
    "could",
    "will",
    "can",
    "does",
    "did",
    "has",
    "you",
    "your",
    "it's",
    "its",
    "into",
    "then",
    "than",
    "also",
    "just",
    "only",
    "using",
    "used",
    "use",
    "issue",
    "issues",
    "problem",
    "error",
    "code",
    "closes",
    "fixes",
    "fix",
    "related",
    "similar",
    "see",
    "please",
    "http",
    "https",
    "com",
    "github",
}


def load_live_numbers(repo: str) -> set[int]:
    detector = SimilarIssueRetriever.load(str(INDEX_DIRS[repo]))
    return {int(n) for n in detector.issue_numbers}


def build_sample() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    rows = []
    for repo in INDEX_DIRS:
        live_numbers = load_live_numbers(repo)
        pairs = select_live_product_pairs(gold, repo, live_numbers)
        log.info("[%s] %d live product-stratum pairs", repo, len(pairs))
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(pairs), size=min(SAMPLE_PER_REPO, len(pairs)), replace=False)
        sample = pairs.iloc[sorted(idx)]
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
                    "confidence": row["confidence"],
                    "verdict": None,  # to fill by hand: "genuine" | "incidental"
                    "verdict_reason": None,
                }
            )
    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE_PATH.write_text(json.dumps(rows, indent=2))
    log.info("Wrote %d-pair sample (%d per repo) to %s", len(rows), SAMPLE_PER_REPO, SAMPLE_PATH)


def _tokens(text: str) -> set[str]:
    return {t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS}


def bootstrap_ci(hits: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    n = len(hits)
    if n == 0:
        return (float("nan"), float("nan"))
    means = [hits[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval -- no extra dependency, standard closed form."""
    p = x / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (center - adj) / denom, (center + adj) / denom


def channel_composition(gold: pd.DataFrame) -> dict:
    """Channel/source breakdown of each repo's live-evaluated product pair set.

    Explains the precision asymmetry between repos (ADR-0032): k8s's set is dominated by
    reference-mined channels (ADR-0030 measured 78-89% precision); vscode's is dominated by
    `title_sim` (title-text similarity), never precision-audited before this ADR.
    """
    composition: dict = {}
    for repo in INDEX_DIRS:
        live_numbers = load_live_numbers(repo)
        pairs = select_live_product_pairs(gold, repo, live_numbers)
        legacy = pairs[pairs["channel"] == "legacy_gold_v1"]
        composition[repo] = {
            "n": len(pairs),
            "by_channel": pairs["channel"].value_counts().to_dict(),
            "by_source_within_legacy_gold_v1": legacy["source"].value_counts().to_dict()
            if len(legacy)
            else {},
        }
    return composition


def analyze() -> None:
    review = json.loads(REVIEW_PATH.read_text())
    gold = pd.read_parquet(GOLD_PATH)

    judged = pd.DataFrame(review)
    assert judged["verdict"].isin(["genuine", "incidental"]).all(), "unjudged rows remain"

    precision_overall = float((judged["verdict"] == "genuine").mean())
    precision_by_repo = (
        judged.groupby("repo")["verdict"].apply(lambda s: float((s == "genuine").mean())).to_dict()
    )

    n_genuine_overall = int((judged["verdict"] == "genuine").sum())
    result: dict = {
        "n_sampled": len(judged),
        "precision_overall": round(precision_overall, 4),
        "precision_overall_wilson95": [
            round(c, 4) for c in wilson_ci(n_genuine_overall, len(judged))
        ],
        "precision_by_repo": {k: round(v, 4) for k, v in precision_by_repo.items()},
        "precision_by_repo_wilson95": {
            repo: [
                round(c, 4)
                for c in wilson_ci(
                    int((judged[judged["repo"] == repo]["verdict"] == "genuine").sum()),
                    len(judged[judged["repo"] == repo]),
                )
            ]
            for repo in judged["repo"].unique()
        },
        "sample_composition": judged["repo"].value_counts().to_dict(),
    }

    detectors = {repo: SimilarIssueRetriever.load(str(path)) for repo, path in INDEX_DIRS.items()}

    per_repo_clean: dict = {}
    miss_vocab_overlaps: list[dict] = []

    for repo in INDEX_DIRS:
        detector = detectors[repo]
        repo_judged = judged[judged["repo"] == repo]
        genuine = repo_judged[repo_judged["verdict"] == "genuine"]
        if genuine.empty:
            continue

        # Need full row data (title/body) -- pull back from gold by (query_number, original_number).
        gold_repo = gold[gold["repo"] == repo]
        hits = []
        for _, srow in genuine.iterrows():
            grow = gold_repo[
                (gold_repo["query_number"] == srow["query_number"])
                & (gold_repo["original_number"] == srow["original_number"])
            ]
            if grow.empty:
                log.warning(
                    "sampled pair not found back in gold (unexpected): %s %s->%s",
                    repo,
                    srow["query_number"],
                    srow["original_number"],
                )
                continue
            grow = grow.iloc[0]
            qtext = query_text(grow)
            results = detector.retrieve(qtext, k=5, exclude_number=int(grow["query_number"]))
            retrieved = {r["number"] for r in results}
            pos = int(grow["original_number"])
            hit = pos in retrieved
            hits.append(hit)
            if not hit:
                overlap = _tokens(
                    str(grow["query_title"]) + " " + str(grow["query_body"])
                ) & _tokens(str(grow["original_title"]) + " " + str(grow["original_body"]))
                miss_vocab_overlaps.append(
                    {
                        "repo": repo,
                        "query_number": int(grow["query_number"]),
                        "original_number": int(grow["original_number"]),
                        "shared_token_count": len(overlap),
                        "shared_tokens_sample": sorted(overlap)[:15],
                    }
                )

        hits_arr = np.array([float(h) for h in hits])
        ci = bootstrap_ci(hits_arr)
        per_repo_clean[repo] = {
            "n_genuine_sampled": len(genuine),
            "recall_at_5_on_genuine_subset": round(float(hits_arr.mean()), 4)
            if len(hits_arr)
            else None,
            "ci95_note": "small-n sample CI, illustrative only" if len(hits_arr) < 30 else None,
            "ci95": [round(c, 4) for c in ci],
        }

    result["clean_subset_recall_at_5"] = per_repo_clean
    result["genuine_misses_vocab_overlap"] = miss_vocab_overlaps
    result["genuine_misses_zero_overlap_count"] = sum(
        1 for m in miss_vocab_overlaps if m["shared_token_count"] == 0
    )
    result["genuine_misses_total"] = len(miss_vocab_overlaps)
    result["channel_composition_of_live_product_sets"] = channel_composition(gold)

    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_OUTPUT.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s", AUDIT_OUTPUT)
    log.info("Precision overall: %.1f%%  by repo: %s", precision_overall * 100, precision_by_repo)
    log.info("Clean-subset R@5: %s", per_repo_clean)


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
