"""D3b false-negative-contamination check (ADR-0049 follow-up, GG's cheap-check escalation).

Candidate B (full-corpus hard-negative mining) regressed WORSE than D3's original restricted-
pool mining, the opposite of what the candidate-pool-mismatch hypothesis predicted. One
mechanistic explanation ADR-0049 flagged but didn't measure: full-corpus mining draws from a much
larger, more diverse candidate space, so its top-ranked "hard negatives" (highest cosine
similarity to the query) are more likely to actually BE genuinely related issues that the
regex-based gold-mining heuristic simply never captured -- i.e. false negatives. Training a
contrastive loss to push a query and its true near-duplicate apart would directly corrupt useful
embedding structure, not just fail to help.

This is directly measurable without retraining: sample mined negatives and blind-label each
(query, negative) pair with the SAME validity rubric used throughout this investigation
(VALID/EXCLUDE_UMBRELLA/EXCLUDE_CAUSAL_ONLY/EXCLUDE_OTHER) -- except here VALID means "this
negative is actually a false negative" rather than "this eval pair is a fair test."

Reads:  data/d3_hard_negatives_k8s_related_fullcorpus_negs.parquet
        data/processed/issues_kubernetes_kubernetes.parquet
Writes: reports/d3b_negs_blind_k8s_fullcorpus.json   (--sample)
        reports/d3b_false_negative_audit.json         (--analyze, after labeling)

Reproduce: python scripts/d3b_false_negative_audit.py --sample
After labeling (fill "label"+"reason" into reports/d3b_negs_labeled_batch_*.json, same blind
dispatch pattern as scripts/mining_precision_strict_audit.py):
           python scripts/d3b_false_negative_audit.py --analyze
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
N_SAMPLE = 40
BATCH_SIZE = 20
MAX_BODY = 512

NEGS_PATH = Path("data/d3_hard_negatives_k8s_related_fullcorpus_negs.parquet")
CORPUS_PATH = Path("data/processed/issues_kubernetes_kubernetes.parquet")
BLIND_PATH = Path("reports/d3b_negs_blind_k8s_fullcorpus.json")
LABELED_GLOB = "reports/d3b_negs_labeled_batch_*.json"
OUTPUT_PATH = Path("reports/d3b_false_negative_audit.json")


def _build_text(title: object, body: object) -> str:
    t = (str(title) if title is not None else "").strip()
    b = (str(body) if body is not None else "").strip()[:MAX_BODY]
    return f"{t}. {b}"


def build_sample() -> None:
    negs = pd.read_parquet(NEGS_PATH)
    corpus = pd.read_parquet(CORPUS_PATH, columns=["number", "title", "body_clean"])
    title_by_num = dict(zip(corpus["number"].astype(int), corpus["title"], strict=True))
    body_by_num = dict(zip(corpus["number"].astype(int), corpus["body_clean"], strict=True))

    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(negs), size=N_SAMPLE, replace=False)
    sample = negs.iloc[sorted(idx)].reset_index(drop=True)

    rows: list[dict] = []
    for i, r in sample.iterrows():
        qn, nn = int(r["query_number"]), int(r["neg_number"])
        rows.append(
            {
                "pair_id": i,
                "query_number": qn,
                "neg_number": nn,
                "neg_rank": int(r["neg_rank"]),
                "neg_score": round(float(r["neg_score"]), 4),
                "query_text": _build_text(title_by_num.get(qn, "?"), body_by_num.get(qn, "")),
                "neg_text": r["neg_text"],
                "label": None,
                "reason": None,
            }
        )
    for i, row in enumerate(rows):
        row["batch"] = i // BATCH_SIZE + 1

    BLIND_PATH.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "wrote %d rows across %d batches to %s", len(rows), rows[-1]["batch"], BLIND_PATH
    )


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (center - adj) / denom, (center + adj) / denom


def analyze() -> None:
    labeled: list[dict] = []
    for path in sorted(Path().glob(LABELED_GLOB)):
        labeled.extend(json.loads(path.read_text(encoding="utf-8")))

    valid_labels = {"VALID", "EXCLUDE_UMBRELLA", "EXCLUDE_CAUSAL_ONLY", "EXCLUDE_OTHER"}
    ids = [r["pair_id"] for r in labeled]
    assert len(ids) == N_SAMPLE, f"expected {N_SAMPLE} labeled rows, got {len(ids)}"
    assert len(set(ids)) == N_SAMPLE, "duplicate pair_id in labeled set"
    assert all(r["label"] in valid_labels for r in labeled), "unjudged or mislabeled rows remain"

    n = len(labeled)
    x = sum(1 for r in labeled if r["label"] == "VALID")
    lo, hi = wilson_ci(x, n)
    from collections import Counter

    result = {
        "n_sampled": n,
        "n_false_negatives": x,
        "false_negative_rate": round(x / n, 4),
        "wilson95": [round(lo, 4), round(hi, 4)],
        "label_counts": dict(Counter(r["label"] for r in labeled)),
    }
    OUTPUT_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(
        "false-negative rate: %d/%d = %.1f%%  Wilson95 [%.1f, %.1f]",
        x, n, 100 * x / n, 100 * lo, 100 * hi,
    )
    logger.info("Wrote %s", OUTPUT_PATH)


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
