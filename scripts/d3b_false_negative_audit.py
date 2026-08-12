"""D3b/D3c false-negative-contamination check (ADR-0049 follow-up, GG's cheap-check escalation).

Candidate B (full-corpus hard-negative mining) regressed WORSE than D3's original restricted-
pool mining, the opposite of what the candidate-pool-mismatch hypothesis predicted. One
mechanistic explanation ADR-0049 flagged: full-corpus mining draws from a much larger, more
diverse candidate space, so its top-ranked "hard negatives" (highest cosine similarity to the
query) are more likely to actually BE genuinely related issues that the regex-based gold-mining
heuristic simply never captured -- i.e. false negatives. Training a contrastive loss to push a
query and its true near-duplicate apart would directly corrupt useful embedding structure, not
just fail to help. Measured for the full-corpus run (D3b): 11/40 = 27.5% (Wilson95 [16.1,42.8]).

Second question, GG's follow-up escalation: does D3's ORIGINAL restricted-pool mining (the 792-
issue training-pool-restricted candidate space, used by both D3 original and Candidate A) share a
comparable false-negative rate? If yes, contamination explains the BASE regression too, not just
Candidate B's excess -- a solvable data-hygiene problem, not a ceiling. If low, the base
regression stays unexplained by this mechanism and the small-dataset-instability reading (ADR-
0049's cross-run loss/regression pattern) regains standing.

Both checks directly measurable without retraining: sample mined negatives and blind-label each
(query, negative) pair with the SAME validity rubric used throughout this investigation
(VALID/EXCLUDE_UMBRELLA/EXCLUDE_CAUSAL_ONLY/EXCLUDE_OTHER) -- except here VALID means "this
negative is actually a false negative" rather than "this eval pair is a fair test."

Reads:  data/d3_hard_negatives_k8s_related_fullcorpus_negs.parquet   (--run fullcorpus)
        data/d3_hard_negatives_k8s_related.parquet                   (--run restricted)
        data/processed/issues_kubernetes_kubernetes.parquet
Writes: reports/d3b_negs_blind_k8s_{run}.json     (--sample)
        reports/d3b_false_negative_audit_{run}.json  (--analyze, after labeling)

Reproduce: python scripts/d3b_false_negative_audit.py --run restricted --sample
After labeling (fill "label"+"reason" into reports/d3b_negs_labeled_{run}_batch_*.json, same
blind dispatch pattern as scripts/mining_precision_strict_audit.py):
           python scripts/d3b_false_negative_audit.py --run restricted --analyze
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SEED = 42
N_SAMPLE = 40
BATCH_SIZE = 20
MAX_BODY = 512

CORPUS_PATH = Path("data/processed/issues_kubernetes_kubernetes.parquet")

RUNS = {
    "fullcorpus": {
        "negs_path": Path("data/d3_hard_negatives_k8s_related_fullcorpus_negs.parquet"),
        "blind_path": Path("reports/d3b_negs_blind_k8s_fullcorpus.json"),
        "labeled_glob": "reports/d3b_negs_labeled_batch_*.json",
        "output_path": Path("reports/d3b_false_negative_audit.json"),
    },
    "restricted": {
        "negs_path": Path("data/d3_hard_negatives_k8s_related.parquet"),
        "blind_path": Path("reports/d3b_negs_blind_k8s_restricted.json"),
        "labeled_glob": "reports/d3b_negs_labeled_restricted_batch_*.json",
        "output_path": Path("reports/d3b_false_negative_audit_restricted.json"),
    },
}


def _build_text(title: object, body: object) -> str:
    t = (str(title) if title is not None else "").strip()
    b = (str(body) if body is not None else "").strip()[:MAX_BODY]
    return f"{t}. {b}"


def build_sample(run: str) -> None:
    cfg = RUNS[run]
    negs = pd.read_parquet(cfg["negs_path"])
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

    cfg["blind_path"].write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "[%s] wrote %d rows across %d batches to %s",
        run, len(rows), rows[-1]["batch"], cfg["blind_path"],
    )


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (center - adj) / denom, (center + adj) / denom


def analyze(run: str) -> None:
    cfg = RUNS[run]
    labeled: list[dict] = []
    for path in sorted(Path().glob(cfg["labeled_glob"])):
        labeled.extend(json.loads(path.read_text(encoding="utf-8")))

    valid_labels = {"VALID", "EXCLUDE_UMBRELLA", "EXCLUDE_CAUSAL_ONLY", "EXCLUDE_OTHER"}
    ids = [r["pair_id"] for r in labeled]
    assert len(ids) == N_SAMPLE, f"expected {N_SAMPLE} labeled rows, got {len(ids)}"
    assert len(set(ids)) == N_SAMPLE, "duplicate pair_id in labeled set"
    assert all(r["label"] in valid_labels for r in labeled), "unjudged or mislabeled rows remain"

    n = len(labeled)
    x = sum(1 for r in labeled if r["label"] == "VALID")
    lo, hi = wilson_ci(x, n)

    result = {
        "run": run,
        "n_sampled": n,
        "n_false_negatives": x,
        "false_negative_rate": round(x / n, 4),
        "wilson95": [round(lo, 4), round(hi, 4)],
        "label_counts": dict(Counter(r["label"] for r in labeled)),
    }
    cfg["output_path"].write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info(
        "[%s] false-negative rate: %d/%d = %.1f%%  Wilson95 [%.1f, %.1f]",
        run, x, n, 100 * x / n, 100 * lo, 100 * hi,
    )
    logger.info("Wrote %s", cfg["output_path"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", choices=list(RUNS), required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sample", action="store_true")
    group.add_argument("--analyze", action="store_true")
    args = parser.parse_args()
    if args.sample:
        build_sample(args.run)
    else:
        analyze(args.run)
