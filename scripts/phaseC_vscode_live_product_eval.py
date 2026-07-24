"""Phase C re-eval: vscode LIVE product-task Recall@5 (baseline reproduction).

ADR-0028 reported vscode's honest product-task number as n=254, R@5=22.44%,
CI[17.7,28.0] -- but the audit JSON (reports/model_eval_audit.json) computed it
against a denominator of 281 evaluable pairs, not the full 505-row product
stratum currently in data/gold_related_v2.parquet (unchanged since the single
commit that created it, cff0c19). No script producing that number was ever
committed. This script applies the SAME method used for k8s (ADR-0030,
scripts/phaseC_k8s_live_product_eval.py): every product-stratum pair whose
query and target both fall in the live index's issue-number set, regardless
of w3-retry split label (zero leakage risk -- live model is pretrained,
untrained on any gold pair). It is expected to surface whatever gap exists
between 254 and the full in-range count so that gap can be explained before
this baseline is used to gate levers 1-3.

Reads:
  data/gold_related_v2.parquet                 -- stratum-labeled gold (ADR-0027)
  data/models/dup_index_microsoft_vscode_bge/   -- LIVE v1 serving index (src/triage_iq/api/loader.py)

Output: reports/phaseC_vscode_live_product_eval.json
Reproduce: python scripts/phaseC_vscode_live_product_eval.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

SEED = 42
N_BOOTSTRAP = 2000
K_VALUES = [1, 5, 10, 20]
MAX_BODY = 512
REPO = "microsoft_vscode"

GOLD_PATH = Path("data/gold_related_v2.parquet")
INDEX_DIR = Path("data/models/dup_index_microsoft_vscode_bge")
OUTPUT_PATH = Path("reports/phaseC_vscode_live_product_eval.json")

# ADR-0028 / reports/model_eval_audit.json: n=254, denominator 281 (not the full
# 505-row product stratum) -- kept here for the discrepancy check, not as ground truth.
ADR0028_REPORTED = {"n": 254, "denominator": 281, "recall_at_5": 0.2244, "ci95": [0.1774, 0.2796]}


def select_live_product_pairs(gold: pd.DataFrame, live_numbers: set[int]) -> pd.DataFrame:
    prod = gold[(gold["repo"] == REPO) & (gold["stratum"] == "product")].copy()
    in_live = prod.apply(
        lambda r: (
            int(r["query_number"]) in live_numbers and int(r["original_number"]) in live_numbers
        ),
        axis=1,
    )
    log.info(
        "product-stratum pairs: %d total, %d fall in the live index's issue-number set",
        len(prod),
        int(in_live.sum()),
    )
    return prod[in_live].reset_index(drop=True)


def bootstrap_ci(hits: np.ndarray) -> tuple[float, float]:
    """Single-proportion percentile bootstrap -- same method as ADR-0028/ADR-0030."""
    rng = np.random.default_rng(SEED)
    n = len(hits)
    means = [hits[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    detector = SimilarIssueRetriever.load(str(INDEX_DIR))
    live_numbers = {int(n) for n in detector.issue_numbers}
    log.info("Loaded live vscode index: %s (%d records)", INDEX_DIR, len(live_numbers))

    pairs = select_live_product_pairs(gold, live_numbers)
    k_max = max(K_VALUES)

    hit_lists: list[list[bool]] = []
    for _, row in pairs.iterrows():
        query_text = str(row["query_title"]) + ". " + str(row["query_body"])[:MAX_BODY]
        results = detector.retrieve(query_text, k=k_max, exclude_number=int(row["query_number"]))
        retrieved = [r["number"] for r in results]
        pos = int(row["original_number"])
        hit_lists.append([n == pos for n in retrieved])

    result: dict = {
        "repo": REPO,
        "index": f"live v1 ({len(live_numbers)} records, actually deployed)",
        "n_pairs": len(pairs),
        "method": "product-stratum pairs (gold_related_v2.parquet) filtered by live-index "
        "membership, not w3-retry split label -- same method as "
        "phaseC_k8s_live_product_eval.py (ADR-0030 zero-leakage reasoning).",
    }
    for k in K_VALUES:
        result[f"recall_at_{k}"] = float(np.mean([any(h[:k]) for h in hit_lists]))

    r5_hits = np.array([float(any(h[:5])) for h in hit_lists])
    ci_lo, ci_hi = bootstrap_ci(r5_hits)
    result["recall_at_5_ci95"] = [round(ci_lo, 4), round(ci_hi, 4)]
    result["bootstrap"] = {"n_resamples": N_BOOTSTRAP, "seed": SEED, "method": "percentile"}
    result["adr0028_reported"] = ADR0028_REPORTED

    log.info(
        "[vscode] LIVE product-task Recall@5 = %.4f  n=%d  95%% CI [%.4f, %.4f]",
        result["recall_at_5"],
        result["n_pairs"],
        ci_lo,
        ci_hi,
    )
    log.info(
        "[ADR-0028, reported] Recall@5 = %.4f  n=%d (denom %d)  95%% CI %s",
        ADR0028_REPORTED["recall_at_5"],
        ADR0028_REPORTED["n"],
        ADR0028_REPORTED["denominator"],
        ADR0028_REPORTED["ci95"],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
