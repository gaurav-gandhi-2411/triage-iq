"""Phase C re-eval: k8s LIVE product-task Recall@5 (ADR-0030 action item).

ADR-0028 called this UNMEASURABLE (0 product-task pairs land in the w3-retry TEST split
against the live index). ADR-0030 found that framing wrong: the live-serving index
(`dup_index_kubernetes_kubernetes_bge`, BAAI/bge-base-en-v1.5) is an off-the-shelf
pretrained embedder, never trained on any gold pair -- only the separate, unshipped
`bge_finetuned_*_v2` artifact is. The w3-retry split exists solely to prevent leakage for
THAT fine-tuned model's training; it carries zero leakage risk for the live model. Every
product-stratum pair whose query and target both fall in the live index's number range
(#1-15,002) is usable for measurement now, regardless of its split label.

Re-runs scripts/08_build_similar_issue_index.py's evaluation method (same retrieve() call,
same self-exclusion) restricted to those pairs, against the actually-deployed index -- not
a new mining or scraping effort. Bootstrap CI uses the same method (2000 resamples, seed=42,
percentile) as the vscode honest number (README.md, ADR-0028) so the two are comparable.

Reads:
  data/gold_related_v2.parquet                    -- stratum-labeled gold (ADR-0027)
  data/models/dup_index_kubernetes_kubernetes_bge/ -- LIVE v1 serving index (src/triage_iq/api/loader.py)

Output: reports/phaseC_k8s_live_product_eval.json
Reproduce: python scripts/phaseC_k8s_live_product_eval.py
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
REPO = "kubernetes_kubernetes"

GOLD_PATH = Path("data/gold_related_v2.parquet")
INDEX_DIR = Path("data/models/dup_index_kubernetes_kubernetes_bge")
OUTPUT_PATH = Path("reports/phaseC_k8s_live_product_eval.json")

# Already-established comparable number (README.md, ADR-0028): vscode honest product-task
# recall@5, live v1 index, measured with this same bootstrap method.
VSCODE_LIVE_PRODUCT_R5 = {"n": 254, "recall_at_5": 0.2244, "ci95": [0.1774, 0.2796]}


def select_live_product_pairs(gold: pd.DataFrame, live_numbers: set[int]) -> pd.DataFrame:
    prod = gold[(gold["repo"] == REPO) & (gold["stratum"] == "product")].copy()
    in_live = prod.apply(
        lambda r: (
            int(r["query_number"]) in live_numbers and int(r["original_number"]) in live_numbers
        ),
        axis=1,
    )
    log.info(
        "product-stratum pairs: %d total, %d fall in the live index's number range "
        "(zero leakage risk -- live model is pretrained, never trained on any gold pair, "
        "ADR-0030)",
        len(prod),
        int(in_live.sum()),
    )
    return prod[in_live].reset_index(drop=True)


def bootstrap_ci(hits: np.ndarray) -> tuple[float, float]:
    """Single-proportion percentile bootstrap -- same method as vscode's 22.4% [17.7, 28.0]."""
    rng = np.random.default_rng(SEED)
    n = len(hits)
    means = [hits[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    detector = SimilarIssueRetriever.load(str(INDEX_DIR))
    live_numbers = {int(n) for n in detector.issue_numbers}
    log.info("Loaded live k8s index: %s (%d records)", INDEX_DIR, len(live_numbers))

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
        "membership, not w3-retry split label -- ADR-0030 zero-leakage reasoning "
        "(pretrained, untrained embedder).",
    }
    for k in K_VALUES:
        result[f"recall_at_{k}"] = float(np.mean([any(h[:k]) for h in hit_lists]))

    r5_hits = np.array([float(any(h[:5])) for h in hit_lists])
    ci_lo, ci_hi = bootstrap_ci(r5_hits)
    result["recall_at_5_ci95"] = [round(ci_lo, 4), round(ci_hi, 4)]
    result["bootstrap"] = {"n_resamples": N_BOOTSTRAP, "seed": SEED, "method": "percentile"}
    result["vscode_comparable"] = VSCODE_LIVE_PRODUCT_R5

    log.info(
        "[k8s] LIVE product-task Recall@5 = %.4f  n=%d  95%% CI [%.4f, %.4f]",
        result["recall_at_5"],
        result["n_pairs"],
        ci_lo,
        ci_hi,
    )
    log.info(
        "[vscode, comparable] Recall@5 = %.4f  n=%d  95%% CI %s",
        VSCODE_LIVE_PRODUCT_R5["recall_at_5"],
        VSCODE_LIVE_PRODUCT_R5["n"],
        VSCODE_LIVE_PRODUCT_R5["ci95"],
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2))
    log.info("Wrote %s", OUTPUT_PATH)


if __name__ == "__main__":
    main()
