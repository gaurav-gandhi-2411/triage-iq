"""Shared helpers for the retrieval-quality-improvement levers (spec.md, ADR-0031).

Reused by scripts/lever1_hybrid_bm25_rrf.py, lever2_reranker.py, lever3_stronger_embedder.py
so all three levers are gated with the identical pair-selection and CI methodology as the
baseline reproduction (scripts/phaseC_{k8s,vscode}_live_product_eval.py) and the paired
bootstrap used for ADR-0027/w3_t5_eval.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_BOOTSTRAP = 2000
K_VALUES = [1, 5, 10, 20]
MAX_BODY = 512

# D1's canonical, hand-verified, issue-level-disjoint eval sets (ADR-0033) -- the SAME
# population scripts/d1_baseline_eval.py measures the canonical baseline against, and the
# SAME index (full current corpus, not the stale served dup_index_*) it uses. Supersedes
# select_live_product_pairs()/gold_related_v2.parquet below, which was the ADR-0030-era,
# unaudited (72% k8s / 20% vscode genuine, per ADR-0032) population ADR-0031's levers
# originally used -- see the ADR correcting ADR-0031/0033/0034.
D1_EVAL_SET_BY_REPO = {
    "kubernetes_kubernetes": "reports/d1_eval_set_k8s_related.json",
    "microsoft_vscode": "reports/d1_eval_set_vscode_duplicate.json",
}
D1_INDEX_DIR_BY_REPO = {
    "kubernetes_kubernetes": "data/models/d1_full_corpus_index_kubernetes_kubernetes_bge",
    "microsoft_vscode": "data/models/d1_full_corpus_index_microsoft_vscode_bge",
}


def load_d1_eval_pairs(repo: str) -> pd.DataFrame:
    """Load D1's canonical eval-set pairs for `repo` (query_body-augmented) as a DataFrame,
    same column shape (`query_title`, `query_body`, `query_number`, `original_number`,
    `original_title`) select_live_product_pairs() used to return.
    """
    path = Path(D1_EVAL_SET_BY_REPO[repo])
    return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))


def select_live_product_pairs(gold: pd.DataFrame, repo: str, live_numbers: set[int]) -> pd.DataFrame:
    """DEPRECATED for the three levers -- see load_d1_eval_pairs() above. Kept only because
    it's still historically referenced (ADR-0031's original numbers used it); not called by
    lever1/2/3 anymore.

    Product-stratum pairs for `repo` whose query and target both fall in the live index.
    Same selection rule as phaseC_k8s_live_product_eval.py / phaseC_vscode_live_product_eval.py:
    filtered by live-index membership, not w3-retry split label (ADR-0030 zero-leakage
    reasoning -- the live model is a pretrained, untrained-on-gold-pairs embedder).
    """
    prod = gold[(gold["repo"] == repo) & (gold["stratum"] == "product")].copy()
    in_live = prod.apply(
        lambda r: (
            int(r["query_number"]) in live_numbers and int(r["original_number"]) in live_numbers
        ),
        axis=1,
    )
    return prod[in_live].reset_index(drop=True)


def query_text(row: pd.Series) -> str:
    # Byte-identical to production (triage.py::_collect_signals): f"{title}. {body}", UNTRUNCATED.
    # The prior [:MAX_BODY] truncation was an eval-only divergence from prod; see the ADR
    # correcting ADR-0031/0033/0034. Corpus-side truncation (_build_text, MAX_BODY) is untouched.
    return str(row["query_title"]) + ". " + str(row["query_body"])


def paired_bootstrap_ci(base_hits: np.ndarray, new_hits: np.ndarray) -> tuple[float, float, float]:
    """TRUE paired bootstrap on the delta (new - base). Verbatim method from
    scripts/w3_t5_eval.py::paired_bootstrap_ci (ADR-0027's primary/corrected method):
    same resample indices for both arms, so the delta's own sampling distribution is
    what's resampled -- not two independently-resampled proportions.

    Returns (ci_lo, ci_hi, point_delta).
    """
    rng = np.random.default_rng(SEED)
    n = len(base_hits)
    d = new_hits - base_hits
    deltas = [d[rng.integers(0, n, n)].mean() for _ in range(N_BOOTSTRAP)]
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), float(d.mean())


def recall_at_k_hits(hit_flags: list[bool], k: int) -> bool:
    return any(hit_flags[:k])
