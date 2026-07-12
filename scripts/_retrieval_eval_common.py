"""Shared helpers for the retrieval-quality-improvement levers (spec.md, ADR-0031).

Reused by scripts/lever1_hybrid_bm25_rrf.py, lever2_reranker.py, lever3_stronger_embedder.py
so all three levers are gated with the identical pair-selection and CI methodology as the
baseline reproduction (scripts/phaseC_{k8s,vscode}_live_product_eval.py) and the paired
bootstrap used for ADR-0027/w3_t5_eval.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
N_BOOTSTRAP = 2000
K_VALUES = [1, 5, 10, 20]
MAX_BODY = 512


def select_live_product_pairs(gold: pd.DataFrame, repo: str, live_numbers: set[int]) -> pd.DataFrame:
    """Product-stratum pairs for `repo` whose query and target both fall in the live index.

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
    return str(row["query_title"]) + ". " + str(row["query_body"])[:MAX_BODY]


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
