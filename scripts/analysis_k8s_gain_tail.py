"""Where does the k8s learned-model MAE gain over the train-median point come from? (analysis only)

Owner wording under test (ADR-0064): "on k8s it is marginally better on mean error (+2%), driven by
a small number of long-running issues". Measures the share of the total mean-absolute-error gain
(naive AE - learned AE, per issue) contributed by the longest-resolving 10% of test issues, with a
seeded bootstrap CI. No network, no LLM; reads reports/served_k8s_rows.parquet (produced at
30e6fd8 by analysis_served_k8s_metrics.py, the learned point) and the k8s temporal train frame.

Usage (repo root):  python scripts/analysis_k8s_gain_tail.py
Output: reports/k8s_gain_tail.json
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SEED = 42
N_BOOT = 2000
TAIL_FRACS = (0.01, 0.05, 0.10, 0.20)


def tail_share(y: np.ndarray, gain: np.ndarray, frac: float) -> float:
    """Share of sum(gain) contributed by the `frac` longest-resolving issues (by true y)."""
    k = max(1, int(round(frac * len(y))))
    idx = np.argsort(-y, kind="stable")[:k]
    return float(gain[idx].sum() / gain.sum())


def main() -> None:
    rows = pd.read_parquet(ROOT / "reports/served_k8s_rows.parquet")
    train = pd.read_parquet(ROOT / "data/processed/kubernetes_kubernetes_temporal_train.parquet")
    train = train[train["resolution_hours"] > 0]
    naive_h = float(train["resolution_hours"].median())
    y = rows.true_hours.to_numpy()
    ae_learned = np.abs(rows.pred_hours.to_numpy() - y)
    ae_naive = np.abs(naive_h - y)
    gain = ae_naive - ae_learned  # positive = learned model better on that issue
    n = len(y)

    rows_path = ROOT / "reports/served_k8s_rows.parquet"
    train_path = ROOT / "data/processed/kubernetes_kubernetes_temporal_train.parquet"
    out: dict = {
        "inputs_sha256": {
            "reports/served_k8s_rows.parquet": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
            "data/processed/kubernetes_kubernetes_temporal_train.parquet": hashlib.sha256(
                train_path.read_bytes()
            ).hexdigest(),
        },
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n": n,
        "naive_train_median_hours": naive_h,
        "mean_gain_days": float(gain.mean() / 24),
        "mean_gain_pct_of_naive_mae": float(100 * gain.mean() / ae_naive.mean()),
        "frac_issues_learned_better": float((gain > 0).mean()),
        "median_gain_days": float(np.median(gain) / 24),
        "tails": {},
    }
    rng = np.random.default_rng(SEED)
    boots = rng.integers(0, n, size=(N_BOOT, n))
    for frac in TAIL_FRACS:
        k = max(1, int(round(frac * n)))
        idx = np.argsort(-y, kind="stable")[:k]
        mask = np.zeros(n, dtype=bool)
        mask[idx] = True
        share = tail_share(y, gain, frac)
        bs = np.array([tail_share(y[b], gain[b], frac) for b in boots])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out["tails"][f"top_{int(frac * 100)}pct_by_true_resolution"] = {
            "k_issues": k,
            "min_true_days_in_tail": float(y[idx].min() / 24),
            "share_of_total_gain": round(share, 4),
            "share_ci95": [round(float(lo), 4), round(float(hi), 4)],
            "gain_days_per_issue_in_tail": float(gain[mask].mean() / 24),
            "gain_days_per_issue_in_rest": float(gain[~mask].mean() / 24),
            "rest_mean_gain_sign": "learned better" if gain[~mask].mean() > 0 else "learned worse",
        }
    Path(ROOT / "reports/k8s_gain_tail.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
