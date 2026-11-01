"""Calibrate the served k8s resolution constant and its conformal interval (G3, D88). Analysis only.

Constant = median(T hours) / 24 over the qp4 k8s TRAIN closed eligible issues (2023-2025). Interval =
the qp4 M0a/CQR recipe (ALPHA 0.2, target coverage 0.80, same as cqr_conformal_adjustments_v2.json):
quantiles q10/q90 of T on the first 60% (chronological) of the train closed issues, CQR adjustment Q in
log1p space from the last 40% (qp4_resolution.cqr_q). The test split is never read, and validation is
only scored, never fitted. Pre-registered in the R7 task before the first computation.

Usage: python scripts/k8s_constant_calibration.py FEATURES_PARQUET
Output: reports/k8s_constant_35d_calibration.json (the values pinned in models/resolution.py).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
ALPHA, SEED, N_BOOT, A_FRAC = 0.2, 42, 2000, 0.6


def _log(v: np.ndarray) -> np.ndarray:
    return np.log1p(np.maximum(v, 0))


def cqr_q(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float = ALPHA) -> float:
    """Same as qp4_resolution.cqr_q (kept local: that script is not importable as a module)."""
    sc = np.sort(np.maximum(lo - y, y - hi))
    k = int(np.ceil((len(sc) + 1) * (1 - alpha)))
    return float(sc[min(k, len(sc)) - 1])


def calibrate(train_closed_hours: np.ndarray) -> dict:
    """train_closed_hours: T of the train closed issues sorted by created_at."""
    k = int(np.floor(len(train_closed_hours) * A_FRAC))
    a, b = train_closed_hours[:k], train_closed_hours[k:]
    q10, q90 = np.quantile(a, [0.1, 0.9])
    q = cqr_q(_log(np.array(q10)), _log(np.array(q90)), _log(b))
    lo_h = max(0.0, float(np.expm1(_log(np.array(q10)) - q)))
    hi_h = float(np.expm1(_log(np.array(q90)) + q))
    return {"n_a": len(a), "n_b": len(b), "q_log": q, "lo_h": lo_h, "hi_h": hi_h,
            "q10_h": float(q10), "q90_h": float(q90)}  # fmt: skip


def main(path: Path) -> None:
    d = pd.read_parquet(path, columns=["split", "classifier_eligible", "created_at", "T", "event"])
    d = d[d["classifier_eligible"]]
    tr = d[(d["split"] == "train") & (d["event"] == 1)].sort_values("created_at")
    const_h = float(np.median(tr["T"]))
    cal = calibrate(tr["T"].to_numpy())
    v = d[d["split"] == "validation"]
    vc, vo = v[v["event"] == 1]["T"].to_numpy(), v[v["event"] == 0]["T"].to_numpy()
    cov = (vc >= cal["lo_h"]) & (vc <= cal["hi_h"])
    rng = np.random.default_rng(SEED)
    boots = [cov[rng.integers(0, len(cov), len(cov))].mean() for _ in range(N_BOOT)]
    ci = np.percentile(boots, [2.5, 97.5])
    ae = np.abs(vc - const_h) / 24  # absolute error of the constant, days, on closed validation
    idx = [rng.integers(0, len(ae), len(ae)) for _ in range(N_BOOT)]
    ci_mean = np.percentile([ae[i].mean() for i in idx], [2.5, 97.5])
    ci_med = np.percentile([np.median(ae[i]) for i in idx], [2.5, 97.5])
    n_all, surely_miss = len(vc) + len(vo), int((vo > cal["hi_h"]).sum())
    out = {
        "method": "qp4 M0a CQR recipe: q10/q90 on first 60% of train closed (chronological), "
        "Q on last 40% in log1p space, alpha 0.2; test never read",
        "source": {"file": "qp4/features.parquet", "sha256_prefix": hashlib.sha256(
            path.read_bytes()).hexdigest()[:16]},  # fmt: skip
        "n_train_closed": len(tr), **cal,
        "served": {"point_days": round(const_h / 24, 4), "lo_days": round(cal["lo_h"] / 24, 4),
                   "hi_days": round(cal["hi_h"] / 24, 4)},  # fmt: skip
        "validation": {
            "n_closed": len(vc), "n_open": len(vo), "target_coverage": 1 - ALPHA,
            "closed_coverage": float(cov.mean()), "closed_coverage_ci95": [float(x) for x in ci],
            "mean_ae_days": float(ae.mean()), "mean_ae_ci95": [float(x) for x in ci_mean],
            "median_ae_days": float(np.median(ae)), "median_ae_ci95": [float(x) for x in ci_med],
            "mean_width_days": (cal["hi_h"] - cal["lo_h"]) / 24,
            "below_lo": int((vc < cal["lo_h"]).sum()), "above_hi": int((vc > cal["hi_h"]).sum()),
            "censored_aware_coverage_bounds_all": [float(cov.sum() / n_all),
                float((cov.sum() + len(vo) - surely_miss) / n_all)],  # fmt: skip
            "open_with_elapsed_above_hi": surely_miss,
        },
    }  # fmt: skip
    (ROOT / "reports/k8s_constant_35d_calibration.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(out["served"]), out["validation"])


if __name__ == "__main__":
    main(Path(sys.argv[1]))
