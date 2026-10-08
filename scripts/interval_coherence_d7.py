from __future__ import annotations

"""Interval coherence under D7 (ADR-0064): vscode served-interval coverage, k8s point clamp.

Reads the per-row served outputs written by driving the real TriageAssistant._collect_signals
over the two test windows (no LLM, no network); see scripts/eval_vscode_naive_serving.py and the
D7 before/after report. Pure arithmetic on those rows:

* vscode (a): coverage of the SERVED interval (model relative width re-centred on the train
  median, then +/- the stored CQR Q exactly as api/app.py attaches it) against the 80 pct target,
  with Wilson CIs, full 616-row window and chronological last-60 pct hold-out, and width
  before (model-centred) vs after (re-centred).
* k8s (b): rows whose served train median lies outside the unchanged model interval, and the
  effect of clamping the point onto the interval on MAE / median absolute error.

Usage: python scripts/interval_coherence_d7.py --vscode ROWS.parquet --k8s ROWS.parquet \
    --out reports/interval_coherence_d7.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

VSCODE_Q_HOURS = 1.2655  # stored CQR Q served for microsoft/vscode (40/60 split), v1 == v2 entry
TARGET = 0.80


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(p, 4), round(c - h, 4), round(c + h, 4)]


def _cov(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    inside = (y >= lo) & (y <= hi)
    return {
        "n": int(len(y)),
        "coverage_wilson95": wilson(int(inside.sum()), len(y)),
        "median_width_d": round(float(np.median((hi - lo) / 24.0)), 3),
        "lower_miss_frac": round(float((y < lo).mean()), 4),
        "upper_miss_frac": round(float((y > hi).mean()), 4),
    }


def vscode_block(df: pd.DataFrame) -> dict:
    df = df.sort_values("number").reset_index(drop=True)  # issue number is monotone in time
    y = df.true_hours.to_numpy()
    q = VSCODE_Q_HOURS
    served_lo = np.maximum(0.0, df.lo_hours.to_numpy() - q)
    served_hi = df.hi_hours.to_numpy() + q
    model_lo = np.maximum(0.0, df.model_lo_hours.to_numpy() - q)
    model_hi = df.model_hi_hours.to_numpy() + q
    hold = slice(int(len(df) * 0.4), None)
    out = {}
    for name, (lo, hi) in {
        "served_recentred_plus_Q": (served_lo, served_hi),
        "served_recentred_raw": (df.lo_hours.to_numpy(), df.hi_hours.to_numpy()),
        "before_model_centred_plus_Q": (model_lo, model_hi),
    }.items():
        out[name] = {
            "full_window": _cov(y, lo, hi),
            "holdout_last_60pct": _cov(y[hold], lo[hold], hi[hold]),
        }
    served = out["served_recentred_plus_Q"]["full_window"]["coverage_wilson95"]
    out["verdict"] = {
        "target": TARGET,
        "served_ci_contains_target": served[1] <= TARGET <= served[2],
        "served_point_meets_target": served[0] >= TARGET,
    }
    return out


def k8s_block(df: pd.DataFrame) -> dict:
    y = df.true_hours.to_numpy()
    lo, hi = df.lo_hours.to_numpy(), df.hi_hours.to_numpy()
    point = df.pred_hours.to_numpy()
    clamped = np.clip(point, lo, hi)
    outside = (point < lo) | (point > hi)

    def stats(p: np.ndarray) -> dict:
        ae = np.abs(p - y) / 24.0
        return {"mae_d": round(float(ae.mean()), 4), "median_ae_d": round(float(np.median(ae)), 4)}

    return {
        "n": int(len(df)),
        "n_point_outside_interval": int(outside.sum()),
        "pct_outside": round(100 * float(outside.mean()), 3),
        "n_below_lo": int((point < lo).sum()),
        "n_above_hi": int((point > hi).sum()),
        "unclamped": stats(point),
        "clamped": stats(clamped),
        "changed_rows_abs_shift_d": [
            round(float(v), 4) for v in np.abs(clamped - point)[outside] / 24.0
        ],
        "interval_untouched": True,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vscode", required=True, type=Path)
    ap.add_argument("--k8s", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    res = {
        "provenance": {
            "script": "scripts/interval_coherence_d7.py",
            "vscode_rows": a.vscode.name,
            "k8s_rows": a.k8s.name,
            "vscode_Q_hours": VSCODE_Q_HOURS,
            "llm_calls": 0,
        },
        "vscode": vscode_block(pd.read_parquet(a.vscode)),
        "k8s": k8s_block(pd.read_parquet(a.k8s)),
    }
    a.out.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
