from __future__ import annotations

"""Offline before/after of serving the naive median as the vscode point estimate (ADR-0064).

Drives the REAL TriageAssistant._collect_signals() (real predictor, real train_df, real
engineer_features and prompt builder) over the reconstructed 616-row microsoft/vscode window,
with a no-op classifier/retriever (neither feeds the resolution stage for vscode) and no LLM
call of any kind. "Before" is the raw model output the assistant also exposes
(signals["model_*"]); "after" is what it serves (signals["pred_days"/"lo_days"/"hi_days"]).

Also evaluates the candidate served intervals (ADR-0064 Decision) with the same chronological
calibration / held-out split scripts/10_calibrate_cqr.py uses (30% and 40% of the window by
created_at, eval-set rows excluded from calibration).

Usage (repo root):
    python scripts/eval_vscode_naive_serving.py --frame <vscode_test_frame.parquet> \
        --out reports/vscode_naive_serving_eval.json

Zero network, zero LLM spend. The frame is NOT committed (derived from the gitignored issues
parquet); its sha256 and window definition are recorded in the output for provenance.
"""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore")

from triage_iq.models.resolution import (  # noqa: E402
    BUCKET_LABELS,
    ResolutionTimePredictor,
    hours_to_bucket,
)
from triage_iq.models.triage import TriageAssistant  # noqa: E402

REPO = "microsoft/vscode"
SLUG = "microsoft_vscode"
TARGET = 0.80
SEED = 42


class _NoopClassifier:
    def predict_proba_calibrated(self, texts):  # noqa: ANN001, ANN201
        return np.array([[0.6, 0.4]])

    def classes_(self):  # noqa: ANN201
        return ["a", "b"]


class _NoopRetriever:
    def retrieve(self, text, k=5, exclude_number=None):  # noqa: ANN001, ANN201
        return []


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(p, 4), round(float(centre - half), 4), round(float(centre + half), 4)]


def _boot_mean(vals: np.ndarray, n_boot: int = 1000) -> list[float]:
    rng = np.random.default_rng(SEED)
    means = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [round(float(vals.mean()), 4), round(float(lo), 4), round(float(hi), 4)]


def _conformal_q(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> float:
    """CQR scalar for intervals [lo, hi] (same rule as ResolutionTimePredictor.calibrate_cqr)."""
    e = np.maximum(lo - y, y - hi)
    n = len(e)
    level = min(np.ceil((n + 1) * TARGET) / n, 1.0)
    return float(np.quantile(e, level))


def _score(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> dict:
    inside = (lo <= y) & (y <= hi)
    width = hi - lo
    alpha = 1 - TARGET
    interval_score = width + (2 / alpha) * np.clip(lo - y, 0, None) + (2 / alpha) * np.clip(y - hi, 0, None)
    return {
        "n": int(len(y)),
        "coverage_wilson95": _wilson(int(inside.sum()), len(y)),
        "median_width_d": round(float(np.median(width)), 3),
        "mean_interval_score_d": round(float(interval_score.mean()), 3),
        "lower_miss_frac": round(float((y < lo).mean()), 4),
        "upper_miss_frac": round(float((y > hi).mean()), 4),
    }


def candidates(
    cal: dict[str, np.ndarray], hold: dict[str, np.ndarray], train_days: np.ndarray, naive: float
) -> dict[str, dict]:
    """Score A / B / C on the held-out rows. `cal`/`hold` hold arrays y, lo, hi, p (model, days)."""
    q10, q90 = (float(v) for v in np.quantile(train_days, [0.10, 0.90]))
    out: dict[str, dict] = {}
    q_prod_d = float(cal["q_prod_d"][0])

    # A: current served interval: model Q10/Q90 + stored (production) conformal Q.
    lo = np.clip(hold["lo"] - q_prod_d, 0, None)
    hi = hold["hi"] + q_prod_d
    out["A_model_quantiles_stored_Q"] = _score(lo, hi, hold["y"])
    qa = _conformal_q(cal["lo"], cal["hi"], cal["y"])
    out["A_model_quantiles_fresh_Q"] = _score(
        np.clip(hold["lo"] - qa, 0, None), hold["hi"] + qa, hold["y"]
    ) | {"Q_d": round(qa, 4)}

    # B: empirical train quantiles (constant interval), raw and split-conformal adjusted.
    n = len(hold["y"])
    out["B0_train_quantiles_raw"] = _score(np.full(n, q10), np.full(n, q90), hold["y"]) | {
        "q10_d": round(q10, 4),
        "q90_d": round(q90, 4),
    }
    qb = _conformal_q(np.full(len(cal["y"]), q10), np.full(len(cal["y"]), q90), cal["y"])
    out["B1_train_quantiles_conformal"] = _score(
        np.clip(np.full(n, q10) - qb, 0, None), np.full(n, q90) + qb, hold["y"]
    ) | {"Q_d": round(qb, 4)}

    # C: centred on the naive median, model's relative width (hi/p, lo/p scaled by naive).
    eps = 1e-3
    def rel(d: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        p = np.clip(d["p"], eps, None)
        return naive * d["lo"] / p, naive * d["hi"] / p

    lo_c, hi_c = rel(hold)
    out["C0_naive_scaled_model_width_raw"] = _score(lo_c, hi_c, hold["y"])
    lc, hc = rel(cal)
    qc = _conformal_q(lc, hc, cal["y"])
    out["C1_naive_scaled_model_width_conformal"] = _score(
        np.clip(lo_c - qc, 0, None), hi_c + qc, hold["y"]
    ) | {"Q_d": round(qc, 4)}
    # C2 = what is served: C with the stored production Q (no new calibration artifact).
    out["C2_naive_scaled_model_width_stored_Q"] = _score(
        np.clip(lo_c - q_prod_d, 0, None), hi_c + q_prod_d, hold["y"]
    ) | {"Q_d": round(q_prod_d, 4)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    models = ROOT / "data" / "models"
    spec = importlib.util.spec_from_file_location("cqr10", ROOT / "scripts" / "10_calibrate_cqr.py")
    assert spec and spec.loader
    cqr = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cqr)

    pred = ResolutionTimePredictor.load(str(models / f"resolution_predictor_{SLUG}.pkl"))
    train = pd.read_parquet(ROOT / "data" / "processed" / f"{SLUG}_temporal_train.parquet")
    frame = pd.read_parquet(args.frame)
    assistant = TriageAssistant(
        repo=REPO, classifier=_NoopClassifier(), detector=_NoopRetriever(), predictor=pred,
        train_df=train, groq_api_key="offline-no-llm",
    )

    rows = []
    for _, r in frame.iterrows():
        # The fields app.py /triage builds for the issue Series.
        issue = pd.Series({
            "number": r["number"], "title": r["title"], "body_clean": r["body_clean"],
            "created_at": r["created_at"],
        })
        s = assistant._collect_signals(issue)
        rows.append({
            "number": int(r["number"]), "created_at": r["created_at"],
            "y_d": float(r["resolution_hours"]) / 24.0,
            "served_p": s["pred_days"], "served_lo": s["lo_days"], "served_hi": s["hi_days"],
            "point_source": s["resolution_point_source"],
            "model_p": s["model_point_days"], "model_lo": s["model_lo_days"],
            "model_hi": s["model_hi_days"], "bucket": s["resolution_bucket"],
            "bucket_conf": s["resolution_conf_pct"],
        })
    d = pd.DataFrame(rows)
    y = d["y_d"].to_numpy()
    naive = float(train["resolution_hours"].median()) / 24.0
    adj = json.loads((models / "cqr_conformal_adjustments.json").read_text())["repos"][REPO]["40_60"]
    q_prod_d = adj["q_adjustment_hours"] / 24.0

    result: dict = {
        "provenance": {
            "script": "scripts/eval_vscode_naive_serving.py",
            "git_head": subprocess.run(  # noqa: S603
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT  # noqa: S607
            ).stdout.strip(),
            "frame_sha256": _sha256(args.frame),
            "frame_rows": len(frame),
            "frame_window": [str(frame["created_at"].min()), str(frame["created_at"].max())],
            "train_parquet_sha256": _sha256(ROOT / "data" / "processed" / f"{SLUG}_temporal_train.parquet"),
            "predictor_pkl_sha256": _sha256(models / f"resolution_predictor_{SLUG}.pkl"),
            "cqr_json_sha256": _sha256(models / "cqr_conformal_adjustments.json"),
            "seed": SEED,
            "llm_calls": 0,
        },
        "naive_days": naive,
        "stored_Q_hours_40_60": adj["q_adjustment_hours"],
        "test_median_days": float(np.median(y)),
        "train_median_days": naive,
        "frac_test_in_hours_bucket": float((hours_to_bucket(y * 24.0) == 0).mean()),
    }

    # --- point: before (raw model) vs after (served) -------------------------------------
    ae_naive = np.abs(naive - y)
    ae_model = np.abs(d["model_p"].to_numpy() - y)
    ae_served = np.abs(d["served_p"].to_numpy() - y)
    result["point"] = {
        "n": len(d),
        "served_point_sources": d["point_source"].value_counts().to_dict(),
        "mae_naive_d": round(float(ae_naive.mean()), 4),
        "mae_before_model_d": round(float(ae_model.mean()), 4),
        "mae_after_served_d": round(float(ae_served.mean()), 4),
        "improve_before_vs_naive_pct": round(100 * float((ae_naive.mean() - ae_model.mean()) / ae_naive.mean()), 2),
        "improve_after_vs_naive_pct": round(100 * float((ae_naive.mean() - ae_served.mean()) / ae_naive.mean()), 2),
        "paired_gain_before_vs_naive_d_boot1000": _boot_mean(ae_naive - ae_model),
        "paired_gain_after_vs_naive_d_boot1000": _boot_mean(ae_naive - ae_served),
        "pred_median_before_d": round(float(np.median(d["model_p"])), 4),
        "pred_after_unique": sorted({round(v, 6) for v in d["served_p"]}),
    }
    maj = max(pred.bucket_train_distribution, key=pred.bucket_train_distribution.get)  # type: ignore[arg-type]
    tb = hours_to_bucket(y * 24.0)
    served_b = np.array([BUCKET_LABELS.index(b) for b in d["bucket"]])
    result["bucket"] = {
        "majority_bucket": maj,
        "acc_served": round(float((served_b == tb).mean()), 4),
        "acc_naive_majority": round(float((tb == BUCKET_LABELS.index(maj)).mean()), 4),
        "served_bucket_values": d["bucket"].value_counts().to_dict(),
    }

    # --- interval: served-as-is chain and candidate comparison ----------------------------
    d = d.sort_values("created_at", kind="stable").reset_index(drop=True)
    eval_nums = cqr.load_eval_numbers_by_repo().get(SLUG, set())
    is_eval = d["number"].astype(int).isin(eval_nums)
    train_days = pd.to_numeric(train["resolution_hours"], errors="coerce").dropna().to_numpy() / 24.0
    result["interval"] = {}
    for frac in (0.30, 0.40):
        cal_df, hold_df, excl = cqr.split_cal_true_test(d, is_eval, frac)

        def pack(x: pd.DataFrame) -> dict[str, np.ndarray]:
            return {
                "y": x["y_d"].to_numpy(), "lo": x["model_lo"].to_numpy(),
                "hi": x["model_hi"].to_numpy(), "p": x["model_p"].to_numpy(),
                "q_prod_d": np.array([q_prod_d]),
            }

        cal, hold = pack(cal_df), pack(hold_df)
        key = f"split_{int(frac * 100)}_{100 - int(frac * 100)}"
        result["interval"][key] = {
            "n_cal": len(cal_df), "n_hold": len(hold_df), "eval_rows_excluded_from_cal": len(excl),
            "hold_median_resolution_d": round(float(np.median(hold["y"])), 4),
            "candidates": candidates(cal, hold, train_days, naive),
        }
    # Chain served today by app.py, assuming the LLM echoes the prompt bounds (not run here).
    lo_s = np.clip(d["served_lo"].to_numpy() - q_prod_d, 0, None)
    hi_s = d["served_hi"].to_numpy() + q_prod_d
    result["served_chain_full_window"] = _score(lo_s, hi_s, d["y_d"].to_numpy())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
