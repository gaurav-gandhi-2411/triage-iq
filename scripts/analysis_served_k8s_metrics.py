"""Served-path k8s resolution metrics + CQR v2 calibration (analysis only, no LLM, no network).

Runs the ACTUAL serving signal path (``TriageAssistant._collect_signals``) over every k8s row of
the ADR-0041 re-split temporal TEST frame, using the models/artifacts loaded by
``ModelStore.load_all`` (the same loader the API uses). The LLM stage is never invoked and every
socket connect / DNS lookup is blocked and recorded; the run aborts if any was attempted.

Outputs:
  reports/served_k8s_rows.parquet          per-row served-path outputs (hours)
  reports/served_k8s_metrics.json          task 3c metrics + provenance
  data/models/cqr_conformal_adjustments_v2.json   task 3d fresh k8s Q (vscode entry carried over)

Usage (from repo root, with data/models + data/processed populated and HF_HUB_OFFLINE=1):
  python scripts/analysis_served_k8s_metrics.py --test-frame <k8s_test_frame.parquet>
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

SEED = 42
N_BOOT = 1000
REPO = "kubernetes/kubernetes"
SLUG = "kubernetes_kubernetes"
CAL_FRAC = 0.30  # same as scripts/10_calibrate_cqr.py for k8s
TARGET_COVERAGE = 0.80


def sha256_file(p: Path) -> str:
    """Return the hex sha256 of a file."""
    return hashlib.sha256(p.read_bytes()).hexdigest()


def block_network() -> list[str]:
    """Make every socket connect / DNS lookup raise, recording attempts. Returns the live log."""
    attempts: list[str] = []

    def _deny_connect(self: socket.socket, address: object, *a: object, **k: object) -> None:
        attempts.append(f"connect {address!r}")
        raise OSError("network blocked by analysis_served_k8s_metrics.py")

    def _deny_gai(host: object, *a: object, **k: object) -> None:
        attempts.append(f"getaddrinfo {host!r}")
        raise OSError("network blocked by analysis_served_k8s_metrics.py")

    socket.socket.connect = _deny_connect  # type: ignore[method-assign]
    socket.socket.connect_ex = _deny_connect  # type: ignore[method-assign,assignment]
    socket.getaddrinfo = _deny_gai  # type: ignore[assignment]
    return attempts


def wilson(p: float, n: int) -> list[float]:
    """[p, lo, hi] with the repo's own Wilson CI (scripts/10_calibrate_cqr.py)."""
    lo, hi = _cqr10().wilson_ci(float(p), n)
    return [round(float(p), 4), round(lo, 4), round(hi, 4)]


_CQR10 = None


def _cqr10():  # noqa: ANN202
    """Import scripts/10_calibrate_cqr.py by path (its name is not a valid module identifier)."""
    global _CQR10
    if _CQR10 is None:
        spec = importlib.util.spec_from_file_location("cqr10", ROOT / "scripts/10_calibrate_cqr.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _CQR10 = mod
    return _CQR10


def boot_mean_ci(vals: np.ndarray) -> list[float]:
    """[mean, lo, hi] percentile bootstrap, N_BOOT resamples, seed 42."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(vals), size=(N_BOOT, len(vals)))
    m = vals[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    return [round(float(vals.mean()), 4), round(float(lo), 4), round(float(hi), 4)]


def collect(test: pd.DataFrame, store_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Run _collect_signals on every row; return per-row outputs + served-path diagnostics."""
    from triage_iq.api.loader import ModelStore

    store = ModelStore.load_all(data_dir=store_dir, groq_api_key="dummy-never-used")
    asst = store.get(REPO).assistant
    pred = asst.predictor

    seen_feats: list[pd.DataFrame] = []
    orig_predict = pred.predict

    def _spy_predict(X: pd.DataFrame) -> np.ndarray:
        seen_feats.append(X.copy())
        return orig_predict(X)

    pred.predict = _spy_predict  # type: ignore[method-assign]
    orig_kwargs = asst._resolution_embedding_kwargs
    kw_nonempty: list[bool] = []

    def _spy_kwargs(query_emb: np.ndarray | None) -> dict:
        out = orig_kwargs(query_emb)
        kw_nonempty.append(bool(out))
        return out

    asst._resolution_embedding_kwargs = _spy_kwargs  # type: ignore[method-assign]

    rows = []
    t0 = time.perf_counter()
    for i, r in enumerate(test.itertuples(index=False)):
        # Exactly the fields app.py:/triage builds (no author, no labels, no timestamps beyond
        # created_at); number is the real issue number (not in the train index: temporal test).
        issue = pd.Series(
            {
                "number": int(r.number),
                "title": r.title,
                "body_clean": r.body_clean,
                "created_at": r.created_at,
            }
        )
        s = asst._collect_signals(issue)
        rows.append(
            {
                "number": int(r.number),
                "true_hours": float(r.resolution_hours),
                "pred_hours": s["pred_days"] * 24.0,
                "lo_hours": s["lo_days"] * 24.0,
                "hi_hours": s["hi_days"] * 24.0,
                "served_bucket": s["resolution_bucket"],
            }
        )
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(test)}  {time.perf_counter() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)

    feats = pd.concat(seen_feats, ignore_index=True)
    emb_cols = [c for c in feats.columns if c.startswith("emb_")]
    fallback = (df.pred_hours == 7.0 * 24) & (df.lo_hours == 24.0) & (df.hi_hours == 30.0 * 24)
    diag = {
        "n_rows": len(df),
        "n_predict_calls": len(seen_feats),
        "n_embedding_kwargs_nonempty": int(sum(kw_nonempty)),
        "n_predictor_fallback_rows": int(fallback.sum()),
        "emb_columns": len(emb_cols),
        "emb_rows_any_nonzero_frac": round(float((feats[emb_cols].abs().sum(axis=1) > 0).mean()), 6),
        "emb_cells_nonzero_frac": round(float((feats[emb_cols].to_numpy() != 0).mean()), 6),
        "collect_seconds": round(time.perf_counter() - t0, 1),
    }
    diag["conformal_adjustments_loaded"] = store.conformal_adjustments
    diag["bucket_classifier_model_present"] = pred.model_bucket is not None
    return df, diag


class _IntervalStub:
    """Lets ResolutionTimePredictor.calibrate_cqr run unchanged on already-served intervals."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray) -> None:
        self._lo, self._hi = lo, hi

    def predict_intervals(self, _X: object) -> tuple[np.ndarray, np.ndarray]:
        return self._lo, self._hi


def main() -> None:
    """Collect served-path outputs, compute 3c metrics and 3d v2 calibration."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frame", required=True, type=Path)
    ap.add_argument("--rows-cache", type=Path, default=ROOT / "reports/served_k8s_rows.parquet")
    ap.add_argument("--reuse-rows", action="store_true", help="skip collection, load rows-cache")
    args = ap.parse_args()

    attempts = block_network()
    from triage_iq.models.resolution import (
        BUCKET_LABELS,
        ResolutionTimePredictor,
        hours_to_bucket,
    )

    models = ROOT / "data/models"
    test = pd.read_parquet(args.test_frame)
    test = test[test["resolution_hours"] > 0].copy()
    test = test.sort_values("created_at").reset_index(drop=True)  # as 10_calibrate_cqr.py
    train = pd.read_parquet(ROOT / f"data/processed/{SLUG}_temporal_train.parquet")
    train = train[train["resolution_hours"] > 0]
    n = len(test)
    print(f"test rows: {n}", flush=True)

    if args.reuse_rows:
        rows = pd.read_parquet(args.rows_cache)
        diag = json.loads((ROOT / "reports/served_k8s_metrics.json").read_text())["served_path_diagnostics"]
    else:
        rows, diag = collect(test, ROOT / "data")
        rows.to_parquet(args.rows_cache, index=False)
    assert len(rows) == n and (rows.number.values == test.number.values).all()
    if attempts:
        raise SystemExit(f"NETWORK ATTEMPTS DETECTED (analysis invalid): {attempts}")

    y = rows.true_hours.to_numpy()
    p = rows.pred_hours.to_numpy()
    lo = rows.lo_hours.to_numpy()
    hi = rows.hi_hours.to_numpy()
    pred = ResolutionTimePredictor.load(str(models / f"resolution_predictor_{SLUG}.pkl"))

    naive_h = float(train["resolution_hours"].median())
    ae = np.abs(p - y)
    ae_naive = np.abs(naive_h - y)
    tb = hours_to_bucket(y)
    sb = np.array([BUCKET_LABELS.index(b) for b in rows.served_bucket])
    maj = max(pred.bucket_train_distribution, key=pred.bucket_train_distribution.get)
    maj_acc = float((tb == BUCKET_LABELS.index(maj)).mean())

    stale = json.loads((models / "cqr_conformal_adjustments.json").read_text())["repos"][REPO]
    q_stale = float(stale["q_adjustment_hours"])
    # served conformal interval == app.py: lower = max(0, lo - q), upper = hi + q
    cov_stale = float(((np.clip(lo - q_stale, 0, None) <= y) & (y <= hi + q_stale)).mean())
    cov_raw = float(((lo <= y) & (y <= hi)).mean())
    bucket_acc = float((sb == tb).mean())

    metrics = {
        "n": n,
        "naive_train_median_hours": round(naive_h, 4),
        "naive_train_median_days": round(naive_h / 24, 4),
        "served_mae_hours": boot_mean_ci(ae),
        "served_mae_days": [round(v / 24, 4) for v in boot_mean_ci(ae)],
        "served_median_ae_hours": round(float(np.median(ae)), 4),
        "served_median_ae_days": round(float(np.median(ae)) / 24, 4),
        "naive_mae_hours": boot_mean_ci(ae_naive),
        "naive_mae_days": [round(v / 24, 4) for v in boot_mean_ci(ae_naive)],
        "naive_median_ae_days": round(float(np.median(ae_naive)) / 24, 4),
        "mae_gain_vs_naive_days_boot": [round(v / 24, 4) for v in boot_mean_ci(ae_naive - ae)],
        "mae_improve_pct_vs_naive": round(
            100 * float((ae_naive.mean() - ae.mean()) / ae_naive.mean()), 3
        ),
        "served_bucket_accuracy_wilson": wilson(bucket_acc, n),
        "served_bucket_is_constant": bool(len(set(rows.served_bucket)) == 1),
        "served_bucket_distribution": rows.served_bucket.value_counts().to_dict(),
        "naive_majority_bucket": maj,
        "naive_majority_bucket_accuracy_wilson": wilson(maj_acc, n),
        "bucket_delta_vs_naive_pp_boot": [
            round(100 * v, 3) for v in boot_mean_ci((sb == tb).astype(float) - (tb == BUCKET_LABELS.index(maj)))
        ],
        "q_stale_hours": q_stale,
        "coverage_cqr_stale_served_wilson": wilson(cov_stale, n),
        "coverage_raw_interval_wilson": wilson(cov_raw, n),
        "nominal_coverage": TARGET_COVERAGE,
    }

    # ---- 3d: fresh CQR on the served-path calibration split ----
    cqr10 = _cqr10()
    eval_nums = cqr10.load_eval_numbers_by_repo().get(SLUG, set())
    is_eval = test["number"].astype(int).isin(eval_nums)
    cal_df, true_df, excluded = cqr10.split_cal_true_test(test, is_eval, CAL_FRAC)
    ci, ti = cal_df.index.to_numpy(), true_df.index.to_numpy()
    stub = _IntervalStub(lo[ci], hi[ci])
    adj = ResolutionTimePredictor.calibrate_cqr(stub, None, y[ci], TARGET_COVERAGE)  # type: ignore[arg-type]
    q_new = adj.q_adjustment_hours

    def cover(sel: np.ndarray, q: float) -> float:
        return float(((np.clip(lo[sel] - q, 0, None) <= y[sel]) & (y[sel] <= hi[sel] + q)).mean())

    nt = len(ti)
    cov_new_true = cover(ti, q_new)
    cov_new_all = cover(np.arange(n), q_new)
    k8s_entry = {
        "split": "30_70",
        "n_calibration": adj.n_calibration,
        "n_true_test": nt,
        "q_adjustment_hours": round(q_new, 4),
        "q_adjustment_days": round(q_new / 24, 4),
        "empirical_test_coverage": round(cov_new_true, 4),
        "coverage_ci95_lower": wilson(cov_new_true, nt)[1],
        "coverage_ci95_upper": wilson(cov_new_true, nt)[2],
        "raw_interval_coverage": round(cover(ti, 0.0), 4),
        "median_width_raw_hours": round(float(np.median(hi[ti] - lo[ti])), 4),
        "median_width_conformal_hours": round(
            float(np.median(np.clip(hi[ti] + q_new, 0, None) - np.clip(lo[ti] - q_new, 0, None))), 4
        ),
    }
    k8s_entry["median_width_raw_days"] = round(k8s_entry["median_width_raw_hours"] / 24, 4)
    k8s_entry["median_width_conformal_days"] = round(k8s_entry["median_width_conformal_hours"] / 24, 4)

    old = json.loads((models / "cqr_conformal_adjustments.json").read_text())
    v2 = {"method": old["method"], "target_coverage": old["target_coverage"], "repos": {}}
    v2["repos"][REPO] = k8s_entry
    v2["repos"]["microsoft/vscode"] = old["repos"]["microsoft/vscode"]
    v2_text = json.dumps(v2, indent=2)  # same writer settings as scripts/10_calibrate_cqr.py
    v2_path = models / "cqr_conformal_adjustments_v2.json"
    v2_path.write_text(v2_text, encoding="utf-8")
    old_text = (models / "cqr_conformal_adjustments.json").read_text(encoding="utf-8")
    # vscode is the last repo entry in both files, so its raw text is the tail from its key on.
    marker = '    "microsoft/vscode": {'
    vscode_byte_identical = (
        old_text.count(marker) == 1
        and v2_text.count(marker) == 1
        and old_text[old_text.index(marker) :] == v2_text[v2_text.index(marker) :]
    )

    holdout = {
        "n_cal": adj.n_calibration,
        "n_hold": nt,
        "eval_rows_excluded_from_cal": len(excluded),
        "q_fresh_hours": round(q_new, 4),
        "q_stale_hours": q_stale,
        "hold_raw_wilson": wilson(cover(ti, 0.0), nt),
        "hold_stale_q_wilson": wilson(cover(ti, q_stale), nt),
        "hold_v2_q_wilson": wilson(cov_new_true, nt),
        "all_rows_v2_q_wilson": wilson(cov_new_all, n),
        "nominal": TARGET_COVERAGE,
    }

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()
    art = {
        str(q.relative_to(ROOT)).replace("\\", "/"): sha256_file(q)
        for q in [
            models / f"resolution_predictor_{SLUG}.pkl",
            models / f"similar_issue_index_{SLUG}_bge/index.faiss",
            models / f"similar_issue_index_{SLUG}_bge/meta.pkl",
            models / "cqr_conformal_adjustments.json",
            models / "cqr_conformal_adjustments_v2.json",
            ROOT / f"data/processed/{SLUG}_temporal_train.parquet",
        ]
    }
    art["test_frame:" + args.test_frame.name] = sha256_file(args.test_frame)
    out = {
        "git_sha_at_run": sha,
        "seed": SEED,
        "bootstrap_resamples": N_BOOT,
        "n": n,
        "artifact_sha256": art,
        "network_attempts": attempts,
        "served_path_diagnostics": diag,
        "served_metrics": metrics,
        "cqr_v2": {
            "holdout": holdout,
            "vscode_entry_byte_identical": vscode_byte_identical,
            "v2_file_sha256": sha256_file(v2_path),
        },
    }
    (ROOT / "reports/served_k8s_metrics.json").write_text(
        json.dumps(out, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"served_metrics": metrics, "cqr_v2": out["cqr_v2"], "diag": diag}, indent=2))


if __name__ == "__main__":
    main()
