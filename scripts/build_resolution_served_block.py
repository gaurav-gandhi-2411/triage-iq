"""Build reports/eval_summary.json["resolution_served"] from its source reports (ADR-0064).

The /eval page's resolution table used to read ``leakage.honest_metrics``, which describes the
superseded 2026-05-30 k8s model (104.05 d vs 106.29 d, n=1498) and a point estimate that is no
longer served. This block describes what production serves: the training-window median as the
point, the bucket classifier (k8s) and the CQR interval. Every number is copied from a committed
source report so ``tests/test_eval_summary_drift.py`` can recompute the block and fail on drift.

Sources:
  reports/served_k8s_metrics.json         k8s served-path metrics (30e6fd8, 2,992 test rows)
  reports/vscode_naive_serving_eval.json  vscode 616-row reconstructed window
  reports/d7_resolution_before_after.json offline before/after of the D7 change (both repos)
  data/models/MANIFEST.sha256             artifact hashes the numbers were produced against

Usage: python scripts/build_resolution_served_block.py [--write]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
SUMMARY = REPORTS / "eval_summary.json"

K8S = "kubernetes/kubernetes"
VSCODE = "microsoft/vscode"


def _manifest() -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (ROOT / "data/models/MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, path = line.strip().split("  ", 1)
            out[path] = digest
    return out


def _load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def build() -> dict:
    """Return the resolution_served block computed from the source reports."""
    k = _load("served_k8s_metrics.json")
    km = k["served_metrics"]
    v = _load("vscode_naive_serving_eval.json")
    d7 = _load("d7_resolution_before_after.json")
    man = _manifest()
    k8s_pred = "data/models/resolution_predictor_kubernetes_kubernetes.pkl"
    vs_pred = "data/models/resolution_predictor_microsoft_vscode.pkl"
    cqr = "data/models/cqr_conformal_adjustments_v2.json"
    c2 = v["interval"]["split_40_60"]["candidates"]["C2_naive_scaled_model_width_stored_Q"]
    return {
        "point_estimate": {
            "served": "train_median",
            "decision": "owner decision D7, 2026-10-08 (ADR-0064)",
            "definition": "median(resolution_hours of the training set) / 24, the same naive "
            "baseline the published comparison uses",
            "why": "The learned point estimate has a worse median absolute error than the "
            "training-window median in both repos; it is not served. The learned parts that stay "
            "are the k8s bucket classifier and the conformal intervals.",
        },
        "provenance": {
            "k8s_metrics": "reports/served_k8s_metrics.json at " + k["git_sha_at_run"][:7],
            "vscode_metrics": "reports/vscode_naive_serving_eval.json",
            "before_after": "reports/d7_resolution_before_after.json",
            "manifest_sha256": {
                k8s_pred: man[k8s_pred],
                vs_pred: man[vs_pred],
                cqr: man[cqr],
            },
        },
        "repos": {
            K8S: {
                "window": "ADR-0041 re-split temporal test, served path through _collect_signals",
                "n_test": km["n"],
                "served_point_days": km["naive_train_median_days"],
                "served": {
                    "mae_days": km["naive_mae_days"][0],
                    "median_ae_days": km["naive_median_ae_days"],
                },
                "learned_model_not_served": {
                    "mae_days": km["served_mae_days"][0],
                    "mae_ci95_days": km["served_mae_days"][1:],
                    "median_ae_days": km["served_median_ae_days"],
                    "mae_gain_vs_naive_days": km["mae_gain_vs_naive_days_boot"][0],
                    "mae_gain_ci95_days": km["mae_gain_vs_naive_days_boot"][1:],
                },
                "bucket": {
                    "served": "classifier",
                    "accuracy": km["served_bucket_accuracy_wilson"][0],
                    "accuracy_ci95": km["served_bucket_accuracy_wilson"][1:],
                    "naive_majority_accuracy": km["naive_majority_bucket_accuracy_wilson"][0],
                    "delta_pp": km["bucket_delta_vs_naive_pp_boot"][0],
                    "delta_ci95_pp": km["bucket_delta_vs_naive_pp_boot"][1:],
                },
                "interval": {
                    "served": "model Q10/Q90 + CQR v2",
                    "nominal_coverage": km["nominal_coverage"],
                    "coverage_raw_full_test": km["coverage_raw_interval_wilson"][0],
                    "coverage_cqr_v2_heldout": k["cqr_v2"]["holdout"]["hold_v2_q_wilson"][0],
                    "coverage_cqr_v2_heldout_ci95": k["cqr_v2"]["holdout"]["hold_v2_q_wilson"][1:],
                    "n_heldout": k["cqr_v2"]["holdout"]["n_hold"],
                },
                "unchanged_by_d7": {
                    "interval_lo_hi_bit_identical": d7[K8S]["lo_hi_bit_identical"],
                    "bucket_identical": d7[K8S]["bucket_identical"],
                },
            },
            VSCODE: {
                "window": "reconstructed last 616 closed issues, 2026-04-21..27",
                "n_test": v["point"]["n"],
                "served_point_days": round(v["train_median_days"], 4),
                "served": {
                    "mae_days": v["point"]["mae_after_served_d"],
                    "median_ae_days": round(d7[VSCODE]["served_after_median_ae_days"], 4),
                },
                "learned_model_not_served": {
                    "mae_days": v["point"]["mae_before_model_d"],
                    "median_ae_days": round(d7[VSCODE]["model_before_median_ae_days"], 4),
                    "mae_gain_vs_naive_days": v["point"]["paired_gain_before_vs_naive_d_boot1000"][0],
                    "mae_gain_ci95_days": v["point"]["paired_gain_before_vs_naive_d_boot1000"][1:],
                },
                "bucket": {
                    "served": "naive majority prior (classifier not trusted, ADR-0025)",
                    "accuracy": v["bucket"]["acc_served"],
                    "naive_majority_accuracy": v["bucket"]["acc_naive_majority"],
                    "delta_pp": 0.0,
                },
                "interval": {
                    "served": "model relative width re-centred on the median + stored CQR Q",
                    "nominal_coverage": 0.8,
                    "coverage_heldout": c2["coverage_wilson95"][0],
                    "coverage_heldout_ci95": c2["coverage_wilson95"][1:],
                    "n_heldout": v["interval"]["split_40_60"]["n_hold"],
                    "median_width_days": c2["median_width_d"],
                    "caveat": "single 7-day window; train median 3.84 d vs test median "
                    "0.049 d; coverage is a measurement, not a guarantee",
                },
            },
        },
    }


def main() -> None:
    """Print the block, or write it into eval_summary.json with --write."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    block = build()
    if not args.write:
        print(json.dumps(block, indent=2))
        return
    text = SUMMARY.read_text(encoding="utf-8")
    nl = "\r\n" if "\r\n" in text else "\n"
    data = json.loads(text)
    data["resolution_served"] = block
    # Additive: the UI keeps reading leakage.honest_metrics (renaming it blanked /eval once), so it
    # stays, labelled as history instead of as the deployed model.
    data["leakage"]["honest_metrics_status"] = (
        "historical: the 2026-05-30 retrain scored on the created_at split (k8s n=1498, vscode n=616). "
        "NOT what production serves today; see resolution_served (ADR-0064)."
    )
    data["_sources"]["resolution_served"] = (
        "scripts/build_resolution_served_block.py from reports/served_k8s_metrics.json + "
        "reports/vscode_naive_serving_eval.json + reports/d7_resolution_before_after.json + "
        "data/models/MANIFEST.sha256 + docs/architecture/adr/0064"
    )
    SUMMARY.write_text(
        json.dumps(data, indent=2, ensure_ascii=False).replace("\n", nl) + nl, encoding="utf-8"
    )
    print("wrote", SUMMARY)


if __name__ == "__main__":
    main()
