"""Conformalized Quantile Regression (CQR) calibration script.

Uses the held-out temporal test set for each repo to compute the scalar CQR
adjustment Q (Romano et al. 2019) at 80% target coverage.

Calibration source: TEST split only — the data the frozen LightGBM model has
never seen (not used for training, early-stopping, or hyperparameter tuning).

Split strategy (temporal, no shuffling):
  kubernetes/kubernetes : 30/70 — cal = first 30% by created_at, true_test = last 70%
  microsoft/vscode      : 30/70 AND 40/60 splits (both reported)

Outputs:
  data/models/cqr_conformal_adjustments.json  — machine-readable artifact
  stdout report                                — human-readable summary
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import faiss
import joblib
import numpy as np
import pandas as pd

from triage_iq.models.resolution import ResolutionTimePredictor, engineer_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
OUTPUT_PATH = MODELS_DIR / "cqr_conformal_adjustments.json"


# ---------------------------------------------------------------------------
# Embedding loading — copied verbatim from scripts/09_train_resolution.py
# ---------------------------------------------------------------------------

def load_embeddings_from_index(repo: str, df: pd.DataFrame) -> np.ndarray | None:
    """Reconstruct BGE embeddings from the Day-5 FAISS index for issues in df."""
    index_dir = MODELS_DIR / f"dup_index_{repo}_bge"
    if not (index_dir / "index.faiss").exists():
        log.warning("BGE index not found for %s — skipping embedding features", repo)
        return None
    meta = joblib.load(str(index_dir / "meta.pkl"))
    index = faiss.read_index(str(index_dir / "index.faiss"))
    num_to_faiss_idx = {int(n): i for i, n in enumerate(meta["issue_numbers"])}
    dim = index.d
    embs = np.zeros((len(df), dim), dtype=np.float32)
    missing = 0
    for row_pos, num in enumerate(df["number"]):
        faiss_idx = num_to_faiss_idx.get(int(num))
        if faiss_idx is not None:
            embs[row_pos] = index.reconstruct(int(faiss_idx))
        else:
            missing += 1
    if missing > 0:
        log.warning("[%s] %d issues not in FAISS index (will use zero embedding)", repo, missing)
    return embs


# ---------------------------------------------------------------------------
# Wilson 95% CI for a proportion
# ---------------------------------------------------------------------------

def wilson_ci(p: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return (lower, upper) 95% Wilson confidence interval for proportion p."""
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(center - half), float(center + half)


# ---------------------------------------------------------------------------
# Single (repo, split_ratio) calibration
# ---------------------------------------------------------------------------

def calibrate_one(
    repo: str,
    cal_df: pd.DataFrame,
    true_test_df: pd.DataFrame,
    train_df: pd.DataFrame,
    predictor: ResolutionTimePredictor,
) -> dict:
    """Run CQR calibration for one repo + one temporal split.

    Args:
        repo: Repo slug (e.g. 'kubernetes_kubernetes').
        cal_df: Calibration rows (earlier portion of test set).
        true_test_df: True-test rows (later portion of test set).
        train_df: Full training set — needed for author-history features.
        predictor: Loaded, frozen ResolutionTimePredictor.

    Returns:
        Dict of metrics for the JSON artifact and stdout report.
    """
    y_cal = cal_df["resolution_hours"].values
    y_true_test = true_test_df["resolution_hours"].values

    # Load embeddings for both blocks
    emb_cal = load_embeddings_from_index(repo, cal_df)
    emb_true_test = load_embeddings_from_index(repo, true_test_df)

    # Feature engineering — use predictor.pca (already fitted; do NOT refit)
    log.info("[%s] Engineering calibration features (%d rows)...", repo, len(cal_df))
    X_cal, _ = engineer_features(
        cal_df, train_df=train_df, embeddings=emb_cal, pca=predictor.pca
    )
    log.info("[%s] Engineering true-test features (%d rows)...", repo, len(true_test_df))
    X_true_test, _ = engineer_features(
        true_test_df, train_df=train_df, embeddings=emb_true_test, pca=predictor.pca
    )

    # 1. CQR calibration
    adj = predictor.calibrate_cqr(X_cal, y_cal, target_coverage=0.80)

    # 2. Conformal coverage on true-test
    conf_lo, conf_hi = predictor.predict_conformal_interval(X_true_test, adj)
    covered = float(((conf_lo <= y_true_test) & (y_true_test <= conf_hi)).mean())

    # 3. Raw (uncalibrated) interval coverage on true-test
    raw_lo, raw_hi = predictor.predict_intervals(X_true_test)
    raw_coverage = float(((raw_lo <= y_true_test) & (y_true_test <= raw_hi)).mean())

    # 4. Median interval widths (hours)
    median_width_raw = float(np.median(raw_hi - raw_lo))
    median_width_conformal = float(np.median(conf_hi - conf_lo))

    # 5. Wilson 95% CI for empirical coverage
    ci_lower, ci_upper = wilson_ci(covered, len(y_true_test))

    Q = adj.q_adjustment_hours
    log.info(
        "[%s] Q=%.2fh (%.2fd)  coverage=%.4f [%.4f, %.4f]  raw_cov=%.4f  "
        "width_raw=%.1fh  width_conf=%.1fh",
        repo, Q, Q / 24, covered, ci_lower, ci_upper, raw_coverage,
        median_width_raw, median_width_conformal,
    )

    return {
        "n_calibration": adj.n_calibration,
        "n_true_test": len(y_true_test),
        "q_adjustment_hours": round(Q, 4),
        "q_adjustment_days": round(Q / 24, 4),
        "empirical_test_coverage": round(covered, 4),
        "coverage_ci95_lower": round(ci_lower, 4),
        "coverage_ci95_upper": round(ci_upper, 4),
        "raw_interval_coverage": round(raw_coverage, 4),
        "median_width_raw_hours": round(median_width_raw, 4),
        "median_width_conformal_hours": round(median_width_conformal, 4),
        "median_width_raw_days": round(median_width_raw / 24, 4),
        "median_width_conformal_days": round(median_width_conformal / 24, 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Run CQR calibration for all repo / split configurations."""

    # ── Load and filter data ─────────────────────────────────────────────────
    results: dict = {}

    repo_configs: list[tuple[str, list[float]]] = [
        ("kubernetes_kubernetes", [0.30]),
        ("microsoft_vscode", [0.30, 0.40]),
    ]

    for repo, split_ratios in repo_configs:
        log.info("=" * 60)
        log.info("Repo: %s", repo)

        train_df = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_train.parquet")
        test_df = pd.read_parquet(PROCESSED_DIR / f"{repo}_temporal_test.parquet")

        # Filter to positive resolution hours (same as training pipeline)
        train_df = train_df[train_df["resolution_hours"] > 0].copy()
        test_df = test_df[test_df["resolution_hours"] > 0].copy()

        # Sort test set by created_at ascending (preserve temporal ordering)
        test_df = test_df.sort_values("created_at").reset_index(drop=True)

        log.info("[%s] train=%d  test=%d", repo, len(train_df), len(test_df))

        predictor = ResolutionTimePredictor.load(
            str(MODELS_DIR / f"resolution_predictor_{repo}.pkl")
        )

        repo_display = repo.replace("_", "/", 1)

        if len(split_ratios) == 1:
            # Single split — store metrics directly under repo key
            cal_frac = split_ratios[0]
            n_cal = int(len(test_df) * cal_frac)
            cal_df = test_df.iloc[:n_cal]
            true_test_df = test_df.iloc[n_cal:]

            log.info(
                "[%s] Split %.0f/%.0f  cal=%d  true_test=%d",
                repo, cal_frac * 100, (1 - cal_frac) * 100, len(cal_df), len(true_test_df),
            )

            metrics = calibrate_one(repo, cal_df, true_test_df, train_df, predictor)
            split_label = f"{int(cal_frac*100)}_{int((1-cal_frac)*100)}"
            results[repo_display] = {"split": split_label, **metrics}
        else:
            # Multiple splits — store under nested split keys
            results[repo_display] = {}
            for cal_frac in split_ratios:
                n_cal = int(len(test_df) * cal_frac)
                cal_df = test_df.iloc[:n_cal]
                true_test_df = test_df.iloc[n_cal:]

                log.info(
                    "[%s] Split %.0f/%.0f  cal=%d  true_test=%d",
                    repo, cal_frac * 100, (1 - cal_frac) * 100, len(cal_df), len(true_test_df),
                )

                metrics = calibrate_one(repo, cal_df, true_test_df, train_df, predictor)
                split_label = f"{int(cal_frac*100)}_{int((1-cal_frac)*100)}"
                results[repo_display][split_label] = metrics

    # ── Write JSON artifact ──────────────────────────────────────────────────
    artifact = {
        "method": "CQR split-conformal, Romano et al. 2019",
        "target_coverage": 0.80,
        "repos": results,
    }
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as fh:
        json.dump(artifact, fh, indent=2)
    log.info("Saved CQR artifact to %s", OUTPUT_PATH)

    # ── Stdout report ────────────────────────────────────────────────────────
    print()
    print("=== CQR Calibration Results ===")
    print()

    def _fmt_row(label: str, m: dict) -> None:
        Q_h = m["q_adjustment_hours"]
        Q_d = m["q_adjustment_days"]
        cov = m["empirical_test_coverage"] * 100
        ci_lo = m["coverage_ci95_lower"] * 100
        ci_hi = m["coverage_ci95_upper"] * 100
        raw_cov = m["raw_interval_coverage"] * 100
        w_raw_d = m["median_width_raw_days"]
        w_conf_d = m["median_width_conformal_days"]
        print(f"{label}")
        print(f"  Q adjustment:          {Q_h:.1f} hours ({Q_d:.1f} days)")
        print(f"  Empirical coverage:    {cov:.1f}% [{ci_lo:.1f}%, {ci_hi:.1f}%] 95% CI  (raw: {raw_cov:.1f}%)")
        print(f"  Median interval width: raw={w_raw_d:.0f}d  conformal={w_conf_d:.0f}d")
        print()

    k8s = results["kubernetes/kubernetes"]
    n_cal_k8s = k8s["n_calibration"]
    n_tt_k8s = k8s["n_true_test"]
    _fmt_row(
        f"kubernetes/kubernetes (split 30/70: {n_cal_k8s} cal / {n_tt_k8s} true-test)",
        k8s,
    )

    vs = results["microsoft/vscode"]
    for split_key in ["30_70", "40_60"]:
        m = vs[split_key]
        n_cal = m["n_calibration"]
        n_tt = m["n_true_test"]
        frac_lo, frac_hi = split_key.split("_")
        _fmt_row(
            f"microsoft/vscode (split {frac_lo}/{frac_hi}: {n_cal} cal / {n_tt} true-test)",
            m,
        )

    print(
        "EXCHANGEABILITY NOTE: vscode train/test gap is ~10 years (train 2015-2016, "
        "test 2026-04-21\nto 2026-04-27). Temporal non-exchangeability is expected. "
        "Report empirical coverage as-is."
    )


if __name__ == "__main__":
    main()
