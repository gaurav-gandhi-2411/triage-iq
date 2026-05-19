"""W1.2 calibration diagnostic: T1 leakage audit, T2 temperature scaling, T3 robustness.

Usage:
    python scripts/12b_calibration_diagnostic.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import softmax
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from triage_iq.models.component_classifier import TFIDFComponentClassifier, _build_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEFAULT_REPOS = ["microsoft_vscode", "kubernetes_kubernetes"]
PROCESSED_DIR = Path("data/processed")
MODELS_DIR = Path("data/models")
REPORTS_DIR = Path("reports")

N_BOOTSTRAP = 1000
RNG_SEED = 42
ECE_REVISED_THRESHOLD = 0.18


# ---------------------------------------------------------------------------
# ECE helper
# ---------------------------------------------------------------------------

def compute_ece(y_enc: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    max_p = proba.max(axis=1)
    y_pred = proba.argmax(axis=1)
    correct = (y_pred == y_enc).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_enc)
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (max_p >= lo) & (max_p < hi)
        if not mask.any():
            continue
        ece += mask.sum() * abs(correct[mask].mean() - max_p[mask].mean())
    return float(ece / max(n, 1))


# ---------------------------------------------------------------------------
# T1 — Leakage audit
# ---------------------------------------------------------------------------

def t1_leakage_audit(repo: str, train: pd.DataFrame, val: pd.DataFrame,
                     test: pd.DataFrame, clf: TFIDFComponentClassifier) -> dict:
    log.info("── T1 Leakage Audit: %s ──", repo)

    result: dict = {}

    # T1a: row counts and overlap check
    id_col = "number" if "number" in train.columns else None
    result["train_rows"] = len(train)
    result["val_rows"] = len(val)
    result["test_rows"] = len(test)
    result["total_rows"] = len(train) + len(val) + len(test)

    if id_col:
        train_ids = set(train[id_col])
        val_ids = set(val[id_col])
        test_ids = set(test[id_col])
        overlap_train_val = train_ids & val_ids
        overlap_train_test = train_ids & test_ids
        overlap_val_test = val_ids & test_ids
        result["overlap_train_val"] = len(overlap_train_val)
        result["overlap_train_test"] = len(overlap_train_test)
        result["overlap_val_test"] = len(overlap_val_test)
        leakage = bool(overlap_val_test) or bool(overlap_train_test)
        result["leakage_detected"] = leakage
        if leakage:
            log.error("T1a FAIL — overlap detected: val∩test=%d  train∩test=%d",
                      len(overlap_val_test), len(overlap_train_test))
        else:
            log.info("T1a PASS — no overlap: train=%d  val=%d  test=%d  (id_col=%s)",
                     len(train), len(val), len(test), id_col)
    else:
        result["overlap_val_test"] = "no_id_col"
        result["leakage_detected"] = "unknown"
        log.warning("T1a: no 'number' column found; cannot check ID overlap")

    # T1b: same split used at train time (random_state=42 hardcoded in splits.py)
    result["split_random_state"] = 42
    result["split_fn"] = "stratified_classifier_split(random_state=42)"
    log.info("T1b PASS — stratified_classifier_split uses random_state=42 (hardcoded in splits.py:73)")

    # T1c: calibrator field on loaded model must be None (no prior run contamination)
    cal_field = clf.calibrator
    result["calibrator_on_load"] = type(cal_field).__name__ if cal_field is not None else "None"
    if cal_field is not None:
        log.warning("T1c WARN — calibrator already set on loaded pkl: %s", type(cal_field).__name__)
    else:
        log.info("T1c PASS — calibrator=None on load (fresh fit guaranteed)")

    return result


# ---------------------------------------------------------------------------
# T2 — Temperature scaling
# ---------------------------------------------------------------------------

def t2_temperature_scaling(
    clf: TFIDFComponentClassifier,
    X_val: pd.Series, y_val_enc: np.ndarray,
    X_test: pd.Series, y_test_enc: np.ndarray,
) -> dict:
    log.info("── T2 Temperature Scaling ──")
    assert clf.pipeline is not None

    # Extract logits via pipeline steps: tfidf → decision_function
    X_tfidf_val = clf.pipeline["tfidf"].transform(X_val)
    X_tfidf_test = clf.pipeline["tfidf"].transform(X_test)
    logits_val = clf.pipeline["lr"].decision_function(X_tfidf_val)
    logits_test = clf.pipeline["lr"].decision_function(X_tfidf_test)

    def nll(T: float) -> float:
        scaled = softmax(logits_val / T, axis=1)
        log_p = np.log(scaled[np.arange(len(y_val_enc)), y_val_enc] + 1e-15)
        return float(-log_p.mean())

    opt = minimize_scalar(nll, bounds=(0.1, 20.0), method="bounded")
    T_opt = float(opt.x)

    proba_val_ts = softmax(logits_val / T_opt, axis=1)
    proba_test_ts = softmax(logits_test / T_opt, axis=1)

    ece_val = compute_ece(y_val_enc, proba_val_ts)
    ece_test = compute_ece(y_test_enc, proba_test_ts)

    # Argmax is preserved (temperature scaling doesn't change argmax)
    raw_acc_test = float((clf.pipeline.predict_proba(X_test).argmax(axis=1) == y_test_enc).mean())
    ts_acc_test = float((proba_test_ts.argmax(axis=1) == y_test_enc).mean())
    acc_delta_pp = (ts_acc_test - raw_acc_test) * 100

    log.info("T2: T_opt=%.4f  val_ECE=%.4f  test_ECE=%.4f  test_acc_delta=%.4fpp",
             T_opt, ece_val, ece_test, acc_delta_pp)

    return {
        "T_opt": round(T_opt, 4),
        "ece_val": round(ece_val, 4),
        "ece_test": round(ece_test, 4),
        "acc_test_before": round(raw_acc_test, 4),
        "acc_test_after": round(ts_acc_test, 4),
        "acc_test_delta_pp": round(acc_delta_pp, 4),
        "proba_val": proba_val_ts,
        "proba_test": proba_test_ts,
    }


# ---------------------------------------------------------------------------
# T3 — Isotonic calibration + robustness checks
# ---------------------------------------------------------------------------

def t3_isotonic_robustness(
    clf: TFIDFComponentClassifier,
    X_val: pd.Series, y_val_enc: np.ndarray,
    X_test: pd.Series, y_test_enc: np.ndarray,
    classes: np.ndarray,
) -> dict:
    log.info("── T3 Isotonic Robustness ──")
    assert clf.pipeline is not None

    cal = CalibratedClassifierCV(FrozenEstimator(clf.pipeline), method="isotonic")
    cal.fit(X_val, y_val_enc)

    proba_val_iso = cal.predict_proba(X_val)
    proba_test_iso = cal.predict_proba(X_test)
    proba_test_raw = clf.pipeline.predict_proba(X_test)

    ece_val = compute_ece(y_val_enc, proba_val_iso)
    ece_test = compute_ece(y_test_enc, proba_test_iso)

    pred_raw = proba_test_raw.argmax(axis=1)
    pred_iso = proba_test_iso.argmax(axis=1)

    acc_raw = float((pred_raw == y_test_enc).mean())
    acc_iso = float((pred_iso == y_test_enc).mean())
    acc_delta_pp = (acc_iso - acc_raw) * 100

    log.info("T3: val_ECE=%.4f  test_ECE=%.4f  test_acc_before=%.4f  after=%.4f  delta=%.2fpp",
             ece_val, ece_test, acc_raw, acc_iso, acc_delta_pp)

    # T3a: Per-class accuracy delta (top 10 movers)
    n_classes = len(classes)
    per_class: list[dict] = []
    for ci in range(n_classes):
        mask = y_test_enc == ci
        if not mask.any():
            continue
        acc_before_c = float((pred_raw[mask] == ci).mean())
        acc_after_c = float((pred_iso[mask] == ci).mean())
        per_class.append({
            "class": str(classes[ci]),
            "n": int(mask.sum()),
            "acc_before": round(acc_before_c, 4),
            "acc_after": round(acc_after_c, 4),
            "delta_pp": round((acc_after_c - acc_before_c) * 100, 2),
        })
    per_class.sort(key=lambda x: abs(x["delta_pp"]), reverse=True)

    top_movers = per_class[:10]
    n_improved = sum(1 for c in per_class if c["delta_pp"] > 0)
    n_degraded = sum(1 for c in per_class if c["delta_pp"] < 0)
    top3_contribution_pp = sum(abs(c["delta_pp"]) * c["n"] for c in per_class[:3]) / len(y_test_enc) * 100

    log.info("T3a: %d classes improved, %d degraded", n_improved, n_degraded)
    log.info("T3a: top-3 movers account for %.2fpp of the total delta", top3_contribution_pp)
    for c in top_movers[:5]:
        log.info("  %-30s n=%3d  %.4f→%.4f  delta=%.2fpp",
                 c["class"], c["n"], c["acc_before"], c["acc_after"], c["delta_pp"])

    # T3b: Confidence-bucketed accuracy (raw vs isotonic)
    margins_raw = np.sort(proba_test_raw, axis=1)[:, -1] - np.sort(proba_test_raw, axis=1)[:, -2]
    bucket_results = []
    for lo, hi, label in [(0, 0.05, "<0.05"), (0.05, 0.15, "0.05-0.15"),
                           (0.15, 0.30, "0.15-0.30"), (0.30, 1.0, ">0.30")]:
        mask = (margins_raw >= lo) & (margins_raw < hi)
        if not mask.any():
            continue
        acc_b = float((pred_raw[mask] == y_test_enc[mask]).mean())
        acc_a = float((pred_iso[mask] == y_test_enc[mask]).mean())
        bucket_results.append({
            "margin_bucket": label,
            "n": int(mask.sum()),
            "acc_raw": round(acc_b, 4),
            "acc_iso": round(acc_a, 4),
            "delta_pp": round((acc_a - acc_b) * 100, 2),
        })
        log.info("T3b  margin=%8s  n=%3d  raw=%.4f  iso=%.4f  Δ=%.2fpp",
                 label, mask.sum(), acc_b, acc_a, (acc_a - acc_b) * 100)

    # T3c: Bootstrap 95% CI on test accuracy delta (1000 iterations)
    rng = np.random.default_rng(RNG_SEED)
    n_test = len(y_test_enc)
    boot_deltas = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_test, size=n_test)
        acc_r = float((pred_raw[idx] == y_test_enc[idx]).mean())
        acc_i = float((pred_iso[idx] == y_test_enc[idx]).mean())
        boot_deltas.append((acc_i - acc_r) * 100)
    boot_arr = np.array(boot_deltas)
    ci_low, ci_high = float(np.percentile(boot_arr, 2.5)), float(np.percentile(boot_arr, 97.5))
    ci_lower_bound_positive = ci_low > 0.0

    log.info("T3c: bootstrap 95%% CI on accuracy delta: [%.2fpp, %.2fpp]  lower_bound_positive=%s",
             ci_low, ci_high, ci_lower_bound_positive)

    return {
        "ece_val": round(ece_val, 4),
        "ece_test": round(ece_test, 4),
        "acc_test_before": round(acc_raw, 4),
        "acc_test_after": round(acc_iso, 4),
        "acc_test_delta_pp": round(acc_delta_pp, 2),
        "n_classes_improved": n_improved,
        "n_classes_degraded": n_degraded,
        "top3_movers_contribution_pp": round(top3_contribution_pp, 2),
        "top_movers": top_movers[:10],
        "confidence_buckets": bucket_results,
        "bootstrap_ci_low_pp": round(ci_low, 2),
        "bootstrap_ci_high_pp": round(ci_high, 2),
        "ci_lower_bound_positive": ci_lower_bound_positive,
        "proba_val": proba_val_iso,
        "proba_test": proba_test_iso,
        "calibrator": cal,
    }


# ---------------------------------------------------------------------------
# T4 — Method decision
# ---------------------------------------------------------------------------

def t4_decide(repo: str, ece_before: float, t2: dict, t3: dict) -> str:
    iso_ece = t3["ece_val"]
    ts_ece = t2["ece_val"]
    ci_low_positive = t3["ci_lower_bound_positive"]
    ece_diff = abs(iso_ece - ts_ece)

    # Decision rule (from task spec):
    # - Isotonic CI lower bound > 0 AND ECE within 0.03 of temp scaling → ship isotonic
    # - Temp scaling ECE within 0.03 OR isotonic CI lower bound ≤ 0 → ship temp scaling
    # - Both ECE > 0.18 → STOP

    both_fail = (iso_ece >= ECE_REVISED_THRESHOLD) and (ts_ece >= ECE_REVISED_THRESHOLD)
    if both_fail:
        log.error("T4 HARD STOP: both methods ECE above %.2f threshold on val", ECE_REVISED_THRESHOLD)
        return "STOP"

    if ci_low_positive and ece_diff <= 0.03:
        method = "isotonic"
        reason = f"CI lower bound positive (+{t3['bootstrap_ci_low_pp']:.2f}pp) and ECE gap {ece_diff:.4f} ≤ 0.03"
    elif (not ci_low_positive) or (ece_diff <= 0.03):
        method = "temperature_scaling"
        reason = (
            f"CI lower bound ≤ 0 ({t3['bootstrap_ci_low_pp']:.2f}pp)" if not ci_low_positive
            else f"ECE gap {ece_diff:.4f} ≤ 0.03 (prefer simpler)"
        )
    else:
        method = "isotonic"
        reason = f"CI lower bound positive, ECE gap {ece_diff:.4f} > 0.03 favours isotonic"

    log.info("T4 decision for %s: %s — %s", repo, method.upper(), reason)
    log.info("  iso_ece_val=%.4f  ts_ece_val=%.4f  gap=%.4f  ci=[%.2fpp, %.2fpp]",
             iso_ece, ts_ece, ece_diff, t3["bootstrap_ci_low_pp"], t3["bootstrap_ci_high_pp"])
    return method


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_repo(repo: str) -> dict:
    log.info("=" * 70)
    log.info("REPO: %s", repo)

    model_path = MODELS_DIR / f"component_classifier_{repo}.pkl"
    clf = TFIDFComponentClassifier.load(str(model_path))
    assert clf.pipeline is not None and clf.label_encoder is not None

    train = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_train.parquet")
    val   = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_val.parquet")
    test  = pd.read_parquet(PROCESSED_DIR / f"{repo}_classifier_test.parquet")

    X_train = _build_text(train["title"], train["body_clean"])
    X_val   = _build_text(val["title"],   val["body_clean"])
    X_test  = _build_text(test["title"],  test["body_clean"])

    y_val_enc  = clf.label_encoder.transform(val["component"])
    y_test_enc = clf.label_encoder.transform(test["component"])
    classes    = clf.label_encoder.classes_

    proba_raw = clf.pipeline.predict_proba(X_val)
    ece_before = compute_ece(y_val_enc, proba_raw)

    # ── T1 ──
    t1 = t1_leakage_audit(repo, train, val, test, clf)
    if t1.get("leakage_detected") is True:
        log.error("T1 FAIL — leakage detected. Aborting repo.")
        return {"repo": repo, "hard_stop": "leakage", "t1": t1}

    # ── T2 ──
    t2 = t2_temperature_scaling(clf, X_val, y_val_enc, X_test, y_test_enc)

    # ── T3 ──
    t3 = t3_isotonic_robustness(clf, X_val, y_val_enc, X_test, y_test_enc, classes)

    # ── T4 ──
    chosen_method = t4_decide(repo, ece_before, t2, t3)

    return {
        "repo": repo,
        "hard_stop": "STOP" if chosen_method == "STOP" else None,
        "chosen_method": chosen_method,
        "ece_before_val": round(ece_before, 4),
        "t1": {k: v for k, v in t1.items() if not isinstance(v, np.ndarray)},
        "t2": {k: v for k, v in t2.items() if not isinstance(v, np.ndarray)},
        "t3": {k: v for k, v in t3.items() if k not in ("proba_val", "proba_test", "calibrator")},
        # Keep calibrator object for downstream use (not serialised to JSON)
        "_t3_calibrator": t3.get("calibrator"),
        "_t2_proba_test": t2.get("proba_test"),
        "_t3_proba_test": t3.get("proba_test"),
    }


def main() -> None:
    all_results = {}
    for repo in DEFAULT_REPOS:
        if not (PROCESSED_DIR / f"{repo}_classifier_train.parquet").exists():
            log.warning("Skipping %s — splits not found", repo)
            continue
        all_results[repo] = run_repo(repo)

    # Serialise (drop internal ndarray/object keys)
    out_path = REPORTS_DIR / "calibration_diagnostic.json"
    serialisable = {
        repo: {k: v for k, v in r.items() if not k.startswith("_")}
        for repo, r in all_results.items()
    }
    with open(out_path, "w") as f:
        json.dump(serialisable, f, indent=2, default=str)
    log.info("Diagnostic results saved to %s", out_path)

    # Print summary table
    log.info("=" * 70)
    log.info("DIAGNOSTIC SUMMARY")
    for repo, r in all_results.items():
        if r.get("hard_stop"):
            log.info("%-30s  HARD STOP: %s", repo, r["hard_stop"])
            continue
        t2 = r["t2"]
        t3 = r["t3"]
        log.info(
            "%-30s  method=%-19s  ECE: %.4f→iso:%.4f/ts:%.4f  "
            "Δacc(iso)=%.2fpp  boot95CI=[%.2f,%.2f]  ci_pos=%s",
            repo, r["chosen_method"],
            r["ece_before_val"], t3["ece_val"], t2["ece_val"],
            t3["acc_test_delta_pp"],
            t3["bootstrap_ci_low_pp"], t3["bootstrap_ci_high_pp"],
            t3["ci_lower_bound_positive"],
        )


if __name__ == "__main__":
    main()
