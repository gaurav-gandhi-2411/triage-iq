"""Open search for a signal correlating with resolution COVERAGE FAILURE (ADR-0023).

ADR-0021 established that the conformal interval's own WIDTH does not predict whether it
covers the true resolution time (mean width statistically indistinguishable between covered
and missed issues, k8s: 90.45d vs 91.01d). This script searches OTHER already-available
per-issue signals for one that does correlate — analysis only, no LLM, no retraining, no
schema/pipeline change, zero live calls (CassettePlayer(strict=True) over the clean cassette).

Target labels (computed per issue from existing ground truth):
  - coverage_failure (binary): did the CQR-adjusted interval FAIL to cover
    actual_resolution_days? Same formula as src/triage_iq/api/app.py's /triage handler.
  - error_magnitude (continuous): |point_estimate_days - actual_resolution_days|.

Candidate signals (11 in the corrected family + 1 exploratory categorical, see below):
  1. classifier_confidence      — TF-IDF top-1 confidence (deterministic, ADR-0004)
  2. retrieval_mean_similarity  — mean of top-5 BGE similarity scores
  3. retrieval_max_similarity   — top-1 (max) similarity score
  4. retrieval_similarity_spread— max - min similarity among top-5
  5. title_length_chars
  6. body_length_chars
  7. code_block_present         — binary, "```" substring in body
  8. raw_quantile_spread_days   — hi_days - lo_days BEFORE conformal adjustment. Mathematically
     identical in correlation to conformal width (ADR-0021) since conformal width = raw + 2*Q,
     Q a per-repo CONSTANT — adding a constant never changes a correlation coefficient. Included
     for completeness/transparency, not treated as new evidence.
  9. resolution_bucket_rank     — ordinal 0-4 (hours..long), BUCKET_LABELS order
  10. component_grounded        — binary, predicted_component in classifier_top3 (ADR-0015)
  11. quantile_asymmetry_days   — |(point-lo) - (hi-point)|, the cheap "ensemble disagreement"
      proxy: how asymmetric the raw quantile model's own uncertainty is around its point
      estimate. No retraining -- arithmetic on already-fitted model outputs.
  12. component_identity (EXPLORATORY ONLY, excluded from the corrected family) — 27 distinct
      values across 54 k8s issues (median group size ~1-2). Chi-square/Kruskal-Wallis
      assumptions are badly violated at this group size; reported for transparency, not
      corrected or used in the verdict.

Statistics: pointbiserialr(signal, coverage_failure) for the binary target, spearmanr(signal,
error_magnitude) for the continuous target — uniform across all non-categorical signals
(pointbiserialr is valid and equals the phi coefficient when the signal is itself binary, e.g.
code_block_present / component_grounded). Multiple-comparison correction: k8s's 11 signals x 2
targets = 22 tests form ONE corrected family (scipy.stats.false_discovery_control, Benjamini-
Hochberg, q=0.05), reported alongside Bonferroni (alpha/22) as the more conservative view.
vscode (n=11) is computed and reported but excluded from the corrected family and from any
verdict claim (indicative-only, ADR-0017).

Usage:
    python scripts/analyze_resolution_diagnosticity.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import numpy as np
import pandas as pd
from scipy import stats

from cassette import CassettePlayer
from frozen_retriever import build_frozen_retrievers
from triage_iq.api.loader import _load_conformal_adjustments
from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.grounding import verify_plan_grounding
from triage_iq.models.resolution import BUCKET_LABELS, ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant

MODELS_DIR = ROOT / "data" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette.json"
REPORT_PATH = ROOT / "reports" / "resolution_diagnosticity.json"

REPO_MAP: dict[str, str] = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}
INDICATIVE_ONLY_REPOS = {"microsoft/vscode"}

# Cohen's convention: |r| >= 0.3 is a "medium" effect -- the meaningfulness bar this search
# uses, stated explicitly so "significant but tiny" cannot pass as a finding.
_MIN_MEANINGFUL_EFFECT = 0.3
_FDR_Q = 0.05
_ALPHA = 0.05

_BUCKET_RANK: dict[str, int] = {label: i for i, label in enumerate(BUCKET_LABELS)}

CI_API_KEY = "ci-replay-only"

CONTINUOUS_SIGNALS = [
    "classifier_confidence",
    "retrieval_mean_similarity",
    "retrieval_max_similarity",
    "retrieval_similarity_spread",
    "title_length_chars",
    "body_length_chars",
    "code_block_present",
    "raw_quantile_spread_days",
    "resolution_bucket_rank",
    "component_grounded",
    "quantile_asymmetry_days",
]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_eval_set(path: Path) -> list[dict]:
    issues: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))
    return issues


def _load_models(
    repo: str, slug: str, cassette: CassettePlayer, frozen_retrievers: dict
) -> TriageAssistant:
    # load_classifier() dispatches on the pkl's model_kind marker (ADR-0036).
    classifier = load_classifier(MODELS_DIR, slug)
    predictor = ResolutionTimePredictor.load(
        str(MODELS_DIR / f"resolution_predictor_{slug}.pkl")
    )
    train_df = pd.read_parquet(PROCESSED_DIR / f"{slug}_temporal_train.parquet")
    return TriageAssistant(
        repo=repo,
        classifier=classifier,
        detector=frozen_retrievers[repo],
        predictor=predictor,
        train_df=train_df,
        groq_api_key=CI_API_KEY,
        cache=cassette,
    )


def compute_diagnostic_cases(
    eval_set_path: Path = EVAL_SET_PATH, cassette_path: Path = CASSETTE_PATH
) -> list[dict]:
    """Replay the cassette and extract targets + all candidate signals, per issue."""
    cassette = CassettePlayer(cassette_path, strict=True)
    issues = _load_eval_set(eval_set_path)
    frozen_retrievers = build_frozen_retrievers(eval_set_path)
    conformal_adjustments = _load_conformal_adjustments(MODELS_DIR)

    assistants: dict[str, TriageAssistant] = {}
    for repo, slug in REPO_MAP.items():
        assistants[repo] = _load_models(repo, slug, cassette, frozen_retrievers)

    cases: list[dict] = []
    for issue in issues:
        repo = issue["repo"]
        assistant = assistants[repo]

        row = pd.Series({
            "title": issue["title"],
            "body_clean": issue["body"],
            "number": issue["number"],
            "created_at": (
                pd.Timestamp(issue["created_at"])
                if issue.get("created_at")
                else pd.Timestamp("now", tz="UTC")
            ),
        })

        signals = assistant._collect_signals(row)
        plan, _raw, _usage, _llm_status, _cache_hit = assistant._call_llm_verbose(signals)

        # --- targets ---
        actual_days = float(issue["actual_resolution_days"])
        pred_days = float(signals["pred_days"])
        lo_days = float(signals["lo_days"])
        hi_days = float(signals["hi_days"])

        adj = conformal_adjustments.get(repo)
        if adj is not None:
            q_days = adj["q_adjustment_hours"] / 24.0
            conformal_lower = max(0.0, lo_days - q_days)
            conformal_upper = hi_days + q_days
        else:
            conformal_lower, conformal_upper = lo_days, hi_days
        coverage_failure = int(not (conformal_lower <= actual_days <= conformal_upper))
        error_magnitude = abs(pred_days - actual_days)

        # --- signals ---
        classifier_top3 = signals["classifier_top3"]
        classifier_confidence = float(classifier_top3[0]["confidence"]) if classifier_top3 else 0.0

        sim_scores = [float(s["score"]) for s in signals["similar_raw"]]
        retrieval_mean_similarity = float(np.mean(sim_scores)) if sim_scores else 0.0
        retrieval_max_similarity = float(np.max(sim_scores)) if sim_scores else 0.0
        retrieval_similarity_spread = (
            float(np.max(sim_scores) - np.min(sim_scores)) if sim_scores else 0.0
        )

        body_text = issue["body"] or ""
        title_text = issue["title"] or ""

        retrieved_numbers = {s["number"] for s in signals["similar_raw"]}
        grounding = verify_plan_grounding(plan, classifier_top3, retrieved_numbers)

        bucket = signals["resolution_bucket"]
        lower_half = pred_days - lo_days
        upper_half = hi_days - pred_days

        cases.append({
            "issue_number": issue["number"],
            "repo": repo,
            "coverage_failure": coverage_failure,
            "error_magnitude": error_magnitude,
            "classifier_confidence": classifier_confidence,
            "retrieval_mean_similarity": retrieval_mean_similarity,
            "retrieval_max_similarity": retrieval_max_similarity,
            "retrieval_similarity_spread": retrieval_similarity_spread,
            "title_length_chars": float(len(title_text)),
            "body_length_chars": float(len(body_text)),
            "code_block_present": int("```" in body_text),
            "raw_quantile_spread_days": hi_days - lo_days,
            "resolution_bucket_rank": float(_BUCKET_RANK.get(bucket, 1)),
            "component_grounded": int(grounding.component_grounded),
            "quantile_asymmetry_days": abs(upper_half - lower_half),
            "predicted_component": plan.predicted_component,
        })

    return cases


def _corrected_test(
    signal_values: np.ndarray, target_values: np.ndarray, method: str
) -> dict[str, float]:
    """Run pointbiserialr (method='pointbiserial') or spearmanr (method='spearman')."""
    if method == "pointbiserial":
        r, p = stats.pointbiserialr(signal_values, target_values)
    else:
        r, p = stats.spearmanr(signal_values, target_values)
    return {"r": float(r) if not np.isnan(r) else 0.0, "p": float(p) if not np.isnan(p) else 1.0}


def analyze() -> dict[str, Any]:
    cases = compute_diagnostic_cases(EVAL_SET_PATH, CASSETTE_PATH)

    per_repo_cases: dict[str, list[dict]] = {repo: [] for repo in REPO_MAP}
    for c in cases:
        per_repo_cases[c["repo"]].append(c)

    results: dict[str, Any] = {"per_repo": {}}
    k8s_pvalues: list[float] = []
    k8s_test_keys: list[tuple[str, str]] = []  # (signal, target)

    for repo, repo_cases in per_repo_cases.items():
        n = len(repo_cases)
        coverage_failure = np.array([c["coverage_failure"] for c in repo_cases])
        error_magnitude = np.array([c["error_magnitude"] for c in repo_cases])

        signal_results: dict[str, Any] = {}
        for signal_name in CONTINUOUS_SIGNALS:
            values = np.array([c[signal_name] for c in repo_cases], dtype=float)
            # Constant signal (zero variance) -> correlation undefined; report explicitly.
            if np.std(values) == 0.0:
                vs_coverage = {"r": 0.0, "p": 1.0, "note": "zero variance in this repo"}
                vs_error = {"r": 0.0, "p": 1.0, "note": "zero variance in this repo"}
            else:
                vs_coverage = _corrected_test(values, coverage_failure, "pointbiserial")
                vs_error = _corrected_test(values, error_magnitude, "spearman")

            signal_results[signal_name] = {
                "vs_coverage_failure": vs_coverage,
                "vs_error_magnitude": vs_error,
            }
            if repo == "kubernetes/kubernetes":
                k8s_pvalues.append(vs_coverage["p"])
                k8s_test_keys.append((signal_name, "vs_coverage_failure"))
                k8s_pvalues.append(vs_error["p"])
                k8s_test_keys.append((signal_name, "vs_error_magnitude"))

        # Exploratory-only categorical signal: component identity. Not corrected, not used
        # in the verdict -- median group size is ~1-2 issues (27 distinct components / 54
        # k8s issues), badly violating chi-square/Kruskal-Wallis sample-size assumptions.
        components = [c["predicted_component"] for c in repo_cases]
        distinct_components = sorted(set(components))
        try:
            groups_error = [
                [c["error_magnitude"] for c in repo_cases if c["predicted_component"] == comp]
                for comp in distinct_components
            ]
            groups_error = [g for g in groups_error if len(g) > 0]
            if len(groups_error) >= 2:
                kw_stat, kw_p = stats.kruskal(*groups_error)
            else:
                kw_stat, kw_p = float("nan"), float("nan")
        except ValueError:
            kw_stat, kw_p = float("nan"), float("nan")

        component_exploratory = {
            "n_distinct_components": len(distinct_components),
            "median_group_size": float(np.median([components.count(c) for c in distinct_components])),
            "kruskal_wallis_h": None if np.isnan(kw_stat) else float(kw_stat),
            "kruskal_wallis_p_uncorrected": None if np.isnan(kw_p) else float(kw_p),
            "caveat": (
                "EXPLORATORY ONLY -- excluded from the corrected family and the verdict. "
                "Median group size too small for reliable inference."
            ),
        }

        results["per_repo"][repo] = {
            "n": n,
            "indicative_only": repo in INDICATIVE_ONLY_REPOS,
            "coverage_failure_count": int(coverage_failure.sum()),
            "coverage_failure_rate": round(float(coverage_failure.mean()), 4),
            "mean_error_magnitude_days": round(float(error_magnitude.mean()), 4),
            "signals": signal_results,
            "component_identity_exploratory": component_exploratory,
        }

    # Multiple-comparison correction over the k8s (powered) family only.
    k8s_pvalues_arr = np.array(k8s_pvalues)
    bh_rejected = stats.false_discovery_control(k8s_pvalues_arr, method="bh") <= _FDR_Q
    bonferroni_threshold = _ALPHA / len(k8s_pvalues_arr)

    correction_detail = []
    for (signal_name, target_key), p, bh_ok in zip(k8s_test_keys, k8s_pvalues, bh_rejected):
        r = results["per_repo"]["kubernetes/kubernetes"]["signals"][signal_name][target_key]["r"]
        survives_bonferroni = p < bonferroni_threshold
        meaningful = abs(r) >= _MIN_MEANINGFUL_EFFECT
        correction_detail.append({
            "signal": signal_name,
            "target": target_key,
            "r": round(r, 4),
            "p_raw": round(p, 4),
            "p_uncorrected_significant": p < _ALPHA,
            "survives_bh_fdr_q05": bool(bh_ok),
            "survives_bonferroni": bool(survives_bonferroni),
            "meaningful_effect_size": bool(meaningful),
            "diagnostic": bool(p < _ALPHA and bh_ok and meaningful),
        })

    results["multiple_comparison_correction"] = {
        "family": "kubernetes/kubernetes only (powered repo) -- 11 signals x 2 targets = 22 tests",
        "n_tests": len(k8s_pvalues_arr),
        "method": "Benjamini-Hochberg FDR (primary, q=0.05) + Bonferroni (reported, alpha=0.05/n)",
        "bonferroni_alpha_threshold": round(bonferroni_threshold, 6),
        "meaningful_effect_threshold": _MIN_MEANINGFUL_EFFECT,
        "per_test": correction_detail,
    }

    diagnostic_signals = [t for t in correction_detail if t["diagnostic"]]
    results["verdict"] = {
        "outcome": "POSITIVE" if diagnostic_signals else "NEGATIVE",
        "diagnostic_signals": diagnostic_signals,
        "summary": (
            f"{len(diagnostic_signals)}/{len(correction_detail)} k8s tests are diagnostic "
            "(raw p<0.05 AND survives BH-FDR q=0.05 AND |r|>=0.3)."
        ),
    }

    results["eval_set_hash"] = _file_sha256(EVAL_SET_PATH)
    results["cassette_hash"] = _file_sha256(CASSETTE_PATH)
    return results


def _print_report(result: dict[str, Any]) -> None:
    print("\n=== RESOLUTION DIAGNOSTICITY SEARCH (ADR-0023) ===\n")
    for repo, data in result["per_repo"].items():
        tag = " (INDICATIVE ONLY)" if data["indicative_only"] else ""
        print(f"  {repo}: n={data['n']}{tag}")
        print(f"    coverage_failure_rate: {data['coverage_failure_rate']*100:.2f}% "
              f"({data['coverage_failure_count']}/{data['n']})")
        print(f"    mean error_magnitude: {data['mean_error_magnitude_days']:.2f} days")
        for signal, vals in data["signals"].items():
            vc = vals["vs_coverage_failure"]
            ve = vals["vs_error_magnitude"]
            print(f"    {signal:<28} vs_coverage r={vc['r']:+.3f} p={vc['p']:.4f}  "
                  f"vs_error rho={ve['r']:+.3f} p={ve['p']:.4f}")
        ce = data["component_identity_exploratory"]
        print(f"    component_identity (EXPLORATORY): {ce['n_distinct_components']} distinct, "
              f"median group size={ce['median_group_size']}, "
              f"Kruskal-Wallis p={ce['kruskal_wallis_p_uncorrected']}")
        print()

    mcc = result["multiple_comparison_correction"]
    print(f"  --- multiple-comparison correction ({mcc['family']}) ---")
    print(f"  n_tests={mcc['n_tests']}, Bonferroni alpha threshold={mcc['bonferroni_alpha_threshold']}")
    for t in mcc["per_test"]:
        flag = "DIAGNOSTIC" if t["diagnostic"] else ""
        print(f"    {t['signal']:<28} {t['target']:<20} r={t['r']:+.3f} p={t['p_raw']:.4f} "
              f"BH={t['survives_bh_fdr_q05']} Bonf={t['survives_bonferroni']} {flag}")

    print(f"\n  VERDICT: {result['verdict']['outcome']} — {result['verdict']['summary']}")


def main() -> None:
    result = analyze()
    _print_report(result)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"\nStructured report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
