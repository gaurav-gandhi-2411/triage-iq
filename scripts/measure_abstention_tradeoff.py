"""Measure the coverage-vs-abstention tradeoff for selective prediction (ADR-0021).

Replays the CURRENT (unmodified, flag-off) synthesis pipeline against the clean cassette
(eval/cassettes/eval_cassette.json) over the full eval/eval_set.jsonl, using the same
cassette-only replay machinery as eval/run_eval.py and scripts/measure_grounding.py
(CassettePlayer(strict=True) -- zero live API calls, zero new models).

For each issue, computes the signals a selective-prediction gate would use:
  - component stage: component_confidence (calibrated TF-IDF, ADR-0004) and
    grounding_status.component_grounded (deterministic, ADR-0015)
  - resolution stage: the CQR-adjusted conformal interval, using the exact same formula
    as src/triage_iq/api/app.py's /triage handler (q_days = Q/24; lower = raw_lower - q_days;
    upper = raw_upper + q_days) applied to the already-recorded plan.expected_resolution_*_days
    -- no new predictor calls needed.

Then sweeps a threshold grid per stage per repo and reports the full coverage-vs-abstention
curve (abstention_rate vs. accuracy/coverage-on-answered), plus a proposed default operating
point (see _pick_default). Reported per-repo, never pooled -- vscode's n=11 is flagged
indicative-only per ADR-0017's data-ceiling finding; k8s's n=54 is where the tradeoff is
actually measured.

Priority stage is explicitly out of scope for v1: priority_guess has no calibrated confidence
signal anywhere in the pipeline, and inventing a proxy threshold with no measured basis would
violate the same discipline used everywhere else in this project. See ADR-0021.

Usage:
    python scripts/measure_abstention_tradeoff.py
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

import pandas as pd

from cassette import CassettePlayer
from frozen_retriever import build_frozen_retrievers
from triage_iq.api.loader import _load_conformal_adjustments
from triage_iq.models.component_classifier import TFIDFComponentClassifier
from triage_iq.models.grounding import verify_plan_grounding
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant

MODELS_DIR = ROOT / "data" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette.json"
REPORT_PATH = ROOT / "reports" / "abstention_tradeoff.json"

REPO_MAP: dict[str, str] = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}

# ADR-0017: vscode's n=11 gives any rate computed on it high variance (one issue = ~9pp) --
# curves are still computed, but reported and consumed as indicative-only, never with the
# same weight as k8s's n=54.
INDICATIVE_ONLY_REPOS = {"microsoft/vscode"}

# A ceiling on the abstention rate considered when picking a proposed default operating
# point -- abstaining on more than half of all issues defeats the purpose of "selective"
# prediction, so points beyond this are excluded from the pick even if their accuracy is
# higher. This bound is a design choice, not derived from the data; the full curve (all
# points, no ceiling) is reported regardless so a human can pick any other point.
_MAX_ABSTENTION_FOR_DEFAULT = 0.50

CI_API_KEY = "ci-replay-only"


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_eval_set(path: Path) -> list[dict]:
    """Read JSONL eval set, returning one dict per non-empty line."""
    issues: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                issues.append(json.loads(line))
    return issues


def _load_models(
    repo: str,
    slug: str,
    cassette: CassettePlayer,
    frozen_retrievers: dict,
) -> TriageAssistant:
    """Load per-repo classifier/predictor/train_df and construct a TriageAssistant.

    Mirrors scripts/measure_grounding.py:_load_models exactly.
    """
    classifier = TFIDFComponentClassifier.load(
        str(MODELS_DIR / f"component_classifier_{slug}.pkl")
    )
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


def compute_abstention_cases(
    eval_set_path: Path = EVAL_SET_PATH, cassette_path: Path = CASSETTE_PATH
) -> list[dict]:
    """Replay the cassette over every issue and compute the per-issue abstention signals.

    Returns one case dict per eval-set issue, in eval_set.jsonl order.
    """
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

        retrieved_numbers = {s["number"] for s in signals["similar_raw"]}
        grounding = verify_plan_grounding(plan, signals["classifier_top3"], retrieved_numbers)

        # Same formula as src/triage_iq/api/app.py's /triage handler.
        adj = conformal_adjustments.get(repo)
        if adj is not None:
            q_days = adj["q_adjustment_hours"] / 24.0
            conformal_lower = max(0.0, plan.expected_resolution_lower_days - q_days)
            conformal_upper = plan.expected_resolution_upper_days + q_days
        else:
            conformal_lower = plan.expected_resolution_lower_days
            conformal_upper = plan.expected_resolution_upper_days

        actual_days = float(issue["actual_resolution_days"])
        covered = conformal_lower <= actual_days <= conformal_upper

        cases.append({
            "issue_number": issue["number"],
            "repo": repo,
            "predicted_component": plan.predicted_component,
            "gold_component": issue["gold_component"],
            "component_correct": plan.predicted_component == issue["gold_component"],
            "component_confidence": float(plan.component_confidence),
            "component_grounded": grounding.component_grounded,
            "conformal_lower_days": round(conformal_lower, 4),
            "conformal_upper_days": round(conformal_upper, 4),
            "conformal_width_days": round(conformal_upper - conformal_lower, 4),
            "actual_resolution_days": round(actual_days, 4),
            "resolution_covered": covered,
        })

    return cases


def _pick_default(
    curve: list[dict], rate_key: str, metric_key: str
) -> dict | None:
    """Pick a proposed default operating point: max metric-on-answered s.t. abstention
    rate <= _MAX_ABSTENTION_FOR_DEFAULT, ties broken toward the lower abstention rate.

    Returns None if no point in the curve has a defined metric within the abstention
    ceiling (e.g. n too small for any answered subset).
    """
    candidates = [
        pt for pt in curve
        if pt[metric_key] is not None and pt[rate_key] <= _MAX_ABSTENTION_FOR_DEFAULT
    ]
    if not candidates:
        return None
    best_metric = max(pt[metric_key] for pt in candidates)
    tied = [pt for pt in candidates if pt[metric_key] == best_metric]
    return min(tied, key=lambda pt: pt[rate_key])


def _sweep_component(cases_repo: list[dict]) -> list[dict]:
    """Sweep confidence threshold t; abstain if confidence < t OR not component_grounded.

    Grounding is a hard trigger (not swept) -- the same two issues (#13057, #311836) that
    are component-ungrounded per ADR-0015 always abstain regardless of t, since a component
    the classifier's own top-3 doesn't support is not something confidence alone should
    override.
    """
    n = len(cases_repo)
    if n == 0:
        return []
    conf_values = sorted({round(c["component_confidence"], 6) for c in cases_repo})
    thresholds = [0.0, *conf_values, conf_values[-1] + 1e-6]

    curve: list[dict] = []
    seen: set[float] = set()
    for t in thresholds:
        if t in seen:
            continue
        seen.add(t)
        abstain_mask = [
            (c["component_confidence"] < t) or (not c["component_grounded"])
            for c in cases_repo
        ]
        n_abstained = sum(abstain_mask)
        n_answered = n - n_abstained
        answered_correct = sum(
            c["component_correct"] for c, ab in zip(cases_repo, abstain_mask) if not ab
        )
        curve.append({
            "confidence_threshold": round(t, 4),
            "abstention_rate": round(n_abstained / n, 4),
            "n_answered": n_answered,
            "n_abstained": n_abstained,
            "accuracy_on_answered": (
                round(answered_correct / n_answered, 4) if n_answered else None
            ),
        })
    return curve


def _sweep_resolution(cases_repo: list[dict]) -> list[dict]:
    """Sweep conformal-interval-width threshold w; abstain if width > w.

    As w decreases, more issues are abstained (their interval is "too wide to be useful");
    the point w >= max(width) is the never-abstain baseline.
    """
    n = len(cases_repo)
    if n == 0:
        return []
    width_values = sorted({round(c["conformal_width_days"], 6) for c in cases_repo})
    thresholds = [width_values[0] - 1e-6, *width_values, width_values[-1] + 1e-6]

    curve: list[dict] = []
    seen: set[float] = set()
    for w in thresholds:
        if w in seen:
            continue
        seen.add(w)
        abstain_mask = [c["conformal_width_days"] > w for c in cases_repo]
        n_abstained = sum(abstain_mask)
        n_answered = n - n_abstained
        answered_covered = sum(
            c["resolution_covered"] for c, ab in zip(cases_repo, abstain_mask) if not ab
        )
        curve.append({
            "width_threshold_days": round(w, 4),
            "abstention_rate": round(n_abstained / n, 4),
            "n_answered": n_answered,
            "n_abstained": n_abstained,
            "coverage_on_answered": (
                round(answered_covered / n_answered, 4) if n_answered else None
            ),
        })
    return curve


def measure() -> dict[str, Any]:
    """Run cassette-only replay and compute both stages' coverage-vs-abstention curves."""
    cases = compute_abstention_cases(EVAL_SET_PATH, CASSETTE_PATH)

    per_repo_cases: dict[str, list[dict]] = {repo: [] for repo in REPO_MAP}
    for case in cases:
        per_repo_cases[case["repo"]].append(case)

    component_stage: dict[str, Any] = {}
    resolution_stage: dict[str, Any] = {}

    for repo, repo_cases in per_repo_cases.items():
        n = len(repo_cases)
        baseline_accuracy = (
            round(sum(c["component_correct"] for c in repo_cases) / n, 4) if n else None
        )
        curve = _sweep_component(repo_cases)
        default = _pick_default(curve, "abstention_rate", "accuracy_on_answered")
        component_stage[repo] = {
            "n": n,
            "indicative_only": repo in INDICATIVE_ONLY_REPOS,
            "baseline_accuracy_no_abstention": baseline_accuracy,
            "curve": curve,
            "proposed_default": default,
        }

        baseline_coverage = (
            round(sum(c["resolution_covered"] for c in repo_cases) / n, 4) if n else None
        )
        r_curve = _sweep_resolution(repo_cases)
        r_default = _pick_default(r_curve, "abstention_rate", "coverage_on_answered")
        resolution_stage[repo] = {
            "n": n,
            "indicative_only": repo in INDICATIVE_ONLY_REPOS,
            "baseline_coverage_no_abstention": baseline_coverage,
            "curve": r_curve,
            "proposed_default": r_default,
        }

    return {
        "eval_set_hash": _file_sha256(EVAL_SET_PATH),
        "cassette_hash": _file_sha256(CASSETTE_PATH),
        "component_stage": component_stage,
        "resolution_stage": resolution_stage,
        "priority_stage": {
            "status": "out_of_scope_v1",
            "reason": (
                "priority_guess has no calibrated confidence signal anywhere in the "
                "pipeline (unlike component_confidence or the CQR interval) -- inventing a "
                "proxy threshold with no measured basis would be a fabricated signal, not a "
                "data-derived one. Flagged, not gated, in this build. See ADR-0021."
            ),
        },
    }


def _print_stage(name: str, stage: dict[str, Any], rate_key: str, metric_key: str, threshold_key: str) -> None:
    print(f"\n=== {name} ===")
    for repo, data in stage.items():
        tag = " (INDICATIVE ONLY, n=%d)" % data["n"] if data["indicative_only"] else ""
        print(f"\n  {repo}: n={data['n']}{tag}")
        baseline_key = "baseline_accuracy_no_abstention" if "accuracy" in metric_key else "baseline_coverage_no_abstention"
        print(f"    no-abstention baseline: {data[baseline_key]}")
        print(f"    {threshold_key:>22} {rate_key:>16} {'n_answered':>11} {metric_key:>20}")
        for pt in data["curve"]:
            print(
                f"    {pt[threshold_key]:>22} {pt[rate_key]:>16} {pt['n_answered']:>11} "
                f"{pt[metric_key]}"
            )
        print(f"    proposed default: {data['proposed_default']}")


def main() -> None:
    """CLI entry point: run measurement, print curves, write reports/abstention_tradeoff.json."""
    result = measure()

    _print_stage(
        "COMPONENT STAGE", result["component_stage"],
        "abstention_rate", "accuracy_on_answered", "confidence_threshold",
    )
    _print_stage(
        "RESOLUTION STAGE", result["resolution_stage"],
        "abstention_rate", "coverage_on_answered", "width_threshold_days",
    )
    print("\n=== PRIORITY STAGE ===")
    print(f"  {result['priority_stage']['status']}: {result['priority_stage']['reason']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStructured report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
