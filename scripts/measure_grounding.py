"""Measure LLM synthesis grounding against classifier_top3 and retrieval outputs.

Replays the CURRENT (unmodified) synthesis pipeline against the CURRENT cassette
(eval/cassettes/eval_cassette.json) over the full eval/eval_set.jsonl, using the
same cassette-only replay machinery as eval/run_eval.py (CassettePlayer(strict=True)
— zero live API calls). For each issue, computes a GroundingReport comparing the
LLM's TriagePlan claims against the classifier_top3 and retrieved similar-issue
numbers that were actually shown to it.

Usage:
    python scripts/measure_grounding.py
"""
from __future__ import annotations

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
from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.grounding import verify_plan_grounding
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant

MODELS_DIR = ROOT / "data" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette.json"
REPORT_PATH = ROOT / "reports" / "grounding_measurement.json"

REPO_MAP: dict[str, str] = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}

CI_API_KEY = "ci-replay-only"


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

    Mirrors eval/run_eval.py:_load_models exactly, using FrozenRetriever instead of
    live FAISS so this is a deterministic, cassette-only replay.
    """
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


def _normalize(label: str) -> str:
    """Case-insensitive, whitespace-normalized form used only for the diagnostic signal."""
    return label.strip().casefold()


def _diagnostic_ci_match(predicted_component: str, classifier_top3: list[dict]) -> bool:
    """Return True if predicted_component would match a top-3 label case-insensitively.

    Purely diagnostic — never used to change all_grounded. Lets us separate real
    hallucination from the LLM changing capitalization/whitespace only.
    """
    normalized_labels = {_normalize(str(entry["label"])) for entry in classifier_top3}
    return _normalize(predicted_component) in normalized_labels


def compute_grounding_reports(
    eval_set_path: Path = EVAL_SET_PATH, cassette_path: Path = CASSETTE_PATH
) -> list[dict]:
    """Replay the cassette over every issue in `eval_set_path` and compute grounding cases.

    Shared by `measure()` (this script) and `eval/test_invariants.py`'s ratchet/pin tests, so
    both consume the identical cassette-replay pipeline instead of two copies of the wiring.

    Returns one case dict per eval-set issue, in eval_set.jsonl order, each with:
    issue_number, repo, component_grounded, component_reason,
    case_insensitive_match_diagnostic, ungrounded_refs, all_grounded, predicted_component,
    classifier_top3_labels, retrieved_numbers.
    """
    cassette = CassettePlayer(cassette_path, strict=True)
    issues = _load_eval_set(eval_set_path)
    frozen_retrievers = build_frozen_retrievers(eval_set_path)

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
        report = verify_plan_grounding(plan, signals["classifier_top3"], retrieved_numbers)
        ci_match = _diagnostic_ci_match(plan.predicted_component, signals["classifier_top3"])

        cases.append({
            "issue_number": issue["number"],
            "repo": repo,
            "component_grounded": report.component_grounded,
            "component_reason": report.component_reason,
            "case_insensitive_match_diagnostic": ci_match,
            "ungrounded_refs": report.ungrounded_refs,
            "all_grounded": report.all_grounded,
            "predicted_component": plan.predicted_component,
            "classifier_top3_labels": [e["label"] for e in signals["classifier_top3"]],
            "retrieved_numbers": sorted(retrieved_numbers),
        })

    return cases


def measure() -> dict[str, Any]:
    """Run cassette-only replay over the full eval set and compute grounding stats.

    Returns a dict with per_repo breakdown, overall totals, and the full list of
    ungrounded cases with enough detail to inspect (not just counts).
    """
    cases = compute_grounding_reports(EVAL_SET_PATH, CASSETTE_PATH)

    per_repo_cases: dict[str, list[dict]] = {repo: [] for repo in REPO_MAP}
    ungrounded_cases: list[dict] = []
    for case in cases:
        per_repo_cases[case["repo"]].append(case)
        if not case["all_grounded"]:
            ungrounded_cases.append(case)

    per_repo: dict[str, dict] = {}
    overall_n = 0
    overall_component_ungrounded = 0
    overall_component_ungrounded_ci_recoverable = 0
    overall_refs_ungrounded = 0
    overall_all_ungrounded = 0

    for repo, cases in per_repo_cases.items():
        n = len(cases)
        component_ungrounded = [c for c in cases if not c["component_grounded"]]
        n_component_ungrounded = len(component_ungrounded)
        n_ci_recoverable = sum(
            1 for c in component_ungrounded if c["case_insensitive_match_diagnostic"]
        )
        n_refs_ungrounded = sum(1 for c in cases if c["ungrounded_refs"])
        n_all_ungrounded = sum(1 for c in cases if not c["all_grounded"])

        per_repo[repo] = {
            "n": n,
            "component_ungrounded_count": n_component_ungrounded,
            "component_ungrounded_pct": round(100 * n_component_ungrounded / n, 2) if n else 0.0,
            "component_ungrounded_ci_recoverable_count": n_ci_recoverable,
            "component_ungrounded_ci_recoverable_pct_of_ungrounded": (
                round(100 * n_ci_recoverable / n_component_ungrounded, 2)
                if n_component_ungrounded
                else 0.0
            ),
            "similar_issue_ungrounded_count": n_refs_ungrounded,
            "similar_issue_ungrounded_pct": round(100 * n_refs_ungrounded / n, 2) if n else 0.0,
            "all_grounded_false_count": n_all_ungrounded,
            "all_grounded_false_pct": round(100 * n_all_ungrounded / n, 2) if n else 0.0,
        }

        overall_n += n
        overall_component_ungrounded += n_component_ungrounded
        overall_component_ungrounded_ci_recoverable += n_ci_recoverable
        overall_refs_ungrounded += n_refs_ungrounded
        overall_all_ungrounded += n_all_ungrounded

    overall = {
        "n": overall_n,
        "component_ungrounded_count": overall_component_ungrounded,
        "component_ungrounded_pct": (
            round(100 * overall_component_ungrounded / overall_n, 2) if overall_n else 0.0
        ),
        "component_ungrounded_ci_recoverable_count": overall_component_ungrounded_ci_recoverable,
        "component_ungrounded_ci_recoverable_pct_of_ungrounded": (
            round(
                100
                * overall_component_ungrounded_ci_recoverable
                / overall_component_ungrounded,
                2,
            )
            if overall_component_ungrounded
            else 0.0
        ),
        "similar_issue_ungrounded_count": overall_refs_ungrounded,
        "similar_issue_ungrounded_pct": (
            round(100 * overall_refs_ungrounded / overall_n, 2) if overall_n else 0.0
        ),
        "all_grounded_false_count": overall_all_ungrounded,
        "all_grounded_false_pct": (
            round(100 * overall_all_ungrounded / overall_n, 2) if overall_n else 0.0
        ),
    }

    return {
        "per_repo": per_repo,
        "overall": overall,
        "ungrounded_cases": ungrounded_cases,
    }


def _print_report(result: dict[str, Any]) -> None:
    """Print a human-readable grounding report to stdout."""
    print("\n=== GROUNDING MEASUREMENT ===\n")

    for repo, data in result["per_repo"].items():
        print(f"  {repo}: n={data['n']}")
        print(
            f"    component ungrounded: {data['component_ungrounded_count']} "
            f"({data['component_ungrounded_pct']}%)"
        )
        print(
            "    of those, case-insensitive/whitespace-normalized match would succeed: "
            f"{data['component_ungrounded_ci_recoverable_count']} "
            f"({data['component_ungrounded_ci_recoverable_pct_of_ungrounded']}% of ungrounded)"
        )
        print(
            f"    similar-issue hallucination (>=1 ungrounded ref): "
            f"{data['similar_issue_ungrounded_count']} ({data['similar_issue_ungrounded_pct']}%)"
        )
        print(
            f"    all_grounded is False: {data['all_grounded_false_count']} "
            f"({data['all_grounded_false_pct']}%)"
        )
        print()

    overall = result["overall"]
    print(f"  OVERALL: n={overall['n']}")
    print(
        f"    component ungrounded: {overall['component_ungrounded_count']} "
        f"({overall['component_ungrounded_pct']}%)"
    )
    print(
        "    of those, case-insensitive/whitespace-normalized match would succeed: "
        f"{overall['component_ungrounded_ci_recoverable_count']} "
        f"({overall['component_ungrounded_ci_recoverable_pct_of_ungrounded']}% of ungrounded)"
    )
    print(
        f"    similar-issue hallucination (>=1 ungrounded ref): "
        f"{overall['similar_issue_ungrounded_count']} ({overall['similar_issue_ungrounded_pct']}%)"
    )
    print(
        f"    all_grounded is False: {overall['all_grounded_false_count']} "
        f"({overall['all_grounded_false_pct']}%)"
    )

    print("\n  --- ungrounded case detail ---")
    if not result["ungrounded_cases"]:
        print("  (none)")
    for case in result["ungrounded_cases"]:
        print(
            f"  #{case['issue_number']} ({case['repo']}): "
            f"predicted_component={case['predicted_component']!r}, "
            f"classifier_top3_labels={case['classifier_top3_labels']}, "
            f"component_reason={case['component_reason']}, "
            f"case_insensitive_match_diagnostic={case['case_insensitive_match_diagnostic']}, "
            f"ungrounded_refs={case['ungrounded_refs']}, "
            f"retrieved_numbers={case['retrieved_numbers']}"
        )


def main() -> None:
    """CLI entry point: run measurement, print report, write reports/grounding_measurement.json."""
    result = measure()
    _print_report(result)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStructured report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
