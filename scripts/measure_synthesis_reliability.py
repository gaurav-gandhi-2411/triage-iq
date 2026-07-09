"""Measure the CURRENT synthesis reliability rate — the measure-first gate for ADR-0022.

Replays the clean, shipped-default synthesis pipeline (TRIAGE_PROMPT_INCLUDE_ATTRIBUTION off,
same as scripts/measure_grounding.py) against the committed cassette
(eval/cassettes/eval_cassette.json) over the full eval/eval_set.jsonl, using the same
cassette-only replay machinery as eval/run_eval.py (CassettePlayer(strict=True) — zero live
API calls).

Calls TriageAssistant.triage_with_metadata (the same method app.py's /triage handler calls) so
every measured signal matches production exactly, including the internal-consistency check
(ADR-0022) computed inside it. For each issue, records:
  - llm_status (from metadata): "ok" | "parse_retry_succeeded" | "parse_failure" — parse-retry
    and parse-failure together are the "malformed" rate this measure-first gate exists to
    report, and the baseline structured generation must beat if built.
  - consistency_status.all_consistent (ADR-0022) — the semantic-inconsistency rate.

This is the baseline structured generation must beat: if malformed + retry + failure are
already ~0%, there is little for schema-constrained generation to fix, and that is reported
as a valid finding (synthesis already robust), not a failure to find something to build.

Usage:
    python scripts/measure_synthesis_reliability.py
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
from triage_iq.models.component_classifier import TFIDFComponentClassifier
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant

MODELS_DIR = ROOT / "data" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette.json"
REPORT_PATH = ROOT / "reports" / "synthesis_reliability.json"

REPO_MAP: dict[str, str] = {
    "microsoft/vscode": "microsoft_vscode",
    "kubernetes/kubernetes": "kubernetes_kubernetes",
}

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
    """Mirrors scripts/measure_grounding.py:_load_models exactly."""
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


def compute_reliability_cases(
    eval_set_path: Path = EVAL_SET_PATH, cassette_path: Path = CASSETTE_PATH
) -> list[dict]:
    """Replay the cassette over every issue and record llm_status + consistency per issue."""
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

        plan, meta = assistant.triage_with_metadata(row)
        consistency = plan.consistency_status

        cases.append({
            "issue_number": issue["number"],
            "repo": repo,
            "llm_status": meta["llm_status"],
            "predicted_component": plan.predicted_component,
            "all_consistent": consistency.all_consistent if consistency is not None else None,
            "priority_resolution_consistent": (
                consistency.priority_resolution_consistent if consistency is not None else None
            ),
            "override_reason_consistent": (
                consistency.override_reason_consistent if consistency is not None else None
            ),
        })

    return cases


def measure() -> dict[str, Any]:
    """Run cassette-only replay and compute the llm_status + consistency distributions."""
    cases = compute_reliability_cases(EVAL_SET_PATH, CASSETTE_PATH)

    per_repo_cases: dict[str, list[dict]] = {repo: [] for repo in REPO_MAP}
    for case in cases:
        per_repo_cases[case["repo"]].append(case)

    per_repo: dict[str, dict] = {}
    overall_counts: dict[str, int] = {"ok": 0, "parse_retry_succeeded": 0, "parse_failure": 0}
    overall_n = 0
    overall_inconsistent = 0
    non_ok_cases: list[dict] = []
    inconsistent_cases: list[dict] = []

    for repo, repo_cases in per_repo_cases.items():
        n = len(repo_cases)
        counts = {"ok": 0, "parse_retry_succeeded": 0, "parse_failure": 0}
        n_inconsistent = 0
        for c in repo_cases:
            counts[c["llm_status"]] = counts.get(c["llm_status"], 0) + 1
            overall_counts[c["llm_status"]] = overall_counts.get(c["llm_status"], 0) + 1
            if c["llm_status"] != "ok":
                non_ok_cases.append(
                    {"issue_number": c["issue_number"], "repo": c["repo"], "llm_status": c["llm_status"]}
                )
            if c["all_consistent"] is False:
                n_inconsistent += 1
                inconsistent_cases.append({
                    "issue_number": c["issue_number"],
                    "repo": c["repo"],
                    "priority_resolution_consistent": c["priority_resolution_consistent"],
                    "override_reason_consistent": c["override_reason_consistent"],
                })

        malformed = counts["parse_retry_succeeded"] + counts["parse_failure"]
        per_repo[repo] = {
            "n": n,
            "counts": counts,
            "malformed_count": malformed,
            "malformed_rate": round(malformed / n, 4) if n else 0.0,
            "inconsistent_count": n_inconsistent,
            "inconsistent_rate": round(n_inconsistent / n, 4) if n else 0.0,
        }
        overall_n += n
        overall_inconsistent += n_inconsistent

    overall_malformed = overall_counts["parse_retry_succeeded"] + overall_counts["parse_failure"]

    return {
        "eval_set_hash": _file_sha256(EVAL_SET_PATH),
        "cassette_hash": _file_sha256(CASSETTE_PATH),
        "per_repo": per_repo,
        "overall": {
            "n": overall_n,
            "counts": overall_counts,
            "malformed_count": overall_malformed,
            "malformed_rate": round(overall_malformed / overall_n, 4) if overall_n else 0.0,
            "inconsistent_count": overall_inconsistent,
            "inconsistent_rate": round(overall_inconsistent / overall_n, 4) if overall_n else 0.0,
        },
        "non_ok_cases": non_ok_cases,
        "inconsistent_cases": inconsistent_cases,
    }


def _print_report(result: dict[str, Any]) -> None:
    print("\n=== SYNTHESIS RELIABILITY (measure-first gate, ADR-0022) ===\n")
    for repo, data in result["per_repo"].items():
        print(f"  {repo}: n={data['n']}")
        print(f"    counts: {data['counts']}")
        print(f"    malformed (retry-succeeded + parse-failure): {data['malformed_count']} "
              f"({data['malformed_rate']*100:.2f}%)")
        print(f"    inconsistent (ADR-0022 verifier): {data['inconsistent_count']} "
              f"({data['inconsistent_rate']*100:.2f}%)")
        print()

    overall = result["overall"]
    print(f"  OVERALL: n={overall['n']}")
    print(f"    counts: {overall['counts']}")
    print(f"    malformed rate: {overall['malformed_count']} ({overall['malformed_rate']*100:.2f}%)")
    print(f"    inconsistent rate: {overall['inconsistent_count']} ({overall['inconsistent_rate']*100:.2f}%)")

    print("\n  --- non-ok llm_status cases ---")
    if not result["non_ok_cases"]:
        print("  (none)")
    for c in result["non_ok_cases"]:
        print(f"  #{c['issue_number']} ({c['repo']}): {c['llm_status']}")

    print("\n  --- inconsistent-plan cases ---")
    if not result["inconsistent_cases"]:
        print("  (none)")
    for c in result["inconsistent_cases"]:
        print(
            f"  #{c['issue_number']} ({c['repo']}): "
            f"priority_resolution_consistent={c['priority_resolution_consistent']}, "
            f"override_reason_consistent={c['override_reason_consistent']}"
        )


def main() -> None:
    """CLI entry point: run measurement, print report, write reports/synthesis_reliability.json."""
    result = measure()
    _print_report(result)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStructured report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
