"""Measure LLM-declared attribution fidelity against classifier_top3 and retrieval outputs.

Replays the attribution-prompt synthesis pipeline (TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1) against
its own dedicated cassette (eval/cassettes/eval_cassette_attribution.json) over the full
eval/eval_set.jsonl, using the same cassette-only replay machinery as eval/run_eval.py and
scripts/measure_grounding.py (CassettePlayer(strict=True) — zero live API calls).

ADR-0020: the attribution prompt is opt-in (TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1, set below) and
off by default, so eval/run_eval.py and scripts/measure_grounding.py keep replaying the legacy
prompt against eval/cassettes/eval_cassette.json unchanged — this script is the only one that
turns attribution on, against its own separate cassette.

For each issue, computes:
  - a GroundingReport (verify_plan_grounding) for continuity with scripts/measure_grounding.py
  - a DeclaredAttributionReport (verify_declared_attribution) scoring the LLM's *declared*
    attribution (ADR-0020) against the same upstream signals
  - a raw-compliance classification (absent / malformed / wellformed / unparseable_raw)
    describing whether the model emitted a well-formed declared_attribution block at all.

Usage:
    python scripts/measure_attribution_fidelity.py
"""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
from pathlib import Path
from typing import Any

# ADR-0020: must be set before any TriageAssistant call builds its messages -- the flag is read
# per-call via os.environ.get, so this only needs to land before compute_attribution_reports runs.
os.environ["TRIAGE_PROMPT_INCLUDE_ATTRIBUTION"] = "1"

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import pandas as pd

from cassette import CassettePlayer
from frozen_retriever import build_frozen_retrievers
from triage_iq.models.component_classifier import load_classifier
from triage_iq.models.grounding import verify_declared_attribution, verify_plan_grounding
from triage_iq.models.resolution import ResolutionTimePredictor
from triage_iq.models.triage import TriageAssistant, TriagePlan

MODELS_DIR = ROOT / "data" / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
EVAL_SET_PATH = ROOT / "eval" / "eval_set.jsonl"
CASSETTE_PATH = ROOT / "eval" / "cassettes" / "eval_cassette_attribution.json"
REPORT_PATH = ROOT / "reports" / "attribution_fidelity.json"

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

    Mirrors eval/run_eval.py:_load_models and scripts/measure_grounding.py:_load_models
    exactly, using FrozenRetriever instead of live FAISS so this is a deterministic,
    cassette-only replay.
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


def _classify_raw_status(raw: str, plan: TriagePlan) -> str:
    """Classify raw LLM output compliance with the declared_attribution schema addition.

    Uses the same markdown-fence-strip + first-{...}-block regex approach as
    TriageAssistant._parse_plan, so this reflects exactly what the assistant saw.

    Returns one of: "unparseable_raw" (no JSON object found, or JSON invalid),
    "absent" (JSON parses but has no declared_attribution key), "malformed" (key
    present but the tolerant validator on TriagePlan discarded it to None), or
    "wellformed".
    """
    text = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text.strip(), flags=re.MULTILINE)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return "unparseable_raw"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "unparseable_raw"
    if "declared_attribution" not in data:
        return "absent"
    if plan.declared_attribution is None:
        return "malformed"
    return "wellformed"


def compute_attribution_reports(
    eval_set_path: Path = EVAL_SET_PATH, cassette_path: Path = CASSETTE_PATH
) -> list[dict]:
    """Replay the cassette over every issue in `eval_set_path` and compute attribution cases.

    Returns one case dict per eval-set issue, in eval_set.jsonl order.
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
        plan, raw, _usage, llm_status, _cache_hit = assistant._call_llm_verbose(signals)

        retrieved_numbers = {s["number"] for s in signals["similar_raw"]}
        g = verify_plan_grounding(plan, signals["classifier_top3"], retrieved_numbers)
        a = verify_declared_attribution(plan, signals["classifier_top3"], retrieved_numbers)
        raw_status = _classify_raw_status(raw, plan)

        cases.append({
            "issue_number": issue["number"],
            "repo": repo,
            "llm_status": llm_status,
            "raw_status": raw_status,
            "compliant": a.compliant,
            "component_declaration": a.component_declaration,
            "cited_refs": a.cited_refs,
            "total_citation_count": a.total_citation_count,
            "grounded_citation_count": a.grounded_citation_count,
            "fabricated_citations": a.fabricated_citations,
            "blanket_citation": a.blanket_citation,
            "retrieved_numbers": sorted(retrieved_numbers),
            "component_grounded": g.component_grounded,
            "ungrounded_refs": g.ungrounded_refs,
            "all_grounded": g.all_grounded,
        })

    return cases


def _aggregate(cases: list[dict]) -> dict[str, Any]:
    """Compute exact-count aggregate statistics over a list of attribution cases."""
    n = len(cases)
    compliant_count = sum(1 for c in cases if c["compliant"])

    raw_status_counts = {"absent": 0, "malformed": 0, "wellformed": 0, "unparseable_raw": 0}
    for c in cases:
        raw_status_counts[c["raw_status"]] += 1

    component_declaration_counts = {
        "grounded_declared": 0, "honest_override": 0, "misattributed": 0, "missing": 0,
    }
    for c in cases:
        component_declaration_counts[c["component_declaration"]] += 1

    plans_with_citations = sum(1 for c in cases if c["total_citation_count"] > 0)
    total_citation_count = sum(c["total_citation_count"] for c in cases)
    grounded_citation_count = sum(c["grounded_citation_count"] for c in cases)
    fabricated_citation_count = sum(len(c["fabricated_citations"]) for c in cases)
    grounded_citation_rate = (
        round(grounded_citation_count / total_citation_count, 4)
        if total_citation_count
        else None
    )
    blanket_citation_count = sum(1 for c in cases if c["blanket_citation"])

    citations_per_plan = sorted(c["total_citation_count"] for c in cases)
    citations_per_plan_distribution = {
        "min": citations_per_plan[0] if citations_per_plan else 0,
        "median": statistics.median(citations_per_plan) if citations_per_plan else 0,
        "max": citations_per_plan[-1] if citations_per_plan else 0,
    }

    llm_status_counts: dict[str, int] = {}
    for c in cases:
        llm_status_counts[c["llm_status"]] = llm_status_counts.get(c["llm_status"], 0) + 1

    component_ungrounded_count = sum(1 for c in cases if not c["component_grounded"])
    similar_issue_ungrounded_count = sum(1 for c in cases if c["ungrounded_refs"])
    all_grounded_false_count = sum(1 for c in cases if not c["all_grounded"])

    return {
        "n": n,
        "compliant_count": compliant_count,
        "raw_status_counts": raw_status_counts,
        "component_declaration_counts": component_declaration_counts,
        "plans_with_citations": plans_with_citations,
        "total_citation_count": total_citation_count,
        "grounded_citation_count": grounded_citation_count,
        "fabricated_citation_count": fabricated_citation_count,
        "grounded_citation_rate": grounded_citation_rate,
        "blanket_citation_count": blanket_citation_count,
        "citations_per_plan_distribution": citations_per_plan_distribution,
        "llm_status_counts": llm_status_counts,
        "continuity": {
            "component_ungrounded_count": component_ungrounded_count,
            "similar_issue_ungrounded_count": similar_issue_ungrounded_count,
            "all_grounded_false_count": all_grounded_false_count,
        },
    }


def measure() -> dict[str, Any]:
    """Run cassette-only replay over the full eval set and compute attribution-fidelity stats.

    Returns a dict with per_repo breakdown, overall totals, and the full list of
    fabricated-citation cases with enough detail to inspect (not just counts).
    """
    cases = compute_attribution_reports(EVAL_SET_PATH, CASSETTE_PATH)

    per_repo_cases: dict[str, list[dict]] = {repo: [] for repo in REPO_MAP}
    fabricated_citation_cases: list[dict] = []
    for case in cases:
        per_repo_cases[case["repo"]].append(case)
        if case["fabricated_citations"]:
            fabricated_citation_cases.append({
                "issue_number": case["issue_number"],
                "repo": case["repo"],
                "fabricated": case["fabricated_citations"],
                "retrieved_numbers": case["retrieved_numbers"],
            })

    per_repo = {repo: _aggregate(repo_cases) for repo, repo_cases in per_repo_cases.items()}
    overall = _aggregate(cases)

    return {
        "per_repo": per_repo,
        "overall": overall,
        "fabricated_citation_cases": fabricated_citation_cases,
    }


def _print_report(result: dict[str, Any]) -> None:
    """Print a human-readable attribution-fidelity report to stdout."""
    print("\n=== ATTRIBUTION FIDELITY MEASUREMENT ===\n")

    for repo, data in result["per_repo"].items():
        print(f"  {repo}: n={data['n']}")
        print(f"    compliant (well-formed declared_attribution): {data['compliant_count']}")
        print(f"    raw status counts: {data['raw_status_counts']}")
        print(f"    component declaration counts: {data['component_declaration_counts']}")
        print(
            f"    plans with >=1 citation: {data['plans_with_citations']} "
            f"(total citations: {data['total_citation_count']}, "
            f"grounded: {data['grounded_citation_count']}, "
            f"fabricated: {data['fabricated_citation_count']})"
        )
        print(f"    grounded_citation_rate: {data['grounded_citation_rate']}")
        print(f"    blanket_citation_count: {data['blanket_citation_count']}")
        print(f"    citations-per-plan distribution: {data['citations_per_plan_distribution']}")
        print(f"    llm_status counts: {data['llm_status_counts']}")
        print(f"    continuity (verify_plan_grounding): {data['continuity']}")
        print()

    overall = result["overall"]
    print(f"  OVERALL: n={overall['n']}")
    print(f"    compliant (well-formed declared_attribution): {overall['compliant_count']}")
    print(f"    raw status counts: {overall['raw_status_counts']}")
    print(f"    component declaration counts: {overall['component_declaration_counts']}")
    print(
        f"    plans with >=1 citation: {overall['plans_with_citations']} "
        f"(total citations: {overall['total_citation_count']}, "
        f"grounded: {overall['grounded_citation_count']}, "
        f"fabricated: {overall['fabricated_citation_count']})"
    )
    print(f"    grounded_citation_rate: {overall['grounded_citation_rate']}")
    print(f"    blanket_citation_count: {overall['blanket_citation_count']}")
    print(f"    citations-per-plan distribution: {overall['citations_per_plan_distribution']}")
    print(f"    llm_status counts: {overall['llm_status_counts']}")
    print(f"    continuity (verify_plan_grounding): {overall['continuity']}")

    print("\n  --- fabricated citation case detail ---")
    if not result["fabricated_citation_cases"]:
        print("  (none)")
    for case in result["fabricated_citation_cases"]:
        print(
            f"  #{case['issue_number']} ({case['repo']}): "
            f"fabricated={case['fabricated']}, retrieved_numbers={case['retrieved_numbers']}"
        )


def main() -> None:
    """CLI entry point: run measurement, print report, write reports/attribution_fidelity.json."""
    result = measure()
    _print_report(result)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nStructured report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
