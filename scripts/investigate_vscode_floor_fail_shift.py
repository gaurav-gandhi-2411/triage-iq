"""Investigate vscode's floor_fail_rate shift (45%->64%, v3 -> combined-recording) after
Cutover A's corpus-truncation fix (ADR-0040). Only vscode's retrieval index changed between
these two recordings -- classifier (ADR-0036) and resolution model/naive-fallback are both
unchanged for vscode. Replays BOTH cassettes per-issue (offline, no live LLM calls) and reports,
per vscode issue: floor-fail status in each recording, judge rationale, and the actual
similar_issues retrieved in each -- so "did retrieval get better or worse for THESE issues" can
be answered directly instead of inferred from the aggregate R@5 delta.

Usage:
    python scripts/investigate_vscode_floor_fail_shift.py <path-to-NEW-cassette.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import pandas as pd  # noqa: E402

from cassette import CassettePlayer  # noqa: E402
from frozen_retriever import build_frozen_retrievers  # noqa: E402
from run_eval import (  # noqa: E402
    CASSETTE_PATH,
    CI_API_KEY,
    EVAL_SET_PATH,
    JUDGE_MODEL,
    JUDGE_PROVIDER,
    REPO_MAP,
    _load_eval_set,
    _load_models,
)
from triage_iq.evaluation.triage_eval import TriageJudge  # noqa: E402

REPO = "microsoft/vscode"


def replay(cassette_path: Path, eval_set_path: Path) -> dict[int, dict]:
    """eval_set_path matters: FrozenRetriever reads similar_issues from THIS file, not from
    the cassette -- v3 must be replayed against the PRE-Cutover-A eval_set.jsonl (the frozen
    snapshot v3 actually saw at record time), never the current one, or this would silently
    compare v3's synthesis/judge output against the NEW index's retrieval instead of its own."""
    cassette = CassettePlayer(cassette_path, strict=True)
    issues = _load_eval_set(eval_set_path)
    frozen_retrievers = build_frozen_retrievers(eval_set_path)

    models: dict[str, dict] = {}
    for repo, slug in REPO_MAP.items():
        models[repo] = _load_models(repo, slug, cassette, frozen_retrievers)

    judge = TriageJudge(
        groq_api_key=CI_API_KEY,
        model=JUDGE_MODEL,
        provider=JUDGE_PROVIDER,
        temperature=0.0,
        ollama_seed=42,
        cache=cassette,
    )

    per_issue: dict[int, dict] = {}
    for issue in issues:
        repo = issue["repo"]
        if repo != REPO:
            continue
        assistant = models[repo]["assistant"]
        row = pd.Series(
            {
                "title": issue["title"],
                "body_clean": issue["body"],
                "number": issue["number"],
                "created_at": (
                    pd.Timestamp(issue["created_at"])
                    if issue.get("created_at")
                    else pd.Timestamp("now", tz="UTC")
                ),
            }
        )
        plan, _meta = assistant.triage_with_metadata(row)
        gold = {
            "component": issue["gold_component"],
            "priority": issue["gold_priority"],
            "actual_resolution_days": issue["actual_resolution_days"],
        }
        score = judge.score(
            issue_title=issue["title"],
            issue_body=issue["body"][:600],
            triage_plan_json=json.dumps(
                plan.model_dump(exclude={"declared_attribution", "abstention_status"}),
                ensure_ascii=False,
            ),
            gold=gold,
        )
        floor_fail = score.component_match == 0 or score.similar_issues_relevance == 0
        per_issue[issue["number"]] = {
            "title": issue["title"][:70],
            "predicted_component": plan.predicted_component,
            "gold_component": issue["gold_component"],
            "component_match": score.component_match,
            "similar_issues_relevance": score.similar_issues_relevance,
            "floor_fail": floor_fail,
            "judge_rationale": score.judge_rationale,
            "similar_issues": [(s.number, round(s.similarity, 3)) for s in plan.similar_issues],
        }
    return per_issue


def main() -> None:
    new_cassette = Path(sys.argv[1])
    old_eval_set_path = Path(sys.argv[2])
    print("Replaying v3 (currently committed cassette, PRE-Cutover-A eval_set.jsonl)...")
    v3 = replay(CASSETTE_PATH, old_eval_set_path)
    print("Replaying NEW (combined recording, downloaded artifact, current eval_set.jsonl)...")
    new = replay(new_cassette, EVAL_SET_PATH)

    print(f"\nv3 vscode floor-fails: {sum(1 for v in v3.values() if v['floor_fail'])}/{len(v3)}")
    print(f"NEW vscode floor-fails: {sum(1 for v in new.values() if v['floor_fail'])}/{len(new)}\n")

    for num in sorted(v3):
        a, b = v3[num], new.get(num)
        if b is None:
            print(f"#{num}: missing from NEW recording")
            continue
        changed = a["floor_fail"] != b["floor_fail"]
        marker = " <<< CHANGED" if changed else ""
        print(f"#{num} {a['title']!r}{marker}")
        print(
            f"  v3 : floor_fail={a['floor_fail']} component_match={a['component_match']} "
            f"similar_issues_relevance={a['similar_issues_relevance']} "
            f"predicted={a['predicted_component']!r} gold={a['gold_component']!r}"
        )
        print(f"       similar_issues={a['similar_issues']}")
        print(f"       rationale: {a['judge_rationale']!r}")
        print(
            f"  NEW: floor_fail={b['floor_fail']} component_match={b['component_match']} "
            f"similar_issues_relevance={b['similar_issues_relevance']} "
            f"predicted={b['predicted_component']!r} gold={b['gold_component']!r}"
        )
        print(f"       similar_issues={b['similar_issues']}")
        print(f"       rationale: {b['judge_rationale']!r}")
        print()


if __name__ == "__main__":
    main()
