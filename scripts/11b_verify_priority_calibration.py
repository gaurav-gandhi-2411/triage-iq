"""Priority-calibration verification script — final iteration.

Tests 2 gold issues + 1 sanity-check issue against the live /triage endpoint.

Gold cases (PR-2 failures, expected after prompt fix):
  #567  high → low    (processes hanging — resource-leak framing)
  #3826 high → medium (C/gdb debugging — niche workflow)

Sanity check (should still be medium — no regression):
  terminal cursor blink (synthetic title, no issue number)

Skipped: #2093 — image-only body stripped to empty; data issue, not prompt issue.

Token budget: ~3,000/call × 3 calls = ~9,000 tokens (well under 12K TPD limit).

Usage:
    python scripts/11b_verify_priority_calibration.py
"""

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://triageiq-api-779563952988.us-central1.run.app"
GOLD_FILE = Path("data/triage_eval_checkpoint.jsonl")
OUT_FILE = Path("reports/priority_calibration_verification.json")

TARGET_ISSUES = {
    567:  {"expected": "low",    "was": "high"},
    3826: {"expected": "medium", "was": "high"},
}

SANITY_CHECK = {
    "title": "Terminal: cursor blink stops after switching tabs",
    "body": (
        "After switching between editor tabs multiple times, the integrated terminal "
        "cursor stops blinking. Reopening the terminal panel fixes it. Reproducible on 1.87.0."
    ),
    "expected": "medium",
}


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def load_issues() -> list[dict]:
    issues = []
    with GOLD_FILE.open() as f:
        for line in f:
            r = json.loads(line)
            if r["repo"] == "microsoft/vscode" and r["issue_number"] in TARGET_ISSUES:
                issues.append(r)
    issues.sort(key=lambda x: x["issue_number"])
    return issues


def call_triage(title: str, body: str, issue_number: int | None = None) -> dict:
    import urllib.request
    payload = json.dumps({
        "repo": "microsoft/vscode",
        "issue_number": issue_number or 0,
        "title": title,
        "body": body,
    }).encode()
    req = urllib.request.Request(
        f"{REPO_URL}/triage",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def main() -> None:
    issues = load_issues()
    if len(issues) != 2:
        print(f"ERROR: expected 2 gold issues in checkpoint, found {len(issues)}")
        sys.exit(1)

    results = []
    total_tokens = 0
    passes = 0
    total_cases = 3  # 2 gold + 1 sanity

    print(f"Commit: {git_sha()}")
    print(f"Endpoint: {REPO_URL}")
    print(f"Started: {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # --- Gold cases ---
    for issue in issues:
        num = issue["issue_number"]
        expected = TARGET_ISSUES[num]["expected"]
        was = TARGET_ISSUES[num]["was"]

        print(f"\n--- Issue #{num} (gold) ---")
        print(f"Title:          {issue['issue_title']}")
        print(f"Gold priority:  {issue['gold_priority']}")
        print(f"Pre-fix was:    {was}")
        print(f"Expected after: {expected}")

        try:
            resp = call_triage(
                title=issue["issue_title"],
                body=issue["issue_body"],
                issue_number=num,
            )
        except Exception as exc:
            print(f"ERROR calling /triage: {exc}")
            sys.exit(1)

        predicted = resp.get("priority_guess", "MISSING")
        rationale = resp.get("priority_rationale", "")
        component = resp.get("predicted_component", "")
        tokens_prompt = resp.get("groq_tokens_prompt", 0) or 0
        tokens_comp = resp.get("groq_tokens_completion", 0) or 0
        call_tokens = tokens_prompt + tokens_comp if (tokens_prompt or tokens_comp) else 3000
        total_tokens += call_tokens

        passed = predicted == expected
        if passed:
            passes += 1

        print(f"Predicted:      {predicted}  ({'PASS' if passed else 'FAIL'})")
        print(f"Component:      {component}")
        print(f"Rationale:      {rationale}")
        print(f"Tokens:         {call_tokens} (prompt={tokens_prompt}, completion={tokens_comp})")

        results.append({
            "case_type": "gold",
            "issue_number": num,
            "title": issue["issue_title"],
            "gold_priority": issue["gold_priority"],
            "gold_component": issue["gold_component"],
            "pre_fix_predicted": was,
            "expected_post_fix": expected,
            "actual_predicted": predicted,
            "predicted_component": component,
            "priority_rationale": rationale,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_comp,
            "pass": passed,
        })

        if total_tokens > 15000:
            print(f"\nSTOP: token budget exceeded ({total_tokens} > 15,000). Aborting.")
            sys.exit(1)

        time.sleep(1)

    # --- Sanity check ---
    print(f"\n--- Sanity check (terminal cursor blink) ---")
    print(f"Title:          {SANITY_CHECK['title']}")
    print(f"Expected:       {SANITY_CHECK['expected']} (regression check — should not change)")

    try:
        resp = call_triage(
            title=SANITY_CHECK["title"],
            body=SANITY_CHECK["body"],
        )
    except Exception as exc:
        print(f"ERROR calling /triage: {exc}")
        sys.exit(1)

    sc_predicted = resp.get("priority_guess", "MISSING")
    sc_rationale = resp.get("priority_rationale", "")
    sc_component = resp.get("predicted_component", "")
    tokens_prompt = resp.get("groq_tokens_prompt", 0) or 0
    tokens_comp = resp.get("groq_tokens_completion", 0) or 0
    call_tokens = tokens_prompt + tokens_comp if (tokens_prompt or tokens_comp) else 3000
    total_tokens += call_tokens

    sc_passed = sc_predicted == SANITY_CHECK["expected"]
    if sc_passed:
        passes += 1

    print(f"Predicted:      {sc_predicted}  ({'PASS' if sc_passed else 'REGRESSION'})")
    print(f"Component:      {sc_component}")
    print(f"Rationale:      {sc_rationale}")
    print(f"Tokens:         {call_tokens} (prompt={tokens_prompt}, completion={tokens_comp})")

    results.append({
        "case_type": "sanity",
        "issue_number": None,
        "title": SANITY_CHECK["title"],
        "expected_post_fix": SANITY_CHECK["expected"],
        "actual_predicted": sc_predicted,
        "predicted_component": sc_component,
        "priority_rationale": sc_rationale,
        "tokens_prompt": tokens_prompt,
        "tokens_completion": tokens_comp,
        "pass": sc_passed,
    })

    print("\n" + "=" * 70)
    print(f"SUMMARY: {passes}/{total_cases} cases correct (2 gold + 1 sanity)")
    print(f"Total tokens used: ~{total_tokens}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "commit_sha": git_sha(),
        "endpoint": REPO_URL,
        "passes": passes,
        "total": total_cases,
        "total_tokens_estimated": total_tokens,
        "note_skipped_2093": "image-only body stripped to empty — data issue, not prompt issue",
        "cases": results,
    }
    OUT_FILE.write_text(json.dumps(output, indent=2))
    print(f"Results saved to {OUT_FILE}")

    gold_passes = sum(1 for r in results if r["case_type"] == "gold" and r["pass"])
    if gold_passes < 2:
        print(f"\nFAILED: {gold_passes}/2 gold cases passed. Review rationales above.")
        print("Hard stop — no further prompt iterations. Document as known limitation.")
        sys.exit(1)
    if not sc_passed:
        print(f"\nWARNING: sanity check regressed (terminal cursor blink → {sc_predicted}).")
        print("Gold cases passed but medium-priority calibration may have shifted.")


if __name__ == "__main__":
    main()
