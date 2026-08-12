"""Repo-wide ADR citation sweep: walks every ADR's backtick-quoted repo-relative paths and
reports which ones don't resolve on the current tree.

Found by hand this session (2026-08-12): ADR-0047 cited `docs/investigations/2026-08-11-mining-
precision-channel-characterization.md` as its evidentiary source, but that file (and its 4
sibling report/script artifacts) had never been merged to main -- a shipped ADR pointing at a
nonexistent evidence path, discovered only because someone happened to go looking. This script
makes that check cheap to re-run instead of relying on it being noticed by accident.

Three categories of "missing" are expected and NOT failures -- auto-detected where possible:
  1. Gitignored on purpose (`git check-ignore` succeeds) -- a deliberate repo-hygiene decision,
     not an oversight (e.g. ADR-0018's `reports/resolution_results.json`).
  2. Removed by a documented refactor -- the path's last touch on `main` is a delete commit whose
     message cites an ADR number (e.g. ADR-0008's duplicate-detection files, removed when the
     commit reframing to similar-issue retrieval cited ADR-0008 itself).
  3. Lives on an unmerged branch tied to a closed (not merged) PR -- code from a rejected
     experiment that was correctly never shipped (e.g. ADR-0006's reranker.py on
     `feat/w1.3-cross-encoder-reranker`, PR #1, closed unmerged).

Anything that doesn't auto-classify into one of those AND isn't in ALLOWLIST below is a real,
unexplained gap -- report it and (with --strict) exit non-zero. ALLOWLIST covers cases the
auto-classifier can't determine from git alone: files that were simply never committed anywhere
(e.g. ADR-0032's `data/w3_hard_negatives_v2.parquet`, confirmed via `git log --all` to have zero
history on any ref) and this script's own regex false positives (a shorthand path range like
`batch_1..15.json`, or a hypothetical filename mentioned in a forward-looking action item that
was never actually used).

Not CI-gating (by design, per GG's instruction) -- a script to run periodically, not a merge
blocker. Reproduce: python scripts/check_adr_citations.py [--strict]
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ADR_DIR = Path("docs/architecture/adr")
REPO_ROOT_DIRS = ("docs", "reports", "scripts", "data", "src", "tests", "evals")
PATH_PATTERN = re.compile(rf"`((?:{'|'.join(REPO_ROOT_DIRS)})/[A-Za-z0-9_./\-]+\.[a-zA-Z0-9]+)`")
ADR_CITATION_PATTERN = re.compile(r"ADR-\d{4}", re.IGNORECASE)

# Paths this script's regex/heuristics can't correctly classify on their own -- each entry
# requires a "reason" naming which of the two non-git-derivable cases applies and why. Mirrors
# merge_gate.py's OVERRIDE_ALLOWLIST/SENSITIVE_PATH_ALLOWLIST pattern: explicit per-path entries
# only, no globs, so adding one always means naming one exact path and a stated reason.
ALLOWLIST: dict[str, dict[str, str]] = {
    "data/w3_hard_negatives_v2.parquet": {
        "category": "never_committed",
        "reason": (
            "ADR-0032's own text already discloses this as an untracked local artifact, never "
            "committed on any ref (confirmed via `git log --all`). The W3 fine-tune it backed "
            "stayed HELD per ADR-0027, never revived -- annotated in ADR-0032 (2026-08-12 "
            "status update), not backfilled: a fresh regeneration wouldn't be the artifact "
            "that was actually measured at the time."
        ),
    },
    "reports/d3a_pool_labeled_batch_1..15.json": {
        "category": "regex_false_positive",
        "reason": (
            "ADR-0049 shorthand for 15 separate files (batch_1.json .. batch_15.json), all of "
            "which exist individually -- not a literal filename."
        ),
    },
    "reports/classifier_results_multilabel.json": {
        "category": "regex_false_positive",
        "reason": (
            "ADR-0036 mentions this as a hypothetical 'or a new X' alternative name in a "
            "forward-looking action-item list, not a claim the file exists. The actual output "
            "was committed as reports/multilabel_classifier_final_training.json."
        ),
    },
}


def run(args: list[str]) -> tuple[int, str]:
    result = subprocess.run(
        args, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )
    return result.returncode, (result.stdout or "").strip()


def is_gitignored(path: str) -> bool:
    code, _ = run(["git", "check-ignore", "-q", path])
    return code == 0


def deleted_by_documented_refactor(path: str) -> str | None:
    """Returns the citing commit subject if the path's last touch on HEAD's history is a delete
    whose commit message cites an ADR number; else None."""
    code, out = run(["git", "log", "-1", "--diff-filter=D", "--format=%s", "--", path])
    if code != 0 or not out:
        return None
    if ADR_CITATION_PATTERN.search(out):
        return out
    return None


def find_branches_containing(path: str) -> list[str]:
    """Returns non-main branch names (local/remote-tracking dedup'd to short names) whose
    history contains the commit that added `path`, or [] if the path was never committed
    anywhere."""
    code, out = run(["git", "log", "--all", "-1", "--diff-filter=A", "--format=%H", "--", path])
    if code != 0 or not out:
        return []
    commit = out
    code, out = run(["git", "branch", "--all", "--contains", commit])
    branches = [
        b.strip().lstrip("* ").replace("remotes/origin/", "")
        for b in out.splitlines()
        if b.strip() and "HEAD" not in b
    ]
    return sorted({b for b in branches if b not in ("main", "master")})


def rejected_branch_pr_state(branch: str) -> str | None:
    """Returns 'CLOSED' if `gh` finds a closed-not-merged PR for this branch (auto-classifiable
    as a correctly-rejected experiment, per GG's `rejected-branch code` category), else None
    (no PR, an open PR, or `gh`/network unavailable -- all left for a human to judge, never
    auto-passed)."""
    code, out = run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "all",
            "--head",
            branch,
            "--json",
            "state",
            "-q",
            ".[0].state",
        ]
    )
    if code != 0:
        return None
    state = out.strip()
    return state if state == "CLOSED" else None


def lives_on_unmerged_branch(path: str) -> tuple[str, str | None] | None:
    """Returns (branch_summary, closed_pr_branch) if the path exists on some non-main branch,
    else None. `closed_pr_branch` is set only if at least one of those branches has a `gh`-
    confirmed closed-not-merged PR -- the auto-classifiable 'rejected-branch code' case."""
    non_main = find_branches_containing(path)
    if not non_main:
        return None
    summary = ", ".join(non_main[:3]) + (
        f" (+{len(non_main) - 3} more)" if len(non_main) > 3 else ""
    )
    for branch in non_main:
        if rejected_branch_pr_state(branch) == "CLOSED":
            return summary, branch
    return summary, None


def collect_cited_paths() -> dict[str, list[str]]:
    by_path: dict[str, list[str]] = {}
    for adr in sorted(ADR_DIR.glob("*.md")):
        text = adr.read_text(encoding="utf-8", errors="replace")
        for m in PATH_PATTERN.finditer(text):
            by_path.setdefault(m.group(1), []).append(adr.name)
    return by_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true", help="exit 1 if any unexplained gap remains")
    args = ap.parse_args()

    by_path = collect_cited_paths()
    missing = {p: adrs for p, adrs in by_path.items() if not Path(p).exists()}

    print(f"Scanned {len(list(ADR_DIR.glob('*.md')))} ADRs, {len(by_path)} unique cited paths.")
    print(f"Missing on current tree: {len(missing)}")
    print()

    unexplained: list[str] = []
    for path in sorted(missing):
        adrs = ", ".join(missing[path])
        allow = ALLOWLIST.get(path)
        if allow:
            print(f"[ALLOWLISTED:{allow['category']}] {path}")
            print(f"    cited by: {adrs}")
            print(f"    reason: {allow['reason']}")
            continue

        refactor_commit = deleted_by_documented_refactor(path)
        if refactor_commit:
            print(f"[OK:superseded_by_refactor] {path}")
            print(f"    cited by: {adrs}")
            print(f"    removed by: {refactor_commit}")
            continue

        if is_gitignored(path):
            print(f"[OK:gitignored] {path}")
            print(f"    cited by: {adrs}")
            continue

        found = lives_on_unmerged_branch(path)
        if found:
            summary, closed_pr_branch = found
            if closed_pr_branch:
                print(f"[OK:rejected_branch] {path}")
                print(f"    cited by: {adrs}")
                print(
                    f"    found on: {summary} -- confirmed closed-not-merged PR on "
                    f"'{closed_pr_branch}' (correctly-rejected experiment, never shipped)"
                )
            else:
                print(f"[FLAG:unmerged_branch] {path}")
                print(f"    cited by: {adrs}")
                print(
                    f"    found on: {summary} -- no closed-not-merged PR found (gh unavailable, "
                    f"no PR ever opened, or the PR is still OPEN) -- verify by hand; this may be "
                    f"a live gap like the ADR-0047 incident, not a settled rejection"
                )
                unexplained.append(path)
            continue

        print(f"[GAP] {path}")
        print(f"    cited by: {adrs}")
        print(
            "    not gitignored, not deleted-by-documented-refactor, not found on any branch "
            "-- genuinely unresolvable. Add an ALLOWLIST entry with a reason, or backfill it."
        )
        unexplained.append(path)

    print()
    if unexplained:
        print(f"UNEXPLAINED: {len(unexplained)} path(s) need attention -- see [FLAG]/[GAP] above.")
        if args.strict:
            return 1
    else:
        print("All missing citations are accounted for (allowlisted, gitignored, or superseded).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
