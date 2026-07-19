"""D2 leakage guard: re-assert train/eval issue-level disjointness before train AND before eval.

D1 (ADR-0033) already asserted this once at data-build time (scripts/d1_build_eval_set.py).
D2 re-derives and re-asserts it independently, at the point of use, because a trained model
evaluated on any training issue is a contaminated fake number (the bug class this project has
caught 5x -- ADR-0018, ADR-0027, ADR-0028, ADR-0030, ADR-0032). Disjointness is checked at the
ISSUE level (query_number/original_number appearing on EITHER side), not just the pair level --
a pair-level check alone would miss issue X appearing in a training pair with Y while being
evaluated in a pair with Z.

Importable (call assert_task_disjoint() / assert_all()) or runnable standalone as a pre-flight
gate. Fails hard (raises SystemExit) on any violation -- this is a hard gate, not a warning.

Reads:  reports/d1_train_pool_{task}.json
        reports/d1_eval_set_{task}.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORTS = Path("reports")

# task -> (train pool file, eval set file). vscode_related has no training pool (eval-only,
# directional, per ADR-0033) so it is not included here -- there is nothing to leak.
TASKS = {
    "vscode_duplicate": (
        "d1_train_pool_vscode_duplicate.json",
        "d1_eval_set_vscode_duplicate.json",
    ),
    "k8s_related": (
        "d1_train_pool_k8s_related.json",
        "d1_eval_set_k8s_related.json",
    ),
}


def _issue_set(pairs: list[dict]) -> set[int]:
    s: set[int] = set()
    for p in pairs:
        s.add(int(p["query_number"]))
        s.add(int(p["original_number"]))
    return s


def assert_task_disjoint(task: str) -> dict:
    train_file, eval_file = TASKS[task]
    train_pairs = json.loads((REPORTS / train_file).read_text(encoding="utf-8"))
    eval_pairs = json.loads((REPORTS / eval_file).read_text(encoding="utf-8"))

    train_issues = _issue_set(train_pairs)
    eval_issues = _issue_set(eval_pairs)
    overlap = train_issues & eval_issues

    if overlap:
        raise SystemExit(
            f"LEAKAGE GUARD FAILED [{task}]: {len(overlap)} issue(s) appear on both sides "
            f"(train pool has {len(train_pairs)} pairs / {len(train_issues)} issues, "
            f"eval set has {len(eval_pairs)} pairs / {len(eval_issues)} issues). "
            f"Overlapping issues (up to 20 shown): {sorted(overlap)[:20]}. "
            f"REFUSING to proceed -- a trained model evaluated on any training issue is a "
            f"contaminated fake number."
        )

    return {
        "task": task,
        "train_pairs": len(train_pairs),
        "train_issues": len(train_issues),
        "eval_pairs": len(eval_pairs),
        "eval_issues": len(eval_issues),
        "overlap": 0,
        "status": "DISJOINT",
    }


def assert_all() -> list[dict]:
    return [assert_task_disjoint(task) for task in TASKS]


def main() -> None:
    results = assert_all()
    for r in results:
        print(
            f"[{r['task']}] DISJOINT -- train={r['train_pairs']} pairs/{r['train_issues']} "
            f"issues, eval={r['eval_pairs']} pairs/{r['eval_issues']} issues, overlap=0"
        )
    print("\nLeakage guard PASSED for all tasks.")


if __name__ == "__main__":
    sys.exit(main())
