"""D3 leakage guard: same discipline as scripts/d2_assert_leakage_guard.py, pointed at the
expanded mining-precision training pools instead of D1's original pools. Eval sets are UNCHANGED
-- reused as-is from D1 (reports/d1_eval_set_{task}.json) per GG's explicit instruction: the held-
out clean eval sets are never rebuilt, only the training side is expanded.

Reads:  reports/mining_precision_train_pool_{task}.json
        reports/d1_eval_set_{task}.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPORTS = Path("reports")

TASKS = {
    "vscode_duplicate": (
        "mining_precision_train_pool_vscode_duplicate.json",
        "d1_eval_set_vscode_duplicate.json",
    ),
    "k8s_related": (
        "mining_precision_train_pool_k8s_related.json",
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
            f"REFUSING to proceed."
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
