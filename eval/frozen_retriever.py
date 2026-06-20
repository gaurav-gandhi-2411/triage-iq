from __future__ import annotations
"""Eval-only frozen retriever — duck-types SimilarIssueRetriever.retrieve().

Used by run_eval.py and record_cassettes.py so the synthesis prompt is a
deterministic function of committed inputs regardless of hardware.

Production /triage is UNCHANGED — it uses live SimilarIssueRetriever.
"""

import json
from pathlib import Path


class FrozenRetriever:
    """Returns pre-frozen top-k similar issues keyed by issue number.

    Signature matches SimilarIssueRetriever.retrieve() exactly:
        retrieve(query_text: str, k: int = 20, exclude_number: int | None = None)
            -> list[dict]
    query_text is ignored — results are keyed by exclude_number (the issue
    being triaged). k is used for slicing if frozen list is longer than k.
    """

    def __init__(self, frozen_by_number: dict[int, list[dict]]) -> None:
        self._frozen = frozen_by_number

    def retrieve(
        self,
        query_text: str,
        k: int = 20,
        exclude_number: int | None = None,
    ) -> list[dict]:
        return self._frozen.get(exclude_number, [])[:k]


def build_frozen_retrievers(eval_set_path: Path) -> dict[str, "FrozenRetriever"]:
    """Build one FrozenRetriever per repo from frozen similar_issues in eval_set.jsonl.

    Raises ValueError if any issue is missing the similar_issues field (i.e. the
    freeze step has not been run yet).
    """
    by_repo: dict[str, dict[int, list[dict]]] = {}
    missing: list[str] = []

    with open(eval_set_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            issue = json.loads(line)
            repo = issue["repo"]
            num = int(issue["number"])
            if "similar_issues" not in issue:
                missing.append(issue["id"])
                continue
            by_repo.setdefault(repo, {})[num] = issue["similar_issues"]

    if missing:
        raise ValueError(
            f"{len(missing)} eval issues are missing 'similar_issues' field. "
            "Run eval/freeze_similar_issues.py first."
        )

    return {repo: FrozenRetriever(frozen) for repo, frozen in by_repo.items()}
