from __future__ import annotations
"""Eval-only frozen retriever — duck-types SimilarIssueRetriever.retrieve().

Used by run_eval.py and record_cassettes.py so the synthesis prompt is a
deterministic function of committed inputs regardless of hardware.

Production /triage is UNCHANGED — it uses live SimilarIssueRetriever.
"""

import json
from pathlib import Path

import numpy as np

# Frozen query embeddings (ADR-0062): the vector SimilarIssueRetriever.retrieve_with_embedding()
# computes for each eval issue, frozen on CPU float32 by eval/freeze_query_embeddings.py so the
# resolution stage's emb_* features -- and therefore the prompt text -- are deterministic on any
# hardware, exactly like the frozen top-k hits. Only repos in triage.RESOLUTION_EMBEDDINGS_REPOS
# have entries. Lives beside the eval set; optional so eval sets without it still load.
QUERY_EMBEDDINGS_FILENAME = "frozen_query_embeddings.npz"


class FrozenRetriever:
    """Returns pre-frozen top-k similar issues keyed by issue number.

    Signature matches SimilarIssueRetriever.retrieve() exactly:
        retrieve(query_text: str, k: int = 20, exclude_number: int | None = None)
            -> list[dict]
    query_text is ignored — results are keyed by exclude_number (the issue
    being triaged). k is used for slicing if frozen list is longer than k.
    """

    def __init__(
        self,
        frozen_by_number: dict[int, list[dict]],
        query_embeddings_by_number: dict[int, np.ndarray] | None = None,
    ) -> None:
        self._frozen = frozen_by_number
        self._query_embeddings = query_embeddings_by_number or {}

    def retrieve(
        self,
        query_text: str,
        k: int = 20,
        exclude_number: int | None = None,
    ) -> list[dict]:
        return self._frozen.get(exclude_number, [])[:k]

    def retrieve_with_embedding(
        self,
        query_text: str,
        k: int = 20,
        exclude_number: int | None = None,
    ) -> tuple[list[dict], np.ndarray | None]:
        """retrieve() plus the frozen query embedding, or None if none was frozen for this issue
        (the pipeline then falls back to zero-filled emb_* with a warning)."""
        hits = self.retrieve(query_text, k=k, exclude_number=exclude_number)
        emb = self._query_embeddings.get(exclude_number) if exclude_number is not None else None
        return hits, emb


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

    emb_path = Path(eval_set_path).parent / QUERY_EMBEDDINGS_FILENAME
    emb_by_repo: dict[str, dict[int, np.ndarray]] = {}
    if emb_path.exists():
        with np.load(emb_path, allow_pickle=False) as z:
            rows = zip(z["repos"].tolist(), z["numbers"].tolist(), z["embeddings"], strict=True)
            for r, n, e in rows:
                emb_by_repo.setdefault(r, {})[int(n)] = np.asarray(e, dtype=np.float32)

    return {
        repo: FrozenRetriever(frozen, emb_by_repo.get(repo))
        for repo, frozen in by_repo.items()
    }
