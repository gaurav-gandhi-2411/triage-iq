"""One-off investigation script (2026-08-11 session, not wired into CI/eval-gate).

Reproduces the k8s_related R@5 eval against the PRODUCTION-served index
(data/models/dup_index_kubernetes_kubernetes_bge -- confirmed byte-identical to the
verified _candidate/ used to measure 24.67%, see ADR-0040), and for every pair:
  - records hit/miss at k=5
  - records whether target_number is present in the live index at all (coverage check)
  - records the query title/body, target title/body, and the actual top-5 retrieved
    titles, so misses can be hand-read and categorized without re-running anything.

Writes reports/track2_k8s_miss_analysis.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from triage_iq.models.similar_issues import SimilarIssueRetriever  # noqa: E402

REPORTS = Path("reports")
MODELS_DIR = Path("data/models")
K = 5
K_RETRIEVE = 20


def query_text(row: dict) -> str:
    return f"{row['query_title']}. {row.get('query_body', '')}"


def main() -> None:
    pairs = json.loads((REPORTS / "d1_eval_set_k8s_related.json").read_text(encoding="utf-8"))
    print(f"Loaded {len(pairs)} k8s_related eval pairs")

    print("Loading PRODUCTION index (dup_index_kubernetes_kubernetes_bge)...")
    detector = SimilarIssueRetriever.load(str(MODELS_DIR / "dup_index_kubernetes_kubernetes_bge"))
    live_numbers = {int(n) for n in detector.issue_numbers}
    print(f"Live index has {len(live_numbers)} issues")

    # Build a lookup from issue_number -> title, for target-side titles not in the
    # eval-set row itself and for annotating retrieved-neighbor titles.
    issues_df_path = Path("data/processed/issues_kubernetes_kubernetes.parquet")
    import pandas as pd

    issues_df = pd.read_parquet(issues_df_path, columns=["number", "title"])
    title_by_number = dict(zip(issues_df["number"].astype(int), issues_df["title"], strict=True))

    results = []
    n_target_missing = 0
    n_query_missing = 0
    for row in pairs:
        qn = int(row["query_number"])
        tn = int(row["original_number"])
        target_in_index = tn in live_numbers
        query_in_index = qn in live_numbers
        if not target_in_index:
            n_target_missing += 1
        if not query_in_index:
            n_query_missing += 1

        if not query_in_index:
            # Can't even issue the query against this index -- record and skip retrieval.
            results.append(
                {
                    "query_number": qn,
                    "target_number": tn,
                    "query_title": row["query_title"],
                    "query_body": row.get("query_body", ""),
                    "target_title": row.get("original_title") or title_by_number.get(tn, "?"),
                    "channel": row.get("channel"),
                    "source": row.get("source"),
                    "target_in_index": target_in_index,
                    "query_in_index": query_in_index,
                    "hit_at_5": None,
                    "retrieved_top5": None,
                    "target_rank": None,
                }
            )
            continue

        retrieved = detector.retrieve(query_text(row), k=K_RETRIEVE, exclude_number=qn)
        retrieved_numbers = [r["number"] for r in retrieved]
        hit = tn in retrieved_numbers[:K]
        try:
            rank = retrieved_numbers.index(tn) + 1  # 1-indexed; None if beyond K_RETRIEVE
        except ValueError:
            rank = None

        results.append(
            {
                "query_number": qn,
                "target_number": tn,
                "query_title": row["query_title"],
                "query_body": row.get("query_body", ""),
                "target_title": row.get("original_title") or title_by_number.get(tn, "?"),
                "channel": row.get("channel"),
                "source": row.get("source"),
                "target_in_index": target_in_index,
                "query_in_index": query_in_index,
                "hit_at_5": hit,
                "retrieved_top5": [
                    {"number": r["number"], "title": title_by_number.get(r["number"], "?"), "score": r["score"]}
                    for r in retrieved[:K]
                ],
                "target_rank": rank,
            }
        )

    n_evaluated = sum(1 for r in results if r["hit_at_5"] is not None)
    n_hits = sum(1 for r in results if r["hit_at_5"] is True)
    misses = [r for r in results if r["hit_at_5"] is False]

    summary = {
        "n_pairs_total": len(pairs),
        "n_query_missing_from_index": n_query_missing,
        "n_target_missing_from_index": n_target_missing,
        "n_evaluated": n_evaluated,
        "n_hits": n_hits,
        "recall_at_5": n_hits / n_evaluated if n_evaluated else None,
        "n_misses": len(misses),
    }
    print(json.dumps(summary, indent=2))

    out = {"summary": summary, "results": results}
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "track2_k8s_miss_analysis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print("Wrote reports/track2_k8s_miss_analysis.json")


if __name__ == "__main__":
    main()
