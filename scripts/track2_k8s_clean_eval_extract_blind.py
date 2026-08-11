"""One-off investigation script (2026-08-11 session, Track 2 follow-up).

Extracts all 150 k8s_related eval pairs into a form suitable for blind hand-verification:
query + target title/body only, no hit/miss or retrieval-score field anywhere in the output.
This file is what gets handed to reviewers (human or agent) for pair-validity labeling.

Reads:  reports/d1_eval_set_k8s_related.json
        data/processed/issues_kubernetes_kubernetes.parquet
Writes: reports/track2_k8s_150_blind.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORTS = Path("reports")


def main() -> None:
    pairs = json.loads((REPORTS / "d1_eval_set_k8s_related.json").read_text(encoding="utf-8"))
    corpus = pd.read_parquet(
        "data/processed/issues_kubernetes_kubernetes.parquet", columns=["number", "title", "body_clean"]
    )
    body_by_number = dict(zip(corpus["number"].astype(int), corpus["body_clean"], strict=True))
    title_by_number = dict(zip(corpus["number"].astype(int), corpus["title"], strict=True))

    blind = []
    for i, row in enumerate(pairs):
        tn = int(row["original_number"])
        blind.append(
            {
                "pair_id": i,
                "query_number": int(row["query_number"]),
                "target_number": tn,
                "query_title": row["query_title"],
                "query_body": row.get("query_body", "") or "",
                "target_title": row.get("original_title") or title_by_number.get(tn, "?"),
                "target_body": (body_by_number.get(tn, "") or "")[:1500],
            }
        )

    out_path = REPORTS / "track2_k8s_150_blind.json"
    out_path.write_text(json.dumps(blind, indent=2), encoding="utf-8")
    print(f"Wrote {len(blind)} blind pairs to {out_path}")


if __name__ == "__main__":
    main()
