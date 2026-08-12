"""D3a candidate-A test (ADR-0048 follow-up): extract the full 448-pair k8s_related training
pool into blind form (query+target title/body only, no channel/source/precision-estimate field)
for batch labeling under the same pre-registered rubric used to build the 66-pair clean eval
subset (docs/investigations/2026-08-11-k8s-retrieval-ceiling-and-vscode-resolution-close.md,
section A2): VALID / EXCLUDE_UMBRELLA / EXCLUDE_CAUSAL_ONLY / EXCLUDE_OTHER.

Batches of 30, same precedent as the eval-set blind labeling and mining_precision_strict_audit.py.

Reads:  reports/mining_precision_train_pool_k8s_related.json
        data/processed/issues_kubernetes_kubernetes.parquet
Writes: reports/d3a_pool_blind_k8s_related.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPORTS = Path("reports")
BATCH_SIZE = 30


def main() -> None:
    pairs = json.loads(
        (REPORTS / "mining_precision_train_pool_k8s_related.json").read_text(encoding="utf-8")
    )
    corpus = pd.read_parquet(
        "data/processed/issues_kubernetes_kubernetes.parquet",
        columns=["number", "title", "body_clean"],
    )
    body_by_number = dict(zip(corpus["number"].astype(int), corpus["body_clean"], strict=True))
    title_by_number = dict(zip(corpus["number"].astype(int), corpus["title"], strict=True))

    blind = []
    for i, row in enumerate(pairs):
        qn, tn = int(row["query_number"]), int(row["original_number"])
        blind.append(
            {
                "pair_id": i,
                "query_number": qn,
                "target_number": tn,
                "query_title": row.get("query_title") or title_by_number.get(qn, "?"),
                "query_body": (body_by_number.get(qn, "") or "")[:1800],
                "target_title": row.get("original_title") or title_by_number.get(tn, "?"),
                "target_body": (body_by_number.get(tn, "") or "")[:1800],
                "label": None,
                "reason": None,
            }
        )

    for i, row in enumerate(blind):
        row["batch"] = i // BATCH_SIZE + 1

    out_path = REPORTS / "d3a_pool_blind_k8s_related.json"
    out_path.write_text(json.dumps(blind, indent=2, ensure_ascii=False), encoding="utf-8")
    n_batches = blind[-1]["batch"] if blind else 0
    print(f"Wrote {len(blind)} pairs across {n_batches} batches to {out_path}")


if __name__ == "__main__":
    main()
