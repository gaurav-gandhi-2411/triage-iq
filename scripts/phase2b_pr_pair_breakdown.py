"""Phase 2b decision (c): 4-way PR/issue breakdown of k8s gold pairs.

The k8s gold was built without PR awareness (numbers are shared between issues and PRs;
ADR-0008 framing). GG's question: how much of the k8s retrieval gold measures the PRODUCT
task (issue-query -> issue-target, "a user triaging an ISSUE wants related ISSUES") vs the
proxy task (PR-query -> the issue it fixes)?

Breaks down:
  - existing k8s gold (1,024 pairs), overall / by source / for the W3 test split specifically
    (the 152 pairs behind ADR-0016's +11.84pp)
  - the new k8s candidates (target always an issue by the Phase 2b filter)
  - vscode for completeness (same shared-number space)

PR flags come from the raw JSONs' "pull_request" key (ground truth from the GitHub API).

Output: reports/phase2b_pr_pair_breakdown.json
Reproduce: python scripts/phase2b_pr_pair_breakdown.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

RAW = {
    "kubernetes_kubernetes": Path("data/raw/kubernetes_kubernetes"),
    "microsoft_vscode": Path("data/raw/microsoft_vscode"),
}
OUT = Path("reports/phase2b_pr_pair_breakdown.json")

_cache: dict[tuple[str, int], bool | None] = {}


def is_pr(repo: str, number: int) -> bool | None:
    """True/False from the raw JSON; None if the number was never scraped."""
    key = (repo, number)
    if key not in _cache:
        f = RAW[repo] / f"{number}.json"
        _cache[key] = (
            "pull_request" in json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        )
    return _cache[key]


def four_way(df: pd.DataFrame, repo: str) -> dict:
    counts = {"issue->issue": 0, "PR->issue": 0, "issue->PR": 0, "PR->PR": 0, "unknown": 0}
    for q, t in zip(df["query_number"].astype(int), df["original_number"].astype(int), strict=True):
        qp, tp = is_pr(repo, q), is_pr(repo, t)
        if qp is None or tp is None:
            counts["unknown"] += 1
            continue
        counts[f"{'PR' if qp else 'issue'}->{'PR' if tp else 'issue'}"] += 1
    counts["n"] = int(len(df))
    counts["product_task_pct"] = round(100 * counts["issue->issue"] / max(len(df), 1), 1)
    return counts


def main() -> None:
    gold = pd.read_parquet("data/gold_related.parquet")
    split = pd.read_parquet("data/w3_split.parquet")
    cand = pd.read_parquet("data/gold_related_v2_candidates.parquet")

    report: dict = {"generated_by": "scripts/phase2b_pr_pair_breakdown.py"}
    for repo in ("kubernetes_kubernetes", "microsoft_vscode"):
        g = gold[gold["repo"] == repo]
        entry = {"existing_gold_overall": four_way(g, repo), "existing_gold_by_source": {}}
        for source, gs in g.groupby("source"):
            entry["existing_gold_by_source"][source] = four_way(gs, repo)
        for split_name in ("train", "test"):
            sp = split[(split["repo"] == repo) & (split["split"] == split_name)]
            entry[f"w3_{split_name}_split"] = four_way(sp, repo)
        c = cand[cand["repo"] == repo]
        entry["phase2b_candidates_overall"] = four_way(c, repo)
        for channel, cs in c.groupby("channel"):
            entry[f"phase2b_candidates_{channel}"] = four_way(cs, repo)
        report[repo] = entry

    OUT.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s", OUT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
