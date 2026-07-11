"""Phase 2b: merge legacy gold + Phase 2b candidates into gold_related_v2 with eval strata.

Strata (GG-approved keep-but-stratify design, mirrors the vscode dup/related design):
  k8s:    gate       = PR-query -> issue-target (the powered proxy stratum, CI-gated)
          product    = issue -> issue           (the product task; directional secondary)
          train_only = any PR-TARGET pair       (kept for training, excluded from ALL eval)
  vscode: gate       = dup_comment pairs        (powered proxy stratum, CI-gated)
          product    = non-dup issue -> issue   (directional secondary)
          train_only = PR-query or PR-target pairs

ADR-0018 disjointness is re-asserted over the FULL merged artifact (legacy pairs included —
the original gold was assembled before the judge-eval existed and legacy pairs touching
judge-eval issues become retrieval-train contamination the moment we retrain): violating
pairs are DROPPED and counted, then a zero-overlap assert runs.

Output: data/gold_related_v2.parquet (+ stratum breakdown in stdout/report)
        gold_related.parquet is left untouched (v1 stays the shipped-baseline reference).
Reproduce: python scripts/phase2b_merge_gold_v2.py
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
GOLD_V1 = Path("data/gold_related.parquet")
CANDIDATES = Path("data/gold_related_v2_candidates.parquet")
TRIAGE_GOLD = Path("data/gold_triage_plans.parquet")
OUT = Path("data/gold_related_v2.parquet")
OUT_REPORT = Path("reports/phase2b_merge_report.json")

_pr_cache: dict[tuple[str, int], bool | None] = {}


def is_pr(repo: str, number: int) -> bool | None:
    key = (repo, number)
    if key not in _pr_cache:
        f = RAW[repo] / f"{number}.json"
        _pr_cache[key] = (
            "pull_request" in json.loads(f.read_text(encoding="utf-8")) if f.exists() else None
        )
    return _pr_cache[key]


def assign_stratum(row: pd.Series) -> str:
    repo = row["repo"]
    qp = is_pr(repo, int(row["query_number"]))
    tp = is_pr(repo, int(row["original_number"]))
    if qp is None or tp is None:
        return "train_only"  # unknown PR status -> never eval on it
    if repo == "kubernetes_kubernetes":
        if tp:
            return "train_only"
        return "gate" if qp else "product"
    # vscode
    if qp or tp:
        return "train_only"
    return "gate" if row.get("source") == "dup_comment" else "product"


def main() -> None:
    v1 = pd.read_parquet(GOLD_V1)
    v1["channel"] = "legacy_gold_v1"
    v1["query_is_pr"] = False  # recomputed below via stratum assignment
    cand = pd.read_parquet(CANDIDATES)
    common = [
        "repo",
        "query_number",
        "original_number",
        "query_title",
        "original_title",
        "query_body",
        "original_body",
        "source",
        "confidence",
        "channel",
    ]
    merged = pd.concat([v1[common], cand[common]], ignore_index=True)

    # dedupe unordered pair keys (candidates were already deduped vs v1, belt-and-braces)
    key = merged.apply(
        lambda r: (r["repo"], frozenset((int(r["query_number"]), int(r["original_number"])))),
        axis=1,
    )
    before = len(merged)
    merged = merged[~key.duplicated()].copy()
    log.info("merged %d pairs (%d duplicate keys dropped)", len(merged), before - len(merged))

    merged["stratum"] = merged.apply(assign_stratum, axis=1)

    # ADR-0018: drop ANY pair touching a judge-eval issue (legacy included)
    tg = pd.read_parquet(TRIAGE_GOLD)
    eval_nums = {
        ("kubernetes_kubernetes" if "kubernetes" in r else "microsoft_vscode", int(n))
        for r, n in zip(tg["repo"], tg["number"], strict=True)
    }
    touches = merged.apply(
        lambda r: bool(
            {(r["repo"], int(r["query_number"])), (r["repo"], int(r["original_number"]))}
            & eval_nums
        ),
        axis=1,
    )
    dropped_judge = merged[touches]
    merged = merged[~touches].copy()
    assert not merged.apply(
        lambda r: bool(
            {(r["repo"], int(r["query_number"])), (r["repo"], int(r["original_number"]))}
            & eval_nums
        ),
        axis=1,
    ).any(), "ADR-0018 disjointness violated after drop — bug"
    log.info(
        "judge-eval disjointness: dropped %d pairs (%d legacy)",
        len(dropped_judge),
        int((dropped_judge["channel"] == "legacy_gold_v1").sum()),
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)

    breakdown = merged.groupby(["repo", "stratum"]).size().unstack(fill_value=0).to_dict("index")
    report = {
        "generated_by": "scripts/phase2b_merge_gold_v2.py",
        "totals": {"v1": int(len(v1)), "candidates": int(len(cand)), "merged": int(len(merged))},
        "judge_eval_dropped": {
            "total": int(len(dropped_judge)),
            "legacy_v1": int((dropped_judge["channel"] == "legacy_gold_v1").sum()),
        },
        "strata_by_repo": breakdown,
        "by_repo_channel": merged.groupby(["repo", "channel"])
        .size()
        .unstack(fill_value=0)
        .to_dict("index"),
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2))
    log.info("Wrote %s (%d pairs) and %s", OUT, len(merged), OUT_REPORT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
