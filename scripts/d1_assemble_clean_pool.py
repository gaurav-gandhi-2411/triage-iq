"""D1 checkpoint 1: assemble the clean product-task training pool from all precision evidence.

Combines three evidence sources, all cited with provenance (rule 65b):
  1. Fresh hand-judged samples this phase (reports/d1_pair_quality_review.json ->
     reports/d1_channel_precision_audit.json) -- k8s_forward_scrape (both sources),
     the tiny legacy body_ref/body_related full-census pools, vscode dup_comment.
  2. ADR-0032's existing 50-pair review, resliced by `source` (not previously reported
     at this granularity) -- k8s_extended_mine/body_related_ext (n=20), legacy body_related
     (n=3), legacy title_sim (n=2, too small to trust, superseded by rule 3).
  3. ADR-0030's channel table (already measured, cited not re-derived) -- vscode's
     "Channel A" (body_related_ext / `vscode_body_refs`) at 30-43% precision (n=30) is
     NOISY, same tier as title_sim -- excluded despite being non-title_sim.

Hard rule (spec.md): title_sim is dropped everywhere, unconditionally, regardless of any
per-repo sample precision -- this is a project-wide decision from ADR-0032, not re-litigated
per repo here.

Inclusion bar: precision comfortably above the noise tier (title_sim ~20%, vscode Channel A
30-43%). All included channels clear ~65%+; most are 80-91%. Every included channel's
precision is stated, not just a pass/fail label (spec.md success criteria).

Reads:  data/gold_related_v2.parquet
        reports/d1_channel_precision_audit.json  (this phase's fresh samples)
        reports/phaseC_pair_quality_review.json  (ADR-0032, resliced by source)
Writes: reports/d1_clean_pool_checkpoint1.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

GOLD_PATH = Path("data/gold_related_v2.parquet")
FRESH_AUDIT = Path("reports/d1_channel_precision_audit.json")
ADR0032_REVIEW = Path("reports/phaseC_pair_quality_review.json")
OUT = Path("reports/d1_clean_pool_checkpoint1.json")


def wilson_ci(x: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = x / n
    denom = 1 + z**2 / n
    center = p + z**2 / (2 * n)
    adj = z * ((p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5)
    return (round((center - adj) / denom, 4), round((center + adj) / denom, 4))


def reslice_adr0032(gold: pd.DataFrame) -> pd.DataFrame:
    review = pd.DataFrame(json.loads(ADR0032_REVIEW.read_text(encoding="utf-8")))
    m = review.merge(
        gold[["repo", "query_number", "original_number", "source", "channel", "stratum"]],
        on=["repo", "query_number", "original_number"],
        suffixes=("_rev", ""),
    )
    return m


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    fresh = json.loads(FRESH_AUDIT.read_text(encoding="utf-8"))["by_channel"]
    resliced = reslice_adr0032(gold)

    pop = gold.groupby(["repo", "channel", "source", "stratum"]).size().to_dict()

    def pop_size(repo: str, channel: str, source: str, stratum: str) -> int:
        return int(pop.get((repo, channel, source, stratum), 0))

    def resliced_precision(repo: str, channel: str, source: str) -> dict | None:
        sub = resliced[
            (resliced["repo"] == repo)
            & (resliced["channel"] == channel)
            & (resliced["source"] == source)
        ]
        if sub.empty:
            return None
        n = len(sub)
        x = int((sub["verdict"] == "genuine").sum())
        return {
            "n_judged": n,
            "n_genuine": x,
            "precision": round(x / n, 4),
            "wilson95": list(wilson_ci(x, n)),
            "provenance": "ADR-0032 review, resliced by source",
        }

    # (repo, channel, source, stratum, decision, precision_evidence, note)
    rows = []

    def add(repo, channel, source, stratum, decision, evidence, note=""):
        n_pool = pop_size(repo, channel, source, stratum)
        rows.append(
            {
                "repo": repo,
                "channel": channel,
                "source": source,
                "stratum": stratum,
                "pool_size": n_pool,
                "decision": decision,
                "precision_evidence": evidence,
                "note": note,
            }
        )

    # ---- k8s ----
    add(
        "kubernetes_kubernetes",
        "k8s_forward_scrape",
        "body_related",
        "product",
        "KEEP",
        fresh["kubernetes_kubernetes::k8s_forward_scrape::body_related::product"]
        | {"provenance": "D1 fresh full census, this phase"},
    )
    add(
        "kubernetes_kubernetes",
        "k8s_forward_scrape",
        "body_related_ext",
        "product",
        "KEEP",
        fresh["kubernetes_kubernetes::k8s_forward_scrape::body_related_ext::product"]
        | {"provenance": "D1 fresh sample n=30 of 454, this phase"},
    )
    add(
        "kubernetes_kubernetes",
        "k8s_extended_mine",
        "body_related_ext",
        "product",
        "KEEP",
        resliced_precision("kubernetes_kubernetes", "k8s_extended_mine", "body_related_ext"),
        note="lowest-precision kept k8s channel (65% vs ~83-84% siblings) -- flagged for GG",
    )
    add(
        "kubernetes_kubernetes",
        "legacy_gold_v1",
        "body_related",
        "product",
        "KEEP",
        resliced_precision("kubernetes_kubernetes", "legacy_gold_v1", "body_related"),
        note="thin sample (n=3 of 37) -- consistent with sibling channels, not independently powered",
    )
    add(
        "kubernetes_kubernetes",
        "legacy_gold_v1",
        "body_ref",
        "product",
        "DROP",
        fresh["kubernetes_kubernetes::legacy_gold_v1::body_ref::product"],
        note="full census n=2, both incidental (self-disclaimed 'maybe/probably not the same') -- negligible size either way",
    )
    add(
        "kubernetes_kubernetes",
        "*",
        "title_sim",
        "*",
        "DROP",
        None,
        note="hard rule (ADR-0032, spec.md) -- title_sim dropped unconditionally, not re-litigated per repo",
    )

    # ---- vscode ----
    add(
        "microsoft_vscode",
        "vscode_dup_scrape",
        "dup_comment",
        "gate",
        "KEEP (duplicate stratum)",
        fresh["microsoft_vscode::vscode_dup_scrape::dup_comment::gate"]
        | {"provenance": "D1 fresh sample n=40 of 2242, this phase"},
        note="vscode's largest channel by far and the deciding factor for its verdict",
    )
    add(
        "microsoft_vscode",
        "legacy_gold_v1",
        "body_ref",
        "product",
        "KEEP (related stratum)",
        fresh["microsoft_vscode::legacy_gold_v1::body_ref::product"]
        | {"provenance": "D1 fresh full census, this phase"},
    )
    add(
        "microsoft_vscode",
        "legacy_gold_v1",
        "body_related",
        "product",
        "KEEP (related stratum)",
        fresh["microsoft_vscode::legacy_gold_v1::body_related::product"]
        | {"provenance": "D1 fresh full census, this phase"},
    )
    add(
        "microsoft_vscode",
        "vscode_body_refs",
        "body_related",
        "product",
        "KEEP (related stratum)",
        fresh["microsoft_vscode::vscode_body_refs::body_related::product"]
        | {"provenance": "D1 fresh full census, this phase"},
    )
    add(
        "microsoft_vscode",
        "vscode_body_refs",
        "body_related_ext",
        "product",
        "DROP",
        {
            "precision_range": [0.30, 0.43],
            "n": 30,
            "provenance": "ADR-0030 Channel A table, cited not re-derived",
        },
        note="noisy, same tier as title_sim -- non-title_sim alone is NOT sufficient for 'clean'",
    )
    add(
        "microsoft_vscode",
        "*",
        "title_sim",
        "*",
        "DROP",
        None,
        note="hard rule (ADR-0032, spec.md) -- title_sim dropped unconditionally",
    )

    kept = [r for r in rows if r["decision"].startswith("KEEP")]
    k8s_kept = [r for r in kept if r["repo"] == "kubernetes_kubernetes"]
    vsc_kept = [r for r in kept if r["repo"] == "microsoft_vscode"]

    def weighted_precision(group: list[dict]) -> float | None:
        num = sum(r["pool_size"] * r["precision_evidence"]["precision"] for r in group)
        den = sum(r["pool_size"] for r in group)
        return round(num / den, 4) if den else None

    summary = {
        "kubernetes_kubernetes": {
            "clean_pool_size": sum(r["pool_size"] for r in k8s_kept),
            "weighted_precision": weighted_precision(k8s_kept),
        },
        "microsoft_vscode": {
            "duplicate_stratum_size": pop_size(
                "microsoft_vscode", "vscode_dup_scrape", "dup_comment", "gate"
            ),
            "duplicate_stratum_precision": fresh[
                "microsoft_vscode::vscode_dup_scrape::dup_comment::gate"
            ]["precision"],
            "related_stratum_size": sum(
                r["pool_size"] for r in vsc_kept if "related" in r["decision"]
            ),
            "related_stratum_precision": weighted_precision(
                [r for r in vsc_kept if "related" in r["decision"]]
            ),
            "combined_clean_pool_size": sum(r["pool_size"] for r in vsc_kept),
            "combined_weighted_precision": weighted_precision(vsc_kept),
        },
    }

    out = {"channel_decisions": rows, "summary": summary}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
