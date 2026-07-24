"""D1 checkpoint 1 (final, post-GG decision): clean product-task pool, per repo, per TASK stratum.

Combines three evidence sources, all cited with provenance (rule 65b):
  1. Fresh hand-judged samples this phase (reports/d1_pair_quality_review.json ->
     reports/d1_channel_precision_audit.json) -- k8s_forward_scrape (both sources),
     the tiny legacy body_ref/body_related full-census pools, vscode dup_comment.
  2. ADR-0032's existing 50-pair review, resliced by `source` (not previously reported
     at this granularity) -- k8s_extended_mine/body_related_ext (n=20), legacy body_related
     (n=3).
  3. ADR-0030's channel table (already measured, cited not re-derived) -- vscode's
     "Channel A" (body_related_ext / `vscode_body_refs`) at 30-43% precision (n=30) is
     NOISY, same tier as title_sim -- excluded despite being non-title_sim.

Hard rule (spec.md): title_sim is dropped everywhere, unconditionally, regardless of any
per-repo sample precision.

GG decisions (checkpoint 1, this phase):
  - k8s_extended_mine/body_related_ext (200 pairs, 65.0% precision) DROPPED -- meaningfully
    below its sibling channels (~83-84%); pool trades size for precision (536 vs 736).
  - vscode's two strata are reported SEPARATELY, never blended into one number: DUPLICATE
    (dup_comment, 2242 @ 85.0%, powered/gateable) vs RELATED (narrow reference channels, 22
    @ 86.4%, too small to power -- reported directional-only, held out entirely as an
    eval-only diagnostic, never trained on, never gated). Blending them would hide that the
    product-valuable RELATED task remains effectively unmeasured -- the exact
    proxy-vs-product trap this project has caught 4 times.
  - Same lens applied to k8s: does it have a DUPLICATE stratum distinct from RELATED?
    NO -- k8s has zero `dup_comment` rows anywhere in gold_related_v2.parquet (ADR-0030:
    "k8s has 0% comments_data coverage -- neither scrape fetched comments"). k8s's entire
    clean pool is the RELATED task only; there is no k8s duplicate stratum to split out.

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
    return review.merge(
        gold[["repo", "query_number", "original_number", "source", "channel", "stratum"]],
        on=["repo", "query_number", "original_number"],
        suffixes=("_rev", ""),
    )


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

    rows: list[dict] = []

    def add(repo, channel, source, stratum, task, decision, evidence, note=""):
        rows.append(
            {
                "repo": repo,
                "channel": channel,
                "source": source,
                "stratum": stratum,
                "task": task,
                "pool_size": pop_size(repo, channel, source, stratum),
                "decision": decision,
                "precision_evidence": evidence,
                "note": note,
            }
        )

    # ---- k8s: RELATED task only -- no duplicate/comment channel exists (0% comment coverage) ----
    add(
        "kubernetes_kubernetes",
        "k8s_forward_scrape",
        "body_related",
        "product",
        "related",
        "KEEP",
        fresh["kubernetes_kubernetes::k8s_forward_scrape::body_related::product"]
        | {"provenance": "D1 fresh full census, this phase"},
    )
    add(
        "kubernetes_kubernetes",
        "k8s_forward_scrape",
        "body_related_ext",
        "product",
        "related",
        "KEEP",
        fresh["kubernetes_kubernetes::k8s_forward_scrape::body_related_ext::product"]
        | {"provenance": "D1 fresh sample n=30 of 454, this phase"},
    )
    add(
        "kubernetes_kubernetes",
        "legacy_gold_v1",
        "body_related",
        "product",
        "related",
        "KEEP",
        resliced_precision("kubernetes_kubernetes", "legacy_gold_v1", "body_related"),
        note="thin sample (n=3 of 37) -- consistent with sibling channels, not independently powered",
    )
    add(
        "kubernetes_kubernetes",
        "k8s_extended_mine",
        "body_related_ext",
        "product",
        "related",
        "DROP",
        resliced_precision("kubernetes_kubernetes", "k8s_extended_mine", "body_related_ext"),
        note="GG decision: 65% meaningfully below sibling channels (~83-84%) -- traded size for precision",
    )
    add(
        "kubernetes_kubernetes",
        "legacy_gold_v1",
        "body_ref",
        "product",
        "related",
        "DROP",
        fresh["kubernetes_kubernetes::legacy_gold_v1::body_ref::product"],
        note="full census n=2, both incidental (self-disclaimed 'maybe/probably not the same')",
    )
    add(
        "kubernetes_kubernetes",
        "*",
        "title_sim",
        "*",
        "related",
        "DROP",
        None,
        note="hard rule -- title_sim dropped unconditionally",
    )

    # ---- vscode: TWO separate tasks, never blended ----
    add(
        "microsoft_vscode",
        "vscode_dup_scrape",
        "dup_comment",
        "gate",
        "duplicate",
        "KEEP",
        fresh["microsoft_vscode::vscode_dup_scrape::dup_comment::gate"]
        | {"provenance": "D1 fresh sample n=40 of 2242, this phase"},
        note="powered/gateable -- vscode's largest channel by far",
    )
    add(
        "microsoft_vscode",
        "legacy_gold_v1",
        "body_ref",
        "product",
        "related",
        "KEEP (eval-only, directional)",
        fresh["microsoft_vscode::legacy_gold_v1::body_ref::product"]
        | {"provenance": "D1 fresh full census, this phase"},
    )
    add(
        "microsoft_vscode",
        "legacy_gold_v1",
        "body_related",
        "product",
        "related",
        "KEEP (eval-only, directional)",
        fresh["microsoft_vscode::legacy_gold_v1::body_related::product"]
        | {"provenance": "D1 fresh full census, this phase"},
    )
    add(
        "microsoft_vscode",
        "vscode_body_refs",
        "body_related",
        "product",
        "related",
        "KEEP (eval-only, directional)",
        fresh["microsoft_vscode::vscode_body_refs::body_related::product"]
        | {"provenance": "D1 fresh full census, this phase"},
        note="GG decision: too small to power (n=22) -- held out entirely as eval-only diagnostic, never trained on, never gated",
    )
    add(
        "microsoft_vscode",
        "vscode_body_refs",
        "body_related_ext",
        "product",
        "related",
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
        "related",
        "DROP",
        None,
        note="hard rule -- title_sim dropped unconditionally",
    )

    def weighted_precision(group: list[dict]) -> float | None:
        num = sum(r["pool_size"] * r["precision_evidence"]["precision"] for r in group)
        den = sum(r["pool_size"] for r in group)
        return round(num / den, 4) if den else None

    k8s_kept = [
        r for r in rows if r["repo"] == "kubernetes_kubernetes" and r["decision"].startswith("KEEP")
    ]
    vsc_dup_kept = [
        r
        for r in rows
        if r["repo"] == "microsoft_vscode" and r["task"] == "duplicate" and r["decision"] == "KEEP"
    ]
    vsc_rel_kept = [
        r
        for r in rows
        if r["repo"] == "microsoft_vscode"
        and r["task"] == "related"
        and r["decision"].startswith("KEEP")
    ]

    summary = {
        "kubernetes_kubernetes": {
            "task": "related (no duplicate/comment channel exists -- 0% comment coverage, ADR-0030)",
            "clean_pool_size": sum(r["pool_size"] for r in k8s_kept),
            "weighted_precision": weighted_precision(k8s_kept),
            "powered_gateable": "TBD at eval-carve step (task 6/7) -- ADR-0030 estimated ~289-385 test "
            "pairs needed for a tight measurement CI; 536 total must cover train+eval",
        },
        "microsoft_vscode": {
            "duplicate_task": {
                "pool_size": sum(r["pool_size"] for r in vsc_dup_kept),
                "weighted_precision": weighted_precision(vsc_dup_kept),
                "status": "powered/gateable -- large enough for a proper train/eval split",
            },
            "related_task": {
                "pool_size": sum(r["pool_size"] for r in vsc_rel_kept),
                "weighted_precision": weighted_precision(vsc_rel_kept),
                "status": "directional-only, eval-only diagnostic (n=22) -- NOT trained on, NOT gated, "
                "NOT blended with the duplicate number",
            },
            "note": "duplicate and related are reported as two separate tasks, never combined into "
            "one 'vscode retrieval' number",
        },
    }

    out = {"channel_decisions": rows, "summary": summary}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
