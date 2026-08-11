"""Mining precision: assemble the expanded high-precision training pool for the D2 retry
fine-tune, per GG's go-ahead (2026-08-11).

Extends D1's original clean-pool decision (scripts/d1_assemble_clean_pool.py,
scripts/d1_build_eval_set.py) with two corrections, both evidenced by this session's fresh
strict-rubric channel audit (docs/investigations/2026-08-11-mining-precision-channel-
characterization.md, reports/mining_precision_strict_audit.json):

1. k8s_extended_mine/body_related_ext (200 product-stratum pairs) -- D1 DROPPED this channel
   citing a 65% precision estimate under D1's own looser genuine/incidental rubric (partial
   n=20 read, ADR-0032). Fresh, larger (n=50), STRICTER-rubric measurement finds 54.0%
   [40.4, 67.0] -- in the same 42-54% band as every k8s channel D1 KEPT (42.1-44.8% under the
   same strict rubric, from the 2026-08-11 k8s clean-eval build). D1's precision-based exclusion
   doesn't survive being re-measured on a rubric consistent with the one D1 itself later adopted
   for eval-set construction. Included here.
2. vscode_body_refs/body_related_ext (206 product-stratum pairs) -- D1 DROPPED this channel
   citing ADR-0030's 30-43% (n=30) estimate, calling it "noisy, same tier as title_sim." Fresh,
   larger (n=50), strict-rubric measurement finds 74.0% [60.5, 84.1] -- the ADR-0030 estimate's
   own upper bound (43%) sits below this sample's CI lower bound (60.5%): a real, non-overlapping
   disagreement, not sampling noise. Included here.

GG's explicit reasoning for folding vscode_dup_scrape/dup_comment into the SAME training pool
as the body-reference channels (both already true in D1's own vscode_duplicate task -- dup_comment
was never actually train-excluded, contra this session's initial mis-framing; see the ADR):
ADR-0027's stratify design correctly kept dup pairs out of EVAL (they'd inflate the measured
number) but that logic doesn't extend to training, where the eval-disjointness guard is what
actually prevents contamination -- training on high-precision pairs and evaluating on the
separately-frozen clean eval set is legitimate.

Held-out eval sets are REUSED AS-IS, never rebuilt or touched:
  - k8s: reports/d1_eval_set_k8s_related.json (150 pairs; == the 2026-08-11 clean-eval 150,
    confirmed byte-for-byte identical pair-id set; the 66-pair VALID subset is scored for the
    39.39% baseline comparison specifically)
  - vscode: reports/d1_eval_set_vscode_duplicate.json (200 pairs; current baseline 53.50%
    [46.5, 60.5], ADR-0040/2026-08-10-confirmed)

Exclusion sources applied (any pair EXPLICITLY hand-judged bad, from any review pass, is
dropped from training regardless of channel):
  - D1's loose genuine/incidental review (reports/d1_pair_quality_review.json, 139 pairs,
    "incidental" verdicts)
  - ADR-0032's genuine/incidental review (reports/phaseC_pair_quality_review.json,
    "incidental" verdicts)
  - This session's strict-rubric review (reports/mining_precision_strict_labeled.json,
    EXCLUDE_UMBRELLA / EXCLUDE_CAUSAL_ONLY / EXCLUDE_OTHER labels)
Unreviewed pairs in each included channel stay in the pool at the channel's measured/estimated
precision as background noise -- same convention D1 itself used (training pool = full channel
volume minus eval-touching issues minus explicitly-reviewed-bad pairs, not 100% hand-verified).

Disjointness (issue-level, ADR-0018/D1-style, HARD-FAIL): once an issue number appears on either
side of EITHER held-out eval set, every training pair touching that issue number -- in ANY
channel, any repo-matching side -- is dropped. Asserted programmatically before writing output.

Reads:  data/gold_related_v2.parquet
        reports/d1_eval_set_k8s_related.json
        reports/d1_eval_set_vscode_duplicate.json
        reports/d1_pair_quality_review.json
        reports/phaseC_pair_quality_review.json
        reports/mining_precision_strict_labeled.json
Writes: data/mining_precision_train_pool_k8s_related.parquet
        data/mining_precision_train_pool_vscode_duplicate.parquet
        reports/mining_precision_train_pool_report.json

Reproduce: python scripts/mining_precision_build_pool.py
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

GOLD_PATH = Path("data/gold_related_v2.parquet")
K8S_EVAL_PATH = Path("reports/d1_eval_set_k8s_related.json")
VSCODE_EVAL_PATH = Path("reports/d1_eval_set_vscode_duplicate.json")
D1_REVIEW = Path("reports/d1_pair_quality_review.json")
ADR0032_REVIEW = Path("reports/phaseC_pair_quality_review.json")
STRICT_REVIEW = Path("reports/mining_precision_strict_labeled.json")

OUT_K8S = Path("data/mining_precision_train_pool_k8s_related.parquet")
OUT_VSCODE = Path("data/mining_precision_train_pool_vscode_duplicate.parquet")
OUT_REPORT = Path("reports/mining_precision_train_pool_report.json")

K8S_CHANNELS = [
    ("k8s_forward_scrape", "body_related", "product"),
    ("k8s_forward_scrape", "body_related_ext", "product"),
    ("legacy_gold_v1", "body_related", "product"),
    ("k8s_extended_mine", "body_related_ext", "product"),  # reversed D1 DROP, see module docstring
]
VSCODE_CHANNELS = [
    ("vscode_dup_scrape", "dup_comment", "gate"),
    ("vscode_body_refs", "body_related", "product"),
    ("vscode_body_refs", "body_related_ext", "product"),  # reversed D1 DROP, see module docstring
    ("legacy_gold_v1", "body_related", "product"),
]


def load_bad_keys() -> set[tuple[str, int, int]]:
    """Union of every pair explicitly hand-judged bad across all review passes."""
    bad: set[tuple[str, int, int]] = set()

    for path, verdict_field, bad_values in (
        (D1_REVIEW, "verdict", {"incidental"}),
        (ADR0032_REVIEW, "verdict", {"incidental"}),
    ):
        if not path.exists():
            continue
        rows = json.loads(path.read_text(encoding="utf-8"))
        for r in rows:
            if r.get(verdict_field) in bad_values:
                bad.add((r["repo"], int(r["query_number"]), int(r["original_number"])))

    if STRICT_REVIEW.exists():
        rows = json.loads(STRICT_REVIEW.read_text(encoding="utf-8"))
        for r in rows:
            if r.get("label") != "VALID":
                bad.add((r["_repo"], int(r["_query_number"]), int(r["_target_number"])))

    return bad


def load_eval_issue_numbers(path: Path) -> set[int]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {int(r["query_number"]) for r in rows} | {int(r["original_number"]) for r in rows}


def build_pool(
    gold: pd.DataFrame,
    repo: str,
    channels: list[tuple[str, str, str]],
    eval_issues: set[int],
    bad_keys: set[tuple[str, int, int]],
) -> tuple[pd.DataFrame, dict]:
    mask = (gold["repo"] == repo) & (
        gold[["channel", "source", "stratum"]].apply(tuple, axis=1).isin(channels)
    )
    raw = gold[mask].copy()
    raw_by_channel = raw.groupby(["channel", "source"]).size().to_dict()

    touches_eval = raw["query_number"].isin(eval_issues) | raw["original_number"].isin(eval_issues)
    dropped_eval = int(touches_eval.sum())
    pool = raw[~touches_eval].copy()

    keys = list(zip(pool["repo"], pool["query_number"].astype(int), pool["original_number"].astype(int)))
    is_bad = [k in bad_keys for k in keys]
    dropped_bad = int(sum(is_bad))
    pool = pool[[not b for b in is_bad]].copy()

    stats = {
        "channels": [f"{c}/{s}/{st}" for c, s, st in channels],
        "raw_by_channel": {f"{c}/{s}": int(n) for (c, s), n in raw_by_channel.items()},
        "raw_total": int(len(raw)),
        "dropped_eval_issue_touch": dropped_eval,
        "dropped_hand_reviewed_bad": dropped_bad,
        "final_pool_size": int(len(pool)),
    }
    return pool, stats


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    bad_keys = load_bad_keys()
    log.info("Loaded %d hand-reviewed-bad pair keys across all review passes", len(bad_keys))

    k8s_eval_issues = load_eval_issue_numbers(K8S_EVAL_PATH)
    vsc_eval_issues = load_eval_issue_numbers(VSCODE_EVAL_PATH)
    log.info(
        "k8s eval touches %d distinct issue numbers; vscode eval touches %d",
        len(k8s_eval_issues),
        len(vsc_eval_issues),
    )

    k8s_pool, k8s_stats = build_pool(
        gold, "kubernetes_kubernetes", K8S_CHANNELS, k8s_eval_issues, bad_keys
    )
    vsc_pool, vsc_stats = build_pool(
        gold, "microsoft_vscode", VSCODE_CHANNELS, vsc_eval_issues, bad_keys
    )

    # HARD-FAIL disjointness assertion, issue-level, both directions
    def issue_set(df: pd.DataFrame) -> set[int]:
        return set(df["query_number"].astype(int)) | set(df["original_number"].astype(int))

    k8s_pool_issues = issue_set(k8s_pool)
    vsc_pool_issues = issue_set(vsc_pool)

    k8s_overlap = k8s_pool_issues & k8s_eval_issues
    vsc_overlap = vsc_pool_issues & vsc_eval_issues
    assert not k8s_overlap, (
        f"DISJOINTNESS VIOLATED: k8s train pool touches {len(k8s_overlap)} eval issue numbers: "
        f"{sorted(k8s_overlap)[:20]}"
    )
    assert not vsc_overlap, (
        f"DISJOINTNESS VIOLATED: vscode train pool touches {len(vsc_overlap)} eval issue numbers: "
        f"{sorted(vsc_overlap)[:20]}"
    )
    log.info("DISJOINTNESS ASSERTIONS PASSED: zero issue-number overlap, both repos.")

    OUT_K8S.parent.mkdir(parents=True, exist_ok=True)
    k8s_pool.to_parquet(OUT_K8S, index=False)
    vsc_pool.to_parquet(OUT_VSCODE, index=False)

    report = {
        "k8s_related": {**k8s_stats, "eval_issue_count": len(k8s_eval_issues), "disjointness": "PASS"},
        "vscode_duplicate": {
            **vsc_stats,
            "eval_issue_count": len(vsc_eval_issues),
            "disjointness": "PASS",
        },
        "comparison_to_D1_original": {
            "k8s_related_D1_original": 264,
            "k8s_related_now": k8s_stats["final_pool_size"],
            "vscode_duplicate_D1_original": 1734,
            "vscode_duplicate_now": vsc_stats["final_pool_size"],
        },
    }
    OUT_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Wrote %s, %s, %s", OUT_K8S, OUT_VSCODE, OUT_REPORT)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
