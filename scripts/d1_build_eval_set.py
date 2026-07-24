"""D1 task 2 (final): freeze the held-out eval set, disjoint from training BY ISSUE NUMBER.

Eval set = every individually hand-verified GENUINE pair, drawn from checkpoint 1's samples
plus the eval-carve top-up round, capped at a per-task target size. Because it's built entirely
from hand-verified pairs, its precision is ~100% by construction (the >=90% hard rule is cleared
trivially) -- reported precision below is the underlying hit-rate across ALL pairs reviewed for
this purpose (genuine + incidental), for honesty about channel yield, not the eval set's own
precision (which is 100% of what's included, by definition of "included").

Disjointness (ADR-0018-style, ISSUE level not just pair level): once an issue number appears on
either side of an eval pair, EVERY training pair touching that issue number (as query or
original, in ANY channel, including channels not used for eval) is dropped from the training
pool. This is the leakage guard that makes D2's future trained-model numbers valid -- a pair-level
dedup alone would still let a fine-tune see issue X in training (paired with Y) and then be
evaluated on X (paired with Z), pulling X's embedding toward Y during training in a way that
could inflate its eval-time similarity to Z's neighborhood.

Reads:  data/gold_related_v2.parquet
        reports/d1_pair_quality_review.json            (checkpoint 1 judged pairs)
        reports/d1_eval_carve_review_k8s.json          (eval-carve top-up, k8s)
        reports/d1_eval_carve_review_vscode.json       (eval-carve top-up, vscode)
        reports/phaseC_pair_quality_review.json        (ADR-0032, for the vscode related stratum
                                                          and the k8s legacy_gold_v1/body_related bank)
Writes: reports/d1_eval_set_k8s_related.json
        reports/d1_eval_set_vscode_duplicate.json
        reports/d1_eval_set_vscode_related.json
        reports/d1_train_pool_k8s_related.json
        reports/d1_train_pool_vscode_duplicate.json
        reports/d1_eval_set_summary.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

GOLD_PATH = Path("data/gold_related_v2.parquet")
CKPT1_REVIEW = Path("reports/d1_pair_quality_review.json")
CARVE_K8S = Path("reports/d1_eval_carve_review_k8s.json")
CARVE_VSCODE = Path("reports/d1_eval_carve_review_vscode.json")
ADR0032_REVIEW = Path("reports/phaseC_pair_quality_review.json")

TARGET_K8S_EVAL = 150
TARGET_VSCODE_DUP_EVAL = 200

REPORTS = Path("reports")


def load_reviews() -> pd.DataFrame:
    frames = []
    for path in (CKPT1_REVIEW, CARVE_K8S, CARVE_VSCODE, ADR0032_REVIEW):
        frames.append(pd.DataFrame(json.loads(path.read_text(encoding="utf-8"))))
    all_reviewed = pd.concat(frames, ignore_index=True, sort=False)
    # de-dupe: a pair may appear in more than one review file only if re-reviewed by mistake
    before = len(all_reviewed)
    all_reviewed = all_reviewed.drop_duplicates(subset=["repo", "query_number", "original_number"])
    if before != len(all_reviewed):
        print(f"WARNING: {before - len(all_reviewed)} duplicate-reviewed pairs collapsed")
    return all_reviewed


def freeze_eval(genuine: pd.DataFrame, target: int, label: str) -> tuple[pd.DataFrame, dict]:
    """Take up to `target` genuine pairs (all of them if fewer), in the fixed hand-judging
    sample order (already shuffled by the SEED=42/43 draws) -- no further randomization here."""
    eval_set = genuine.head(target).copy()
    provenance = {
        "label": label,
        "target": target,
        "available_genuine": len(genuine),
        "eval_set_size": len(eval_set),
        "achieved_target": len(eval_set) >= target,
    }
    return eval_set, provenance


def build_training_pool(
    gold: pd.DataFrame,
    repo: str,
    channel_source_stratum: list[tuple[str, str, str]],
    eval_issue_numbers: set[int],
) -> pd.DataFrame:
    """All pairs in the given clean (channel, source, stratum) triples for `repo`, MINUS any pair
    touching an eval issue number (query or original) -- issue-level disjointness, not just
    pair-level dedup. `stratum` MUST be included in the match key: k8s's `k8s_forward_scrape`
    channel has both `gate` (PR-query proxy, excluded from the clean pool) and `product` rows
    sharing the same (channel, source) -- matching on (channel, source) alone would silently
    pull the proxy-task gate pairs back into the "clean" training pool.
    """
    mask = (gold["repo"] == repo) & (
        gold[["channel", "source", "stratum"]].apply(tuple, axis=1).isin(channel_source_stratum)
    )
    pool = gold[mask].copy()
    touches_eval = pool["query_number"].isin(eval_issue_numbers) | pool["original_number"].isin(
        eval_issue_numbers
    )
    dropped = int(touches_eval.sum())
    pool = pool[~touches_eval].copy()
    return pool, dropped


def main() -> None:
    gold = pd.read_parquet(GOLD_PATH)
    reviewed = load_reviews()
    reviewed_genuine = reviewed[reviewed["verdict"] == "genuine"]

    summary: dict = {}

    # ---------------- k8s (related task) ----------------
    k8s_genuine = reviewed_genuine[
        (reviewed_genuine["repo"] == "kubernetes_kubernetes")
        & (
            reviewed_genuine[["channel", "source"]]
            .apply(tuple, axis=1)
            .isin(
                [
                    ("k8s_forward_scrape", "body_related"),
                    ("k8s_forward_scrape", "body_related_ext"),
                    ("legacy_gold_v1", "body_related"),
                ]
            )
        )
    ]
    k8s_eval, k8s_prov = freeze_eval(k8s_genuine, TARGET_K8S_EVAL, "k8s_related")
    k8s_eval_issues = set(k8s_eval["query_number"]) | set(k8s_eval["original_number"])
    k8s_train, k8s_dropped = build_training_pool(
        gold,
        "kubernetes_kubernetes",
        [
            ("k8s_forward_scrape", "body_related", "product"),
            ("k8s_forward_scrape", "body_related_ext", "product"),
            ("legacy_gold_v1", "body_related", "product"),
        ],
        k8s_eval_issues,
    )
    # also drop confirmed-incidental pairs from training (hand-verified noise, not just unreviewed)
    incidental_keys = set(
        zip(
            reviewed[reviewed["verdict"] == "incidental"]["repo"],
            reviewed[reviewed["verdict"] == "incidental"]["query_number"],
            reviewed[reviewed["verdict"] == "incidental"]["original_number"],
            strict=True,
        )
    )
    k8s_train_keys = list(
        zip(
            k8s_train["repo"],
            k8s_train["query_number"],
            k8s_train["original_number"],
            strict=True,
        )
    )
    k8s_train = k8s_train[[k not in incidental_keys for k in k8s_train_keys]]
    summary["k8s_related"] = {
        **k8s_prov,
        "train_pool_size": len(k8s_train),
        "eval_issues_dropped_from_train": k8s_dropped,
    }

    # ---------------- vscode (duplicate task) ----------------
    vsc_dup_genuine = reviewed_genuine[
        (reviewed_genuine["repo"] == "microsoft_vscode")
        & (reviewed_genuine["source"] == "dup_comment")
    ]
    vsc_dup_eval, vsc_dup_prov = freeze_eval(
        vsc_dup_genuine, TARGET_VSCODE_DUP_EVAL, "vscode_duplicate"
    )
    vsc_dup_eval_issues = set(vsc_dup_eval["query_number"]) | set(vsc_dup_eval["original_number"])
    vsc_dup_train, vsc_dup_dropped = build_training_pool(
        gold,
        "microsoft_vscode",
        [("vscode_dup_scrape", "dup_comment", "gate")],
        vsc_dup_eval_issues,
    )
    vsc_dup_train_keys = list(
        zip(
            vsc_dup_train["repo"],
            vsc_dup_train["query_number"],
            vsc_dup_train["original_number"],
            strict=True,
        )
    )
    vsc_dup_train = vsc_dup_train[[k not in incidental_keys for k in vsc_dup_train_keys]]
    summary["vscode_duplicate"] = {
        **vsc_dup_prov,
        "train_pool_size": len(vsc_dup_train),
        "eval_issues_dropped_from_train": vsc_dup_dropped,
    }

    # ---------------- vscode (related task, eval-only, directional) ----------------
    vsc_rel_genuine = reviewed_genuine[
        (reviewed_genuine["repo"] == "microsoft_vscode")
        & (
            reviewed_genuine[["channel", "source"]]
            .apply(tuple, axis=1)
            .isin(
                [
                    ("legacy_gold_v1", "body_ref"),
                    ("legacy_gold_v1", "body_related"),
                    ("vscode_body_refs", "body_related"),
                ]
            )
        )
    ]
    summary["vscode_related"] = {
        "label": "vscode_related",
        "eval_set_size": len(vsc_rel_genuine),
        "status": "eval-only diagnostic, NOT trained on, NOT gated, NOT blended with duplicate task",
    }

    # ---------------- assert disjointness, write everything ----------------
    def issue_set(df: pd.DataFrame) -> set:
        return set(df["query_number"]) | set(df["original_number"])

    assert not (issue_set(k8s_eval) & issue_set(k8s_train)), "k8s eval/train issue overlap -- BUG"
    assert not (issue_set(vsc_dup_eval) & issue_set(vsc_dup_train)), (
        "vscode duplicate eval/train issue overlap -- BUG"
    )

    def dump(df: pd.DataFrame, name: str) -> None:
        cols = [
            "repo",
            "query_number",
            "original_number",
            "query_title",
            "original_title",
            "channel",
            "source",
            "stratum",
        ]
        (REPORTS / name).write_text(df[cols].to_json(orient="records", indent=2), encoding="utf-8")

    dump(k8s_eval, "d1_eval_set_k8s_related.json")
    dump(vsc_dup_eval, "d1_eval_set_vscode_duplicate.json")
    dump(vsc_rel_genuine, "d1_eval_set_vscode_related.json")
    dump(k8s_train, "d1_train_pool_k8s_related.json")
    dump(vsc_dup_train, "d1_train_pool_vscode_duplicate.json")

    (REPORTS / "d1_eval_set_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print("\nDisjointness assertions PASSED (issue-level, both eval sets).")


if __name__ == "__main__":
    main()
