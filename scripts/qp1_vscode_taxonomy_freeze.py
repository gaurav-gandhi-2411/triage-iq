"""vscode component-taxonomy FREEZE artifact (label-only: no model, no text). ADR-0070 cites this rule.

PRE-REGISTERED RULE (2026-10-20; do not change after seeing results).
 * FAMILY F = V_all of qp0_taxonomy_coverage.py (V_old union V_new, labels with >= 100 issues, with
   its exclusion set, which was itself extended post hoc on 2026-10-20 after reading the real label
   list) MINUS NON_OWNERSHIP = {error-telemetry, chat-ext-issue, api-proposal} (bot crash reports, a
   triage marker, an issue type).
 * CLASSES = labels in F with >= 30 issues among issues created <= 2025-06-30 (train period); no
   stability condition (mirrors k8s). Every other F label (sparse in the train period) maps to
   `other`. REPLACED RULE: the first version (2026-10-20, same day, BEFORE any model was scored) also
   required >= 5 issues in each of 2023 and 2024 and gave 89 classes; it was replaced because it
   dropped 11 classes with >= 30 train issues that are real product areas (terminal-suggest, chat,
   chat-mcp, ...), a rule artefact. Both versions are reported side by side in `rule_comparison`;
   the choice rests on the TRAIN / VALIDATION shares there, the test shares are descriptive only.
 * Primary label = the earliest `labeled` event among the F labels still present at the end, ties
   alphabetical (qp0_ceiling_labels.per_issue). Population = issues with >= 1 F label at the end.
 * Splits by created_at with the boundaries of qp0_make_splits (train < 2025-07-01, validation
   2025-07-01 .. 2025-12-31, test >= 2026-01-01). Only created_at is used, so this report does not
   depend on the (not yet final) duplicate-pair leakage flags of the split manifest, which will be
   recomputed after the re-derive; classifier_eligible filtering is NOT applied here.
 * Reported (counts only): the class list with train counts per year and its sha256; per split the
   labelled issues, the share whose primary is a class vs `other`, and for the test split the share
   of `other` whose primary label has ZERO train issues (never seen) vs sparse; coverage of F at 72 h
   and at the end per split (share of all issues in the split); the majority class share on train and
   the always-majority accuracy on validation / test from TRAIN priors (a label-only baseline).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qp0_ceiling_labels as cl  # noqa: E402
import qp0_make_splits as ms  # noqa: E402
import qp0_taxonomy_coverage as tc  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED RULE", 1)[1].strip()
NON_OWNERSHIP = {"error-telemetry", "chat-ext-issue", "api-proposal"}
SPLITS = ("train", "validation", "test")
MIN_TRAIN, MIN_YEAR, STABLE_YEARS = 30, 5, (2023, 2024)
Pred = Callable[[str], bool]


def vscode_family(issues: pd.DataFrame) -> Pred:
    """F = V_all minus NON_OWNERSHIP, with V_new computed on the 2023-2026 issues."""
    v_all = tc.families("microsoft/vscode", issues)["V_all"]
    return lambda x: v_all(x) and x not in NON_OWNERSHIP


def pick_classes(
    issues: pd.DataFrame, family: Pred, stable: bool = False
) -> tuple[list[str], dict[str, dict[str, int]]]:
    """Classes (>= 30 train-period issues; ``stable`` adds the superseded 2023 / 2024 condition) and
    the per-year train-period counts of every F label (issues, not events)."""
    train = issues[issues["split"] == "train"]
    by_year: dict[str, Counter[int]] = {}
    for yr, ls in zip(train["created_at"].dt.year, train["labels_final"], strict=True):
        for lbl in {x for x in ls if family(x)}:
            by_year.setdefault(lbl, Counter())[yr] += 1
    table = {k: {str(y): c.get(y, 0) for y in (2023, 2024, 2025)} | {"train_total": sum(c.values())} for k, c in by_year.items()}  # fmt: skip
    classes = sorted(
        k for k, v in table.items()
        if v["train_total"] >= MIN_TRAIN and (not stable or all(v[str(y)] >= MIN_YEAR for y in STABLE_YEARS))
    )  # fmt: skip
    return classes, table


def assess(
    d: pd.DataFrame, classes: list[str], table: dict[str, dict[str, int]]
) -> tuple[dict, str, Counter]:
    """Per-split counts for one class list, the train-prior majority class and the train priors."""
    lab = d[d["final"].map(bool).astype(bool)].copy()
    lab["mapped"] = lab["primary"].where(lab["primary"].isin(classes), "other")
    prior = Counter(lab.loc[lab["split"] == "train", "mapped"])
    majority = sorted(prior, key=lambda c: (-prior[c], c))[0]
    seen_in_train = {k for k, v in table.items() if v["train_total"] > 0}
    out: dict[str, Any] = {}
    for s in SPLITS:
        sub, alls = lab[lab["split"] == s], d[d["split"] == s]
        other = sub[sub["mapped"] == "other"]
        out[s] = {
            "issues": len(alls), "labelled": len(sub), "primary_applied_at_creation": int((sub["primary_hours"] == 0).sum()),
            "coverage_f_final": float(alls["final"].map(bool).mean()), "coverage_f_72h": float(alls["t72"].map(bool).mean()),
            "primary_is_class": int((sub["mapped"] != "other").sum()), "primary_is_other": len(other),
            "share_primary_is_class": float((sub["mapped"] != "other").mean()),
            "other_never_seen_in_train": int((~other["primary"].isin(seen_in_train)).sum()),
            "other_sparse_seen_in_train": int(other["primary"].isin(seen_in_train).sum()),
        }  # fmt: skip
        if s != "train":
            out[s]["always_majority_accuracy"] = float((sub["mapped"] == majority).mean())
    return out, majority, prior


def freeze(issues: pd.DataFrame, events: pd.DataFrame, family: Pred) -> dict[str, Any]:
    """The frozen class list (revised rule) and the counts-only report for a family predicate."""
    iss = issues.sort_values("number").reset_index(drop=True)
    iss["split"] = ms.assign_split(iss["created_at"])
    classes, table = pick_classes(iss, family)
    old_classes, _ = pick_classes(iss, family, stable=True)
    d = cl.per_issue(iss, events, family).merge(iss[["number", "split"]], on="number")
    out, majority, prior = assess(d, classes, table)
    old_out, _, _ = assess(d, old_classes, table)

    def row(c: list[str], o: dict) -> dict[str, Any]:
        v = {
            "n_classes": len(c),
            "classes_sha256": hashlib.sha256("\n".join(c).encode()).hexdigest(),
        }
        for s in SPLITS:
            v[f"{s}_share_primary_in_class"] = o[s]["share_primary_is_class"]
            v[f"{s}_never_seen_share_of_labelled"] = (
                o[s]["other_never_seen_in_train"] / o[s]["labelled"]
            )
        return v

    return {
        "design": DESIGN, "generated_with": "scripts/qp1_vscode_taxonomy_freeze.py",
        "non_ownership_excluded": sorted(NON_OWNERSHIP), "n_classes": len(classes), "classes": classes,
        "classes_sha256": hashlib.sha256("\n".join(classes).encode()).hexdigest(),
        "class_train_counts_per_year": {k: table[k] for k in classes},
        "f_labels_not_classes": {k: v for k, v in sorted(table.items()) if k not in classes},
        "superseded_rule_v1_stability": {"classes": old_classes, "n_classes": len(old_classes),
                                         "dropped_vs_final": sorted(set(classes) - set(old_classes))},  # fmt: skip
        "rule_comparison": {"v1_stability_replaced": row(old_classes, old_out), "v2_final": row(classes, out),
                            "note": "selection rests on train / validation shares; test shares are descriptive"},  # fmt: skip
        "train_majority": {"class": majority, "share_of_labelled_train": prior[majority] / sum(prior.values()),
                           "other_share": prior["other"] / sum(prior.values())},  # fmt: skip
        "per_split": out,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vscode taxonomy freeze artifact (label-only)")
    ap.add_argument(
        "--derived-dir", type=Path, default=Path(r"D:\triageiq-data\derived\microsoft__vscode")
    )
    ap.add_argument(
        "--out", type=Path, default=Path("reports/qp1_vscode_taxonomy_frozen_2026-10.json")
    )
    a = ap.parse_args(argv)
    issues = pd.read_parquet(a.derived_dir / "issues.parquet", columns=["number", "created_at", "created_year", "labels_final", "labels_t72h", "text_hash"])  # fmt: skip
    issues = issues[issues["created_year"].isin(tc.YEARS)]
    events = pd.read_parquet(a.derived_dir / "label_events.parquet")
    rep = freeze(issues, events, vscode_family(issues))
    rep["repo"] = "microsoft/vscode"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"n_classes": rep["n_classes"], "classes_sha256": rep["classes_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
