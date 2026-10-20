from __future__ import annotations

# Summary tables / stats.json for the derived QP0 tables (quality-push Phase 0.2a). Facts only:
# counts and coverage fractions, never a verdict (taxonomy / priority decisions are made by the
# orchestrator from these numbers). Importable by qp0_derive_tables.py; also runnable standalone
# to recompute stats from existing parquet files (the --stats-only path).
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd

YEARS = (2023, 2024, 2025, 2026)
PRIORITY_RE = re.compile(r"priority|(?:^|[^a-z0-9])p[01](?:$|[^a-z0-9])", re.I)
# HEURISTIC, used only to compute the vscode multi-label raw rate: workflow/type labels that are
# not components and are therefore excluded before taking the 40 most frequent labels.
TYPE_LABELS_HEURISTIC = frozenset(
    [
        "bug", "feature-request", "enhancement", "question", "*question", "*duplicate",
        "*out-of-scope", "info-needed", "needs-more-info", "*caused-by-extension",
        "*dev-question", "important", "insiders-released", "verified", "verification-needed",
        "confirmed", "triage-needed", "unreleased", "upstream", "under-discussion",
    ]
)  # fmt: skip
STATS_TABLES = ("issues", "duplicate_pairs", "fix_links", "fix_groups", "prs")


def label_frequency(iss: pd.DataFrame) -> pd.DataFrame:
    """Every label (final or t72h) with counts overall, per creation year and at t72h."""
    fin = iss[["created_year", "labels_final"]].explode("labels_final").dropna()
    # explode repeats each issue's index per label; pandas 3 crosstab cannot align duplicates
    fin = fin.rename(columns={"labels_final": "label"}).reset_index(drop=True)
    t72 = iss["labels_t72h"].explode().dropna().value_counts()
    idx = fin["label"].value_counts().index.union(t72.index)
    out = pd.DataFrame(index=idx)
    out["n_issues_final"] = fin["label"].value_counts().reindex(idx, fill_value=0)
    by_year = pd.crosstab(fin["label"], fin["created_year"]).reindex(columns=YEARS, fill_value=0)
    for y in YEARS:
        out[f"n_issues_{y}"] = by_year[y].reindex(idx, fill_value=0)
    out["n_issues_t72h"] = t72.reindex(idx, fill_value=0)
    out["first_seen_year"] = fin.groupby("label")["created_year"].min().reindex(idx)
    out.index.name = "label"
    return out.sort_values(["n_issues_final", "label"], ascending=[False, True]).reset_index()


def priority_signals(iss: pd.DataFrame) -> pd.DataFrame:
    """Per priority-like label (and milestone-at-close): counts and share of issues."""
    n = max(len(iss), 1)
    labels = {x for ls in iss["labels_final"] for x in ls if PRIORITY_RE.search(x)} | {"important"}
    rows = []
    for lab in sorted(labels):
        nf = sum(lab in ls for ls in iss["labels_final"])
        n72 = sum(lab in ls for ls in iss["labels_t72h"])
        rows.append((lab, "label", nf, n72, nf / n, n72 / n))
    closed = iss[iss["closed_at"].notna()]
    nm = int(closed["milestone_title"].notna().sum())
    # milestone share is over CLOSED issues; the t72h columns do not apply to it
    rows.append(
        ("milestone_present_at_close", "milestone", nm, None, nm / max(len(closed), 1), None)
    )
    cols = ["signal", "kind", "n_final", "n_t72h", "share_final", "share_t72h"]
    return pd.DataFrame(rows, columns=cols)


def _multilabel(iss: pd.DataFrame) -> dict[str, Any]:
    """Raw counts of issues with >=1 / >=2 labels from a label family."""
    allf = [x for ls in iss["labels_final"] for x in ls]
    if any(x.startswith("area/") for x in allf):
        fam, keep = "area/*", lambda x: x.startswith("area/")  # noqa: E731
    else:
        top = pd.Series([x for x in allf if x not in TYPE_LABELS_HEURISTIC]).value_counts()
        fam, keep = "top40_non_type_heuristic", set(top.index[:40]).__contains__
    out: dict[str, Any] = {"family": fam, "n_issues": len(iss)}
    for col in ("labels_final", "labels_t72h"):
        k = iss[col].map(lambda ls: sum(1 for x in ls if keep(x)))
        out[col] = {"n_ge1": int((k >= 1).sum()), "n_ge2": int((k >= 2).sum())}
    return out


def compute_stats(t: dict[str, pd.DataFrame], repo: str, size_mb: float) -> dict[str, Any]:
    iss, prs, dup, fix = t["issues"], t["prs"], t["duplicate_pairs"], t["fix_links"]
    pr_years = prs["created_at"].dt.year if len(prs) else pd.Series(dtype="int64")
    per_year = {}
    for y in sorted(set(iss["created_year"]) | set(pr_years)):
        i = iss[iss["created_year"] == y]
        per_year[int(y)] = {
            "issues": len(i),
            "closed": int((i["state"] == "CLOSED").sum()),
            "open": int((i["state"] == "OPEN").sum()),
            "prs": int((pr_years == y).sum()),
        }
    mf = fix[fix["pr_merged"].astype(bool)]  # only merged PRs count as fixes
    flag = lambda s: int(s.fillna(False).astype(bool).sum())  # noqa: E731 - empty-table safe
    return {
        "repo": repo,
        "scrape_cutoff": str(iss["scraped_at_max"].max()),
        "per_year": per_year,
        "label_coverage": {
            "n_issues": len(iss),
            "share_ge1_final": float((iss["labels_final"].map(len) >= 1).mean()),
            "share_ge1_t72h": float((iss["labels_t72h"].map(len) >= 1).mean()),
        },
        "duplicates": {
            "by_method": dup["method"].value_counts().to_dict(),
            "n_pairs": len(dup),
            "canonical_is_pr": flag(dup["canonical_is_pr"]),
            "canonical_newer": flag(dup["canonical_newer"]),
            "canonical_not_in_table": len(dup) - flag(dup["canonical_in_table"]),
            "pairs_with_multiple_targets": int((dup["n_targets"] > 1).sum()),
            "unresolved_dup_labels": flag(iss["dup_label_unresolved"]),
            "closed_as_duplicate_state_reason": flag(iss["closed_as_duplicate"]),
            "closed_as_duplicate_unresolved": flag(iss["closed_as_duplicate_unresolved"]),
            "event_canonical_foreign_repo": int(iss["canon_foreign_repo"].sum()),
            "event_canonical_repo_unknown": int(iss["canon_repo_unknown"].sum()),
            "event_canonical_is_self_unusable": int(iss["canon_self"].sum()),
            "dup_target_direction_ambiguous": int(iss["dup_target_ambiguous"].sum()),
        },
        "closers": {  # ClosedEvent closer PRs excluded because their repo is not verifiable
            "closer_repo_unknown": int(iss["closer_repo_unknown"].sum()),
            "closer_foreign_repo": int(iss["closer_foreign_repo"].sum()),
        },
        "anomalies": {"closed_without_closed_at": flag(iss["anomaly_closed_no_closed_at"])},
        "fix_links": {
            "by_method_all": fix["method"].value_counts().to_dict(),
            "by_method_merged": mf["method"].value_counts().to_dict(),
            "issues_with_merged_fix": int(mf["issue"].nunique()),
            "prs_fixing_ge2_issues": len(t["fix_groups"]),
        },
        "multilabel": _multilabel(iss),
        "censored_open_issues": int(iss["censored"].sum()),
        "priority_signals": [  # NaN is not valid JSON: inapplicable cells become null
            {k: None if pd.isna(v) else int(v) if k.startswith("n_") else v for k, v in r.items()}
            for r in priority_signals(iss).to_dict("records")
        ],
        # everything outside this key is a pure function of the raw data; disk usage is not
        "volatile": {"storage_mb_raw_plus_derived": round(size_mb, 1)},
    }


def _dir_mb(*dirs: Path) -> float:
    return sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file()) / 1e6


def write_stats(t: dict[str, pd.DataFrame], repo: str, raw: Path, out: Path) -> dict[str, Any]:
    """Overwrite label_frequency.csv, priority_signals.csv and stats.json in ``out``."""
    label_frequency(t["issues"]).to_csv(out / "label_frequency.csv", index=False)
    priority_signals(t["issues"]).to_csv(out / "priority_signals.csv", index=False)
    stats = compute_stats(t, repo, _dir_mb(raw, out))
    (out / "stats.json").write_text(json.dumps(stats, indent=1, default=str), encoding="utf-8")
    return stats


def stats_from_parquet(repo: str, raw: Path, out: Path) -> dict[str, Any]:
    """Recompute the three summary files from the parquet tables already in ``out``."""
    t = {n: pd.read_parquet(out / f"{n}.parquet") for n in STATS_TABLES}
    return write_stats(t, repo, raw, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recompute QP0 stats from derived parquet files")
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    data = a.data_dir or Path(os.environ.get("TRIAGEIQ_DATA_DIR", r"D:\triageiq-data"))
    slug = a.repo.replace("/", "__")
    stats = stats_from_parquet(a.repo, data / "raw" / slug, a.out_dir or data / "derived" / slug)
    print(json.dumps(stats, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
