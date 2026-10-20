"""Answerability counts for the k8s "rank EARLIER merged PRs" retrieval task (D83). DATA ONLY.

PRE-REGISTERED DEFINITIONS (2026-10-20; no model, no ranking, no relevance score is built here).
 * Task: for an issue q (created t_q) with an audited fix PR F(q) = fix_links method
   closed_event_closer_pr, same repo, merged (several: the earliest merged), rank EARLIER merged PRs.
 * INDEX(q) = merged PRs whose base branch is the repo default branch ("master"; cherry-picks into
   release-* are excluded and counted), created on or after 2023-01-01 (the scrape window; older PRs
   are unavailable: a limitation), and mergedAt STRICTLY before t_q. F(q) is never in the index
   (it merges after t_q; violations are counted and F is removed anyway).
 * POSITIVE for q = a PR p in INDEX(q) with files(p) intersect INFORMATIVE_FILES(F(q)) non-empty;
   INFORMATIVE_FILES(F) = files(F) minus EXCLUDED files. EXCLUDED = (a) patterns: go.mod / go.sum /
   go.work / go.work.sum (any directory), vendor/ and staging/**/vendor/, zz_generated.*,
   *.pb.go, generated.proto, api/openapi-spec/**, CHANGELOG/**, OWNERS, OWNERS_ALIASES; (b) any path
   changed by MORE than 1.0% of the merged master PRs with mergedAt <= 2025-06-30 (train period);
   that list is written to its own report with its sha256 and is then FROZEN. A PR with
   files_truncated is treated as having its first 300 files and is flagged.
 * ANSWERABLE iff INFORMATIVE_FILES(F(q)) is non-empty AND q has >= 1 positive in INDEX(q).
 * Splits = the frozen ISSUE splits (by the issue's created_at). No labels are used.
 * Context only (not selection criteria): thresholds 0.5% / 2%, cherry-picks included, the share of
   "trivially popular" positives (overlap only in the 100 most-changed informative files), and the
   Wilson half-width / cluster counts at the answerable test n (issues sharing a fix PR share positives).
"""

from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qp0_fixlink_audit import wilson  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DEFINITIONS", 1)[1].strip()
WINDOW_START = pd.Timestamp("2023-01-01T00:00:00Z")
TRAIN_END = pd.Timestamp("2025-06-30T23:59:59Z")
THRESHOLD = 0.01
SPLITS = ("train", "validation", "test")
PATTERNS = re.compile(
    r"(^|/)(go\.mod|go\.sum|go\.work|go\.work\.sum|OWNERS|OWNERS_ALIASES|generated\.proto)$"
    r"|(^|/)zz_generated[^/]*$|\.pb\.go$|^vendor/|/vendor/|^api/openapi-spec/|^CHANGELOG/"
)


def pattern_excluded(path: str) -> bool:
    return PATTERNS.search(path) is not None


def load_prs(pr_files_dir: Path) -> list[dict[str, Any]]:
    """Non-missing rows of <raw>/pr_files/*.jsonl.gz as plain dicts (read-only)."""
    out = []
    for f in sorted(pr_files_dir.glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                if r.get("missing"):
                    continue
                out.append(
                    {
                        "number": r["number"], "merged": bool(r["merged"]), "base": r["baseRefName"],
                        "merged_at": pd.Timestamp(r["mergedAt"]) if r.get("mergedAt") else None,
                        "created_at": pd.Timestamp(r["createdAt"]),
                        "files": tuple(n["path"] for n in r["files"]["nodes"]),
                        "truncated": bool(r.get("files_truncated")),
                    }
                )  # fmt: skip
    return out


def common_files(
    prs: list[dict[str, Any]], default: str, thr: float
) -> tuple[dict[str, float], int]:
    """Paths changed by MORE than ``thr`` of merged default-branch PRs merged by TRAIN_END."""
    pop = [p for p in prs if p["merged"] and p["base"] == default and p["merged_at"] <= TRAIN_END]
    cnt: Counter[str] = Counter(f for p in pop for f in set(p["files"]))
    return {f: c / len(pop) for f, c in cnt.items() if c / len(pop) > thr}, len(pop)


def fix_queries(
    issues: pd.DataFrame, links: pd.DataFrame, split: pd.Series
) -> list[dict[str, Any]]:
    """One query per issue with an audited merged same-repo closer PR (the earliest merged)."""
    ok = links[(links["method"] == "closed_event_closer_pr") & links["pr_merged"]]
    first = ok.sort_values(["issue", "pr_merged_at", "pr"]).drop_duplicates("issue")
    t = issues.set_index("number")["created_at"]
    return [
        {"issue": int(i), "fix": int(f), "t": t[i], "split": split.get(i)}
        for i, f in zip(first["issue"], first["pr"], strict=True)
        if i in t.index and split.get(i) in SPLITS
    ]  # fmt: skip


class Index:
    """Per-file time-ordered lists of index PRs (merged on the default branch, in the window)."""

    def __init__(self, prs: list[dict[str, Any]], default: str, with_cherry: bool = False) -> None:
        sel = [
            p
            for p in prs
            if p["merged"]
            and p["created_at"] >= WINDOW_START
            and (with_cherry or p["base"] == default)
        ]
        sel.sort(key=lambda p: (p["merged_at"], p["number"]))
        self.times = [p["merged_at"] for p in sel]
        self.trunc = {p["number"]: p["truncated"] for p in sel}
        self.files: dict[str, tuple[list[pd.Timestamp], list[int]]] = defaultdict(lambda: ([], []))
        for p in sel:
            for f in set(p["files"]):
                self.files[f][0].append(p["merged_at"])
                self.files[f][1].append(p["number"])
        self.pr_files = {p["number"]: set(p["files"]) for p in sel}

    def size_before(self, t: pd.Timestamp) -> int:
        return bisect.bisect_left(self.times, t)  # STRICTLY before t

    def positives(self, informative: set[str], t: pd.Timestamp, fix: int) -> set[int]:
        out: set[int] = set()
        for f in informative:
            times, nums = self.files.get(f, ([], []))
            out.update(nums[: bisect.bisect_left(times, t)])
        out.discard(fix)  # F merges after t by construction; never a candidate
        return out


def evaluate(
    queries: list[dict[str, Any]],
    prs: dict[int, dict[str, Any]],
    index: Index,
    excluded: set[str],
    top100: set[str],
) -> list[dict[str, Any]]:
    """Per-query answerability record (counts only, no ranking)."""
    res = []
    for q in queries:
        f = prs.get(q["fix"])
        rec = {
            **q,
            "fix_known": f is not None,
            "informative": 0,
            "positives": 0,
            "answerable": False,
        }
        if f is not None:
            info = {x for x in f["files"] if x not in excluded}
            pos = index.positives(info, q["t"], q["fix"])
            rec |= {
                "informative": len(info),
                "positives": len(pos),
                "answerable": bool(info) and bool(pos),
            }
            rec |= {"fix_after_query": bool(f["merged_at"] is not None and f["merged_at"] > q["t"]), "fix_truncated": f["truncated"]}  # fmt: skip
            rec["pos_truncated"] = any(index.trunc[n] for n in pos)
            rec["fix_in_window"] = bool(f["created_at"] >= WINDOW_START)
            rec["trivial_pairs"] = sum(index.pr_files[n] & info <= top100 for n in pos)
            rec["index_size"] = index.size_before(q["t"])
        res.append(rec)
    return res


def summarise(res: list[dict[str, Any]], split: str) -> dict[str, Any]:
    r = [x for x in res if x["split"] == split]
    ans = [x for x in r if x["answerable"]]
    pos = np.array([x["positives"] for x in ans]) if ans else np.array([0])
    clusters = Counter(x["fix"] for x in ans)
    m = np.array(list(clusters.values())) if clusters else np.array([1])
    n = len(ans)
    half = lambda p, k: 1.96 * (p * (1 - p) / k) ** 0.5 if k else None  # noqa: E731
    n_eff = float(m.sum() ** 2 / (m**2).sum())
    tp = sum(x["trivial_pairs"] for x in ans)
    return {
        "issues_with_fix_link": len(r), "fix_pr_files_known": sum(x["fix_known"] for x in r),
        "fix_has_informative_files": sum(x["informative"] > 0 for x in r), "answerable": n,
        "share_answerable_of_fix_linked": n / len(r) if r else None,
        "median_index_size_at_query": float(np.median([x["index_size"] for x in r if x["fix_known"]])) if r else None,
        "positives_per_answerable": {"min": int(pos.min()), "q1": float(np.percentile(pos, 25)), "median": float(np.median(pos)),
                                     "q3": float(np.percentile(pos, 75)), "max": int(pos.max()), "mean": float(pos.mean())},  # fmt: skip
        "fix_pr_in_window": sum(x.get("fix_in_window", False) for x in r), "fix_pr_older_than_window": sum(x["fix_known"] and not x["fix_in_window"] for x in r),  # fmt: skip
        "fix_not_after_query": sum(x["fix_known"] and not x["fix_after_query"] for x in r),
        "queries_fix_truncated": sum(x.get("fix_truncated", False) for x in r), "queries_positive_truncated": sum(x.get("pos_truncated", False) for x in r),  # fmt: skip
        "distinct_fix_prs_among_answerable": len(clusters), "effective_n_kish": n_eff,
        "wilson_half_width_at_n": {"p0.5": half(0.5, n), "p0.3": half(0.3, n)},
        "wilson_half_width_at_n_eff": {"p0.5": half(0.5, n_eff), "p0.3": half(0.3, n_eff)},
        "trivially_popular_positive_share": tp / int(pos.sum()) if n and pos.sum() else None,
        "answerable_with_a_nontrivial_positive": sum(x["trivial_pairs"] < x["positives"] for x in ans),
        "wilson_95_answerable_share": list(wilson(n, len(r))) if r else None,
    }  # fmt: skip


def run(
    prs: list[dict[str, Any]],
    issues: pd.DataFrame,
    links: pd.DataFrame,
    split: pd.Series,
    default: str = "master",
) -> dict:
    by_num = {p["number"]: p for p in prs}
    merged = [p for p in prs if p["merged"]]
    common, n_train = common_files(prs, default, THRESHOLD)
    queries = fix_queries(issues, links, split)

    def variant(thr: float, cherry: bool) -> dict[str, Any]:
        cm, _ = common_files(prs, default, thr)
        excl = {f for p in merged for f in p["files"] if pattern_excluded(f)} | set(cm)
        cnt: Counter[str] = Counter(f for p in merged if p["base"] == default and p["merged_at"] <= TRAIN_END for f in set(p["files"]) if f not in excl)  # fmt: skip
        top = {f for f, _ in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))[:100]}
        res = evaluate(queries, by_num, Index(prs, default, cherry), excl, top)
        return {s: summarise(res, s) for s in SPLITS}

    base = variant(THRESHOLD, False)
    years = Counter(p["merged_at"].year for p in merged if p["base"] == default)
    return {
        "design": DESIGN, "generated_with": "scripts/qp2_prior_pr_counts.py", "default_branch": default,
        "pr_population": {
            "rows": len(prs), "merged": len(merged), "merged_default_branch": sum(p["base"] == default for p in merged),
            "merged_other_base": sum(p["base"] != default for p in merged),
            "other_base_top": Counter(p["base"] for p in merged if p["base"] != default).most_common(8),
            "files_truncated": sum(p["truncated"] for p in merged), "merged_per_year_default_branch": dict(sorted(years.items())),
            "median_files_per_pr": float(np.median([len(p["files"]) for p in merged])),
            "merged_created_before_window": sum(p["created_at"] < WINDOW_START for p in merged),
        },  # fmt: skip
        "common_files": {"threshold": THRESHOLD, "n_train_prs": n_train, "n_files": len(common)},
        "per_split": base,
        "sensitivity": {"threshold_0.5pct": variant(0.005, False), "threshold_2pct": variant(0.02, False), "cherry_picks_included": variant(THRESHOLD, True)},  # fmt: skip
        "limitations": ["PRs created before 2023-01-01 are unavailable as candidates", "files_truncated PRs use their first 300 files"],
    }  # fmt: skip


def common_report(prs: list[dict[str, Any]], default: str) -> dict[str, Any]:
    common, n = common_files(prs, default, THRESHOLD)
    files = sorted(common)
    top = sorted(common.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "frozen": True, "threshold": THRESHOLD, "train_end": str(TRAIN_END), "n_train_prs": n, "n_files": len(files),
        "sha256_sorted_newline_joined": hashlib.sha256("\n".join(files).encode()).hexdigest(),
        "top25": [{"path": f, "share": s} for f, s in top[:25]], "files": files,
    }  # fmt: skip


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prior-PR answerability counts (data only)")
    ap.add_argument("--repo", default="kubernetes/kubernetes")
    ap.add_argument("--data-dir", type=Path, default=Path(r"D:\triageiq-data"))
    ap.add_argument("--default-branch", default="master")
    a = ap.parse_args(argv)
    slug = a.repo.replace("/", "__")
    prs = load_prs(a.data_dir / "raw" / slug / "pr_files")
    derived, splits = a.data_dir / "derived" / slug, a.data_dir / "splits" / slug
    split = pd.concat(
        [pd.read_parquet(splits / f"{s}.parquet", columns=["number", "split"]) for s in SPLITS]
    ).set_index("number")["split"]
    issues = pd.read_parquet(derived / "issues.parquet", columns=["number", "created_at"])
    links = pd.read_parquet(derived / "fix_links.parquet")
    rep = run(prs, issues, links, split, a.default_branch)
    rep["repo"] = a.repo
    reports = Path("reports")
    reports.mkdir(exist_ok=True)
    common = common_report(prs, a.default_branch)
    (reports / f"qp2_prior_pr_common_files_{slug}_2026-10.json").write_text(
        json.dumps(common, indent=1, sort_keys=True), encoding="utf-8"
    )
    rep["common_files"] |= {
        "sha256": common["sha256_sorted_newline_joined"],
        "top25": common["top25"],
    }
    (reports / f"qp2_prior_pr_counts_{slug}_2026-10.json").write_text(
        json.dumps(rep, indent=1, sort_keys=True, default=str), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                s: {
                    k: v
                    for k, v in rep["per_split"][s].items()
                    if k in ("issues_with_fix_link", "answerable")
                }
                for s in SPLITS
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
