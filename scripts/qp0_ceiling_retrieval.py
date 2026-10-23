"""Retrieval ceiling analysis (spec 4.4, part 2): how answerable are duplicate / fix-link queries?

PRE-REGISTERED DESIGN (2026-10-20; do not change after seeing results). No model is trained or
scored; TF-IDF (sublinear tf, English stop words, min_df 2, fitted on all issues' title + first
1000 chars of body) is a text-similarity MEASUREMENT and the TF-IDF rank is a labelled BASELINE,
not a ceiling. The index for a query is all issues created strictly BEFORE it.
 (a) Duplicate task, for dup_task_pairs.parquet (queries in validation/test) and, separately,
     all duplicate_pairs.parquet. Each (source, canonical) pair is one query/target sample.
     Reported: share whose canonical is in the index (in the table AND earlier than the query),
     share of canonicals older than the scrape window (not in the table, not a PR, number below
     the smallest scraped issue number), share that are PRs / other missing / newer; TF-IDF
     cosine between query and target (min, quartiles, max; share >= 0.1/0.2/0.3/0.5); TF-IDF
     rank of the target among all earlier issues -> Recall@1/5/10/50 with Wilson 95%, under TWO
     denominators: indexed queries only, and all pairs (unindexed counted as misses). Rank =
     1 + number of earlier issues with strictly higher cosine than the target.
 (b) Fix-linked task. Groups = issues fixed by the same merged same-repo PR from
     fix_links.parquet, method closed_event_closer_pr only (the only method the fix-link audit
     validated). Reported: number of groups, size distribution, per split the number of
     ANSWERABLE queries (issues with >= 1 earlier group member), cosine quantiles over
     (query, earlier member) pairs, Recall@k of the best-ranked member, and a CONTROL of one
     random earlier same-year issue per query (random.Random(42)). Emitted only if
     reports/qp0_fixlink_audit_<slug>_2026-10.json exists with acceptance.result == 'pass';
     otherwise {withheld: reason}.
 (c) Every section states its query count n; small n limits any claim.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer  # similarity measurement only

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qp0_fixlink_audit import wilson  # noqa: E402

DESIGN = __doc__.split("PRE-REGISTERED DESIGN", 1)[1].strip()
KS = (1, 5, 10, 50)
COS_THRESHOLDS = (0.1, 0.2, 0.3, 0.5)
METHOD = "closed_event_closer_pr"
SPLITS = ("train", "validation", "test")


def rate(k: int, n: int) -> dict[str, Any]:
    ci = wilson(k, n)
    return {"k": int(k), "n": int(n), "rate": k / n if n else None, "wilson95": ci and list(ci)}


def cos_summary(vals: list[float]) -> dict[str, Any]:
    """Quantiles and threshold shares of a list of cosines."""
    a = np.array(vals, dtype=float)
    if not len(a):
        return {"n": 0}
    q = np.quantile(a, [0, 0.25, 0.5, 0.75, 1])
    return {
        "n": len(a), "quantiles_min_q1_med_q3_max": [float(x) for x in q],
        **{f"share_ge_{t}": float((a >= t).mean()) for t in COS_THRESHOLDS},
    }  # fmt: skip


def recall(ranks: list[int], n_total: int) -> dict[str, Any]:
    return {f"recall@{k}": rate(sum(r <= k for r in ranks), n_total) for k in KS}


class Index:
    """TF-IDF vectors of all issues plus 'earlier than' masks."""

    def __init__(self, issues: pd.DataFrame) -> None:
        self.df = issues.sort_values("number").reset_index(drop=True)
        text = self.df["title"].fillna("") + " " + self.df["body"].fillna("").str.slice(0, 1000)
        self.x = TfidfVectorizer(
            sublinear_tf=True, stop_words="english", min_df=2, dtype=np.float32
        ).fit_transform(text)
        self.pos = {int(n): i for i, n in enumerate(self.df["number"])}
        self.created = self.df["created_at"].to_numpy()

    def scores(self, q: int) -> tuple[np.ndarray, np.ndarray]:
        """(cosine to every issue, mask of issues created strictly before q)."""
        i = self.pos[q]
        return self.x.dot(self.x[i].T).toarray().ravel(), self.created < self.created[i]

    def rank(self, q: int, targets: list[int]) -> tuple[int, float]:
        """(best rank, its cosine) among earlier issues for the best-scoring target."""
        s, earlier = self.scores(q)
        t = max(targets, key=lambda n: (s[self.pos[n]], -n))
        return 1 + int((earlier & (s > s[self.pos[t]])).sum()), float(s[self.pos[t]])


def pair_section(idx: Index, pairs: pd.DataFrame, is_pr: set[int]) -> dict[str, Any]:
    """Duplicate-task metrics for (source, canonical) pairs."""
    min_num, n = int(idx.df["number"].min()), len(pairs)
    kinds = {
        "indexed": 0,
        "pr": 0,
        "older_than_scrape_window": 0,
        "newer_or_equal": 0,
        "other_missing": 0,
    }
    ranks, cosines = [], []
    for s, c in zip(pairs["source"], pairs["canonical"], strict=True):
        s, c = int(s), int(c)
        if s not in idx.pos:
            kinds["other_missing"] += 1
        elif c in idx.pos:
            if idx.created[idx.pos[c]] < idx.created[idx.pos[s]]:
                kinds["indexed"] += 1
                r, cs = idx.rank(s, [c])
                ranks.append(r)
                cosines.append(cs)
            else:
                kinds["newer_or_equal"] += 1
        elif c in is_pr:
            kinds["pr"] += 1
        elif c < min_num:
            kinds["older_than_scrape_window"] += 1
        else:
            kinds["other_missing"] += 1
    return {
        "n_pairs": n, "canonical_breakdown": {k: rate(v, n) for k, v in kinds.items()},
        "cosine_query_target": cos_summary(cosines),
        "tfidf_baseline_recall_indexed_queries_only": recall(ranks, len(ranks)),
        "tfidf_baseline_recall_all_pairs_unindexed_are_misses": recall(ranks, n),
    }  # fmt: skip


def fix_groups(links: pd.DataFrame, issue_nums: set[int]) -> dict[int, list[int]]:
    """PR -> sorted issues (>= 2, in the table) for audited merged same-repo closer links."""
    ok = links[(links["method"] == METHOD) & links["pr_merged"] & links["issue"].isin(issue_nums)]
    g = ok.groupby("pr")["issue"].agg(lambda s: sorted(set(map(int, s))))
    return {int(pr): v for pr, v in g.items() if len(v) >= 2}


def fix_section(idx: Index, groups: dict[int, list[int]], split: pd.Series, seed: int = 42) -> dict:
    members: dict[int, set[int]] = {}
    for v in groups.values():
        for a in v:
            members.setdefault(a, set()).update(b for b in v if b != a)
    sizes = pd.Series([len(v) for v in groups.values()])
    rng, year = random.Random(seed), idx.df["created_at"].dt.year.to_numpy()
    out: dict[str, Any] = {
        "n_groups": len(groups),
        "group_size_dist": {str(k): int(v) for k, v in sizes.clip(upper=5).value_counts().sort_index().items()},
        "note": "groups use only the audited closed_event_closer_pr method; n is small",
    }  # fmt: skip
    for sp in SPLITS:
        qs = sorted(q for q in members if split.get(q) == sp)
        ranks, cos, c_ranks, c_cos, ans = [], [], [], [], 0
        for q in qs:
            earlier = [
                m for m in sorted(members[q]) if idx.created[idx.pos[m]] < idx.created[idx.pos[q]]
            ]
            if not earlier:
                continue
            ans += 1
            r, _ = idx.rank(q, earlier)
            ranks.append(r)
            s, mask = idx.scores(q)
            cos += [float(s[idx.pos[m]]) for m in earlier]
            pool = np.flatnonzero(mask & (year == year[idx.pos[q]]))
            if len(pool):
                pick = int(idx.df["number"].iloc[int(rng.choice(sorted(pool.tolist())))])
                cr, cc = idx.rank(q, [pick])
                c_ranks.append(cr)
                c_cos.append(cc)
        out[sp] = {
            "n_split_issues": int((split == sp).sum()), "n_fix_linked_issues": len(qs),
            "n_answerable_queries": ans, "cosine_query_earlier_member": cos_summary(cos),
            "tfidf_baseline_recall": recall(ranks, ans),
            "control_random_same_year": {"cosine": cos_summary(c_cos), "recall": recall(c_ranks, len(c_ranks))},
        }  # fmt: skip
    return out


def audit_gate(audit_path: Path) -> str | None:
    """None if the fix-link audit passed, else the reason the fix section is withheld."""
    if not audit_path.exists():
        return f"fix-link audit report missing: {audit_path.name}"
    result = json.loads(audit_path.read_text(encoding="utf-8")).get("acceptance", {}).get("result")
    return None if result == "pass" else f"fix-link audit acceptance is {result!r}, not 'pass'"


def run(derived: Path, splits_dir: Path, audit_path: Path) -> dict[str, Any]:
    # column-pruned: the 100k-issue vscode table with all columns does not fit the RAM budget
    issues = pd.read_parquet(derived / "issues.parquet", columns=["number", "created_at", "title", "body"])
    idx = Index(issues)
    split = pd.concat(
        [pd.read_parquet(splits_dir / f"{s}.parquet", columns=["number", "split"]) for s in SPLITS]
    ).set_index("number")["split"]
    allp = pd.read_parquet(derived / "duplicate_pairs.parquet")
    is_pr = set(allp.loc[allp["canonical_is_pr"], "canonical"].astype(int))
    rep: dict[str, Any] = {"design": DESIGN, "generated_with": "scripts/qp0_ceiling_retrieval.py"}
    rep["n_issues_indexed"] = len(idx.df)
    rep["duplicate_task_pairs_val_test"] = pair_section(
        idx, pd.read_parquet(splits_dir / "dup_task_pairs.parquet"), is_pr
    )
    rep["all_duplicate_pairs"] = pair_section(idx, allp, is_pr)
    why = audit_gate(audit_path)
    if why:
        rep["fix_linked"] = {"withheld": why}
    else:
        links = pd.read_parquet(derived / "fix_links.parquet")
        rep["fix_linked"] = fix_section(idx, fix_groups(links, set(idx.pos)), split)
        rep["fix_linked"]["fix_groups_parquet_rows_reference"] = len(
            pd.read_parquet(derived / "fix_groups.parquet")
        )
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Retrieval-side ceiling analysis")
    ap.add_argument("--repo", required=True)
    ap.add_argument("--derived-dir", type=Path, default=None)
    ap.add_argument("--splits-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    slug = a.repo.replace("/", "__")
    derived = a.derived_dir or Path(r"D:\triageiq-data") / "derived" / slug
    splits_dir = a.splits_dir or Path(r"D:\triageiq-data") / "splits" / slug
    out = a.out or Path(f"reports/qp0_ceiling_retrieval_{slug}_2026-10.json")
    audit = (
        Path(__file__).resolve().parents[1] / "reports" / f"qp0_fixlink_audit_{slug}_2026-10.json"
    )
    rep = run(derived, splits_dir, audit)
    rep["repo"] = a.repo
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rep, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
