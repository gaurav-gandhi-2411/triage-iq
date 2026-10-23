from __future__ import annotations

# Derive analysis tables (quality-push Phase 0.2a) from the RAW scrape written by
# qp0_scrape_github.py. Read-only on <data-dir>/raw; every output goes to --out-dir, and a re-run
# overwrites it (idempotent). Each derivation is a pure function so it is unit-testable.
import argparse
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from qp0_derive_events import (
    METHOD_RANK,
    RawDataError,
    closed_as_duplicate,
    closers,
    extract_duplicates,
    has_dup_label,
    label_events,
    login,
    parse_ts,
    target_pairs,
)
from qp0_derive_stats import audited_fix_links, stats_from_parquet, write_stats

T = TypeVar("T")
BODY_CAP = 20000
COMMENT_CAP = 5000  # saved comments only support evidence extraction; keep them small
HORIZON = timedelta(hours=72)


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


def text_hash(title: str | None, body: str | None) -> str:
    """sha1 of normalised title + newline + normalised body[:20000] (near-duplicate key)."""
    return hashlib.sha1(f"{_norm(title)}\n{_norm((body or '')[:BODY_CAP])}".encode()).hexdigest()  # noqa: S324


def creation_labels(events: list[dict[str, Any]], final: list[str]) -> set[str]:
    """Labels present from creation: issue templates / forms apply them without a LabeledEvent.

    A label whose FIRST event is an unlabel existed before it; a final label with no event at all
    was never touched since creation.
    """
    first: dict[str, str] = {}
    for e in sorted(events, key=lambda e: e["ts"]):
        first.setdefault(e["label"], e["action"])
    return {k for k, a in first.items() if a == "unlabeled"} | {x for x in final if x not in first}


def replay_labels(
    events: list[dict[str, Any]], at: datetime, initial: set[str] | frozenset[str] = frozenset()
) -> list[str]:
    """Label set at time ``at`` (inclusive), replayed in timestamp order (stable on ties),
    starting from the ``initial`` (creation-time) labels."""
    state: dict[str, None] = dict.fromkeys(initial)
    for e in sorted(events, key=lambda e: e["ts"]):
        if e["ts"] > at:
            break
        if e["action"] == "labeled":
            state[e["label"]] = None
        else:
            state.pop(e["label"], None)
    return sorted(state)


def _comment(c: dict[str, Any]) -> dict[str, Any]:
    return {
        "body": (c.get("body") or "")[:COMMENT_CAP],
        "created_at": c.get("createdAt"),
        "author_assoc": c.get("authorAssociation"),
        "author": login(c),
    }


def issue_row(rec: dict[str, Any], repo: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One issues.parquet row plus the issue's label events."""
    created, closed = parse_ts(rec["createdAt"]), parse_ts(rec.get("closedAt"))
    ev = label_events(rec)
    ok_closers, cnt = closers(rec, repo)
    ls = [x["name"] for x in rec["labels"]["nodes"] if x]
    labeled = [e["ts"] for e in ev if e["action"] == "labeled"]
    body = (rec.get("body") or "")[:BODY_CAP]
    return {
        "number": rec["number"],
        "node_id": rec["id"],
        "created_at": created,
        "closed_at": closed,
        "state": rec["state"],
        "state_reason": rec.get("stateReason"),
        "author": login(rec),
        "author_assoc": rec.get("authorAssociation"),
        "title": rec.get("title"),
        "body": body,
        "n_comments": rec["comments"]["totalCount"],
        "milestone_title": (rec.get("milestone") or {}).get("title"),
        "labels_final": ls,
        "labels_t72h": replay_labels(ev, created + HORIZON, creation_labels(ev, ls)),
        "first_label_ts": min(labeled) if labeled else None,
        "text_hash": text_hash(rec.get("title"), body),
        "created_year": created.year,
        "closing_comments": [_comment(c) for c in rec["lastComments"]["nodes"] if c],
        "closed_by_prs": list(dict.fromkeys(c["number"] for c in ok_closers)),
        "closer_repo_unknown": cnt["closer_repo_unknown"],
        "closer_foreign_repo": cnt["closer_foreign_repo"],
        "closed_as_duplicate": closed_as_duplicate(rec),
        # loud, not silent: a CLOSED issue without closedAt gets resolution_hours=None
        "anomaly_closed_no_closed_at": rec["state"] == "CLOSED" and closed is None,
        "resolution_hours": (closed - created).total_seconds() / 3600 if closed else None,
        # still open at scrape time => right-censored for the resolution-time task
        "censored": rec["state"] == "OPEN",
        "scraped_at": parse_ts(rec.get("_scraped_at")),
    }, ev


PR_COLS = [
    "number", "title", "body", "merged", "merged_at", "created_at", "closed_at", "labels",
    "closing_issue_numbers", "changed_files", "additions", "deletions", "author_assoc",
]  # fmt: skip


def pr_row(rec: dict[str, Any], repo: str) -> dict[str, Any]:
    refs = [
        x["number"]
        for x in rec["closingIssuesReferences"]["nodes"]
        if x and x["repository"]["nameWithOwner"].lower() == repo.lower()
    ]
    return {
        "number": rec["number"],
        "title": rec.get("title"),
        "body": (rec.get("body") or "")[:BODY_CAP],
        "merged": bool(rec.get("merged")),
        "merged_at": parse_ts(rec.get("mergedAt")),
        "created_at": parse_ts(rec["createdAt"]),
        "closed_at": parse_ts(rec.get("closedAt")),
        "labels": [x["name"] for x in rec["labels"]["nodes"] if x],
        "closing_issue_numbers": refs,
        "changed_files": rec.get("changedFiles"),
        "additions": rec.get("additions"),
        "deletions": rec.get("deletions"),
        "author_assoc": rec.get("authorAssociation"),
        "scraped_at": parse_ts(rec.get("_scraped_at")),
    }


def extract_fix_links(
    rec: dict[str, Any], repo: str, prs: dict[int, dict[str, Any]], refs: dict[int, list[int]]
) -> list[dict[str, Any]]:
    """Issue->PR links. ``prs``: number -> {merged, merged_at}; ``refs``: issue -> PR numbers
    whose closingIssuesReferences name it. Unmerged links are kept, flagged pr_merged=False;
    only merged ones count as fixes (see fix_groups)."""
    n, out = rec["number"], {}

    def add(pr: int, method: str, merged: bool, merged_at: str | None) -> None:
        info = prs.get(pr)  # the PR table, when it has the PR, is more authoritative
        if info:
            merged, merged_at = info["merged"], info["merged_at"]
        else:
            merged_at = parse_ts(merged_at)
        out[(pr, method)] = {
            "issue": n,
            "pr": pr,
            "method": method,
            "pr_merged": bool(merged),
            "pr_merged_at": merged_at,
        }

    # no cross-reference method: GitHub clears willCloseTarget once the PR merges, so it never
    # yields merged links (1 of 375 xrefs true, 0 merged, schema-2 sample) - the ClosedEvent
    # closer already covers every merge-closed issue
    for c in closers(rec, repo)[0]:
        add(c["number"], "closed_event_closer_pr", c.get("merged"), c.get("mergedAt"))
    for pr in refs.get(n, []):
        if pr in prs:
            add(pr, "pr_closing_refs", prs[pr]["merged"], None)
    return list(out.values())


def fix_groups(links: pd.DataFrame) -> pd.DataFrame:
    """PRs that merged-fix >= 2 distinct issues (audited link methods only, D79)."""
    g = audited_fix_links(links).groupby("pr")["issue"].agg(lambda s: sorted(set(s)))
    g = g[g.map(len) >= 2]
    return pd.DataFrame({"pr": g.index.astype("int64"), "issues": list(g)})


def iter_records(kind_dir: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Records from <kind_dir>/<YYYY>/*.jsonl.gz (checkpoint.json / *.tmp never match). Each
    record gets ``_loc`` = "file:line" so later errors can name their source."""
    for f in sorted(kind_dir.glob("*/*.jsonl.gz"))[:limit]:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise RawDataError(f"{f}:{i} invalid JSON: {e}") from e
                yield {**rec, "_loc": f"{f}:{i}"}


def load_dup_targets(base: Path) -> dict[int, dict[str, Any]]:
    """Rows of <base>/dup_targets/<NNNNN>.jsonl.gz (the duplicate-target re-scrape), by number."""
    out: dict[int, dict[str, Any]] = {}
    for f in sorted((base / "dup_targets").glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                out[row["number"]] = row
    return out


def load_timelines(base: Path) -> dict[int, dict[str, Any]]:
    """Rows of <base>/timelines/<NNNNN>.jsonl.gz (complete per-issue re-scrape), by number."""
    out: dict[int, dict[str, Any]] = {}
    for f in sorted((base / "timelines").glob("*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                out[row["number"]] = row
    return out


def best_pairs(pairs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per (source, canonical): the best-ranked method wins (first seen on a tie)."""
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for p in pairs:
        k = (p["source"], p["canonical"])
        if k not in best or METHOD_RANK[p["method"]] < METHOD_RANK[best[k]["method"]]:
            best[k] = p
    return list(best.values())


def _derive(rec: dict[str, Any], fn: Callable[[dict[str, Any]], T], kind: str = "issue") -> T:
    """Run a per-record derivation; any malformed-record failure names file:line and number."""
    try:
        return fn(rec)
    except (RawDataError, KeyError, TypeError, AttributeError, ValueError) as e:
        raise RawDataError(f"{rec['_loc']} {kind} #{rec.get('number')}: {e}") from e


def build_tables(repo: str, base: Path, limit: int | None = None) -> dict[str, pd.DataFrame]:
    """Read raw files under ``base`` (= <data>/raw/<owner>__<repo>) and derive every table."""
    prs_rows = {
        r["number"]: _derive(r, lambda x: pr_row(x, repo), "PR")
        for r in iter_records(base / "prs", limit)
    }
    prs = {k: {"merged": v["merged"], "merged_at": v["merged_at"]} for k, v in prs_rows.items()}
    pr_set = set(prs_rows)  # prose pairs whose target is a known PR are dropped
    refs: dict[int, list[int]] = {}
    for k, v in prs_rows.items():
        for i in v["closing_issue_numbers"]:
            refs.setdefault(i, []).append(k)
    issues: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    dups: list[dict[str, Any]] = []
    fixes: list[dict[str, Any]] = []
    recs = {r["number"]: r for r in iter_records(base / "issues", limit)}  # last copy wins
    targets = load_dup_targets(base)
    timelines = load_timelines(base)
    other_side: list[dict[str, Any]] = []  # pairs whose source is not the issue carrying the event
    if not recs:
        raise SystemExit(f"no issue files under {base / 'issues'}")
    for n, rec in recs.items():
        diag: Counter[str] = Counter()
        # search-embedded timelines drop events: prefer the complete direct one when it exists
        tl = timelines.get(n)
        direct = tl is not None and not tl.get("missing")
        if direct:
            rec = {**rec, "timelineItems": tl["timelineItems"]}
        row, ev = _derive(rec, lambda x: issue_row(x, repo))
        d = _derive(rec, lambda x, dg=diag: extract_duplicates(x, repo, dg, pr_set))
        row["timeline_source"] = "direct" if direct else "embedded"
        row["timeline_direct_missing"] = tl is not None and bool(tl.get("missing"))
        row["canon_foreign_repo"] = diag["canon_foreign_repo"]
        row["canon_repo_unknown"] = diag["canon_repo_unknown"]
        row["canon_self"] = diag["canon_self"]
        if n in targets:
            tp = _derive(rec, lambda x, dg=diag, dt=targets[n]: target_pairs(dt, recs, repo, dg))
            d, other_side = (
                d + [p for p in tp if p["source"] == n],
                other_side + [p for p in tp if p["source"] != n],
            )
        row["dup_target_ambiguous"] = diag["dup_target_ambiguous"]
        row["dup_label_unresolved"] = has_dup_label(rec) and not d
        row["closed_as_duplicate_unresolved"] = row["closed_as_duplicate"] and not d
        issues[n] = row
        events += ev
        dups += d
        fixes += _derive(rec, lambda x: extract_fix_links(x, repo, prs, refs))
    dups = best_pairs(dups + other_side)
    iss = pd.DataFrame(list(issues.values()))
    has_pair = iss["number"].isin(
        {p["source"] for p in dups}
    )  # incl. pairs found from the other side
    iss["dup_label_unresolved"] &= ~has_pair
    iss["closed_as_duplicate_unresolved"] &= ~has_pair
    cutoff = max(
        [iss["scraped_at"].max()] + [v["scraped_at"] for v in prs_rows.values() if v["scraped_at"]]
    )
    iss["scraped_at_max"] = cutoff
    prs_df = pd.DataFrame(list(prs_rows.values()), columns=PR_COLS + ["scraped_at"])
    prs_df = prs_df.drop(columns="scraped_at")
    cols = {
        "label_events": ["number", "ts", "action", "label", "actor"],
        "duplicate_pairs": ["source", "canonical", "method", "ts", "evidence", "n_targets"],
        "fix_links": ["issue", "pr", "method", "pr_merged", "pr_merged_at"],
    }
    dup_df = pd.DataFrame(dups, columns=cols["duplicate_pairs"])
    dup_df["canonical_is_pr"] = dup_df["canonical"].isin(set(prs_rows))
    created = iss.set_index("number")["created_at"]
    dup_df["canonical_in_table"] = dup_df["canonical"].isin(created.index)
    c_new, c_src = dup_df["canonical"].map(created), dup_df["source"].map(created)
    # null (not False) when the canonical is outside the table: the comparison is unknowable
    dup_df["canonical_newer"] = (c_new > c_src).astype("boolean").where(c_new.notna())
    fix_df = pd.DataFrame(fixes, columns=cols["fix_links"])
    fix_df["pr_merged_at"] = pd.to_datetime(fix_df["pr_merged_at"], utc=True)
    return {
        "issues": iss,
        "label_events": pd.DataFrame(events, columns=cols["label_events"]),
        "duplicate_pairs": dup_df,
        "fix_links": fix_df,
        "fix_groups": fix_groups(fix_df) if len(fix_df) else pd.DataFrame(columns=["pr", "issues"]),
        "prs": prs_df,
    }


def write_outputs(t: dict[str, pd.DataFrame], repo: str, raw: Path, out: Path) -> dict[str, Any]:
    """Overwrite every parquet + summary file in ``out`` and return the stats."""
    out.mkdir(parents=True, exist_ok=True)
    for name, df in t.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
    return write_stats(t, repo, raw, out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Derive QP0 tables from raw GitHub scrape")
    ap.add_argument("--repo", required=True, help="owner/name")
    default_dir = os.environ.get("TRIAGEIQ_DATA_DIR", r"D:\triageiq-data")
    ap.add_argument("--data-dir", type=Path, default=Path(default_dir))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--limit-files", type=int, default=None, help="first N day-files per kind")
    ap.add_argument("--stats-only", action="store_true", help="recompute stats from out-dir")
    a = ap.parse_args(argv)
    slug = a.repo.replace("/", "__")
    out = a.out_dir or a.data_dir / "derived" / slug
    raw = a.data_dir / "raw" / slug
    if a.stats_only:
        stats = stats_from_parquet(a.repo, raw, out)
    else:
        stats = write_outputs(build_tables(a.repo, raw, a.limit_files), a.repo, raw, out)
    print(json.dumps(stats, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
