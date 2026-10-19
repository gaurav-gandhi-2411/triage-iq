from __future__ import annotations

# Derive analysis tables (quality-push Phase 0.2a) from the RAW scrape written by
# qp0_scrape_github.py. Read-only on <data-dir>/raw; every output goes to --out-dir, and a re-run
# overwrites it (idempotent). Each derivation is a pure function so it is unit-testable.
import argparse
import gzip
import hashlib
import json
import os
import re
import sys
from collections.abc import Iterator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from qp0_derive_stats import stats_from_parquet, write_stats

BODY_CAP = 20000
COMMENT_CAP = 5000  # saved comments only support evidence extraction; keep them small
HORIZON = timedelta(hours=72)
# lower rank wins when several methods find the same (source, canonical) pair
METHOD_RANK = {
    "marked_as_duplicate_event": 0,
    "vscode_bot_comment": 1,
    "k8s_close_comment": 2,
    "dup_label_with_ref": 3,
}
_REF = r"(?:https?://github\.com/([\w.-]+/[\w.-]+)/issues/(\d+)|#(\d+))"
# "duplicate of #N", "Dup with <url>", "/duplicate 12", "marked as a duplicate of [x](url)"
DUP_RE = re.compile(
    r"\b(?:duplicate[sd]?|dups?)\b(?:\s+(?:of|with|to|as|issue))*[\s:]*(?:\[[^\]]*\]\(\s*)?" + _REF,
    re.I,
)
CLOSE_RE = re.compile(r"(?:^|\s)/close\b")


def parse_ts(s: str | None) -> datetime | None:
    """Parse a GitHub ISO timestamp (UTC 'Z') into an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


def text_hash(title: str | None, body: str | None) -> str:
    """sha1 of normalised title + newline + normalised body[:20000] (near-duplicate key)."""
    return hashlib.sha1(f"{_norm(title)}\n{_norm((body or '')[:BODY_CAP])}".encode()).hexdigest()  # noqa: S324


def _login(node: dict[str, Any] | None) -> str | None:
    return ((node or {}).get("author") or (node or {}).get("actor") or {}).get("login")


def label_events(rec: dict[str, Any]) -> list[dict[str, Any]]:
    """Labeled/Unlabeled timeline events in timeline (chronological) order."""
    out = []
    for n in rec.get("timelineItems", {}).get("nodes", []):
        if n and n["__typename"] in ("LabeledEvent", "UnlabeledEvent") and n.get("label"):
            act = "labeled" if n["__typename"] == "LabeledEvent" else "unlabeled"
            out.append(
                {
                    "number": rec["number"],
                    "ts": parse_ts(n["createdAt"]),
                    "action": act,
                    "label": n["label"]["name"],
                    "actor": _login(n),
                }
            )
    return out


def replay_labels(events: list[dict[str, Any]], at: datetime) -> list[str]:
    """Label set at time ``at`` (inclusive), replayed in timestamp order (stable on ties)."""
    state: dict[str, None] = {}
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
        "author": _login(c),
    }


def issue_row(rec: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One issues.parquet row plus the issue's label events."""
    created, closed = parse_ts(rec["createdAt"]), parse_ts(rec.get("closedAt"))
    ev = label_events(rec)
    nodes = [n for n in rec.get("timelineItems", {}).get("nodes", []) if n]
    closers = [
        n["closer"]["number"]
        for n in nodes
        if n["__typename"] == "ClosedEvent" and (n.get("closer") or {}).get("number")
    ]
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
        "author": _login(rec),
        "author_assoc": rec.get("authorAssociation"),
        "title": rec.get("title"),
        "body": body,
        "n_comments": rec["comments"]["totalCount"],
        "milestone_title": (rec.get("milestone") or {}).get("title"),
        "labels_final": ls,
        "labels_t72h": replay_labels(ev, created + HORIZON),
        "first_label_ts": min(labeled) if labeled else None,
        "text_hash": text_hash(rec.get("title"), body),
        "created_year": created.year,
        "closing_comments": [_comment(c) for c in rec["lastComments"]["nodes"] if c],
        "closed_by_prs": list(dict.fromkeys(closers)),
        "resolution_hours": (closed - created).total_seconds() / 3600 if closed else None,
        # still open at scrape time => right-censored for the resolution-time task
        "censored": rec["state"] == "OPEN",
        "scraped_at": parse_ts(rec.get("_scraped_at")),
    }, ev


def _ref_number(m: re.Match[str], repo: str) -> int | None:
    """Same-repo issue number of a DUP_RE match; cross-repo URLs are not usable pairs."""
    if m.group(3):
        return int(m.group(3))
    return int(m.group(2)) if m.group(1).lower() == repo.lower() else None


def _snippet(text: str, m: re.Match[str]) -> str:
    return " ".join(text[max(0, m.start() - 40) : m.end() + 60].split())[:200]


def _comment_text(c: dict[str, Any]) -> tuple[str | None, str]:
    """(author, text). k8s-ci-robot echoes the human's "/close" comment as a ">" quote; that
    echo is the only evidence when the original is older than the last-5 saved comments."""
    login, body = _login(c), (c.get("body") or "").replace("\r", "")
    if login == "k8s-ci-robot":
        if "In response to" not in body:
            return login, ""
        body = "\n".join(ln.lstrip()[1:] for ln in body.splitlines() if ln.lstrip()[:1] == ">")
    return login, body


def extract_duplicates(rec: dict[str, Any], repo: str) -> list[dict[str, Any]]:
    """Duplicate pairs (source=this issue) from events, comments and a ``*duplicate`` label.

    Per canonical the best-ranked method wins. A label alone yields nothing: it only lends
    its method to a target found in a comment that names a duplicate.
    """
    src = rec["number"]
    has_label = has_dup_label(rec)
    found: dict[int, dict[str, Any]] = {}

    def add(canon: int | None, method: str, ts: str, ev: str) -> None:
        if canon is None or canon == src:
            return
        row = {"source": src, "canonical": canon, "method": method, "ts": parse_ts(ts)}
        row["evidence"] = ev[:200]
        old = found.get(canon)
        if old is None or METHOD_RANK[method] < METHOD_RANK[old["method"]]:
            found[canon] = row

    for n in (n for n in rec.get("timelineItems", {}).get("nodes", []) if n):
        typ, canon = n["__typename"], (n.get("canonical") or {}).get("number")
        if typ == "MarkedAsDuplicateEvent":
            add(canon, "marked_as_duplicate_event", n["createdAt"], f"marked by {_login(n)}")
        elif typ == "UnmarkedAsDuplicateEvent":
            found.pop(canon, None)  # un-marking later retracts the pair
    for c in (c for c in rec["lastComments"]["nodes"] if c):
        login, text = _comment_text(c)
        m = DUP_RE.search(text)
        if not m:
            continue
        if login and login != "k8s-ci-robot" and login.lower().endswith(("bot", "[bot]")):
            method = "vscode_bot_comment"
        elif CLOSE_RE.search(text):
            method = "k8s_close_comment"
        elif has_label:
            method = "dup_label_with_ref"
        else:
            continue  # a bare mention such as "similar to #5" without close/label is not a pair
        add(_ref_number(m, repo), method, c["createdAt"], _snippet(text, m))
    return list(found.values())


def has_dup_label(rec: dict[str, Any]) -> bool:
    return any("duplicate" in (x or {}).get("name", "").lower() for x in rec["labels"]["nodes"])


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

    for t in (t for t in rec.get("timelineItems", {}).get("nodes", []) if t):
        c, s = t.get("closer") or {}, t.get("source") or {}
        if t["__typename"] == "ClosedEvent" and "number" in c:
            add(c["number"], "closed_event_closer_pr", c.get("merged"), c.get("mergedAt"))
        elif (
            t["__typename"] == "CrossReferencedEvent"
            and t.get("willCloseTarget")
            and not t.get("isCrossRepository")
            and "merged" in s  # only a PullRequest source has this field
            and s.get("repository", {}).get("nameWithOwner", "").lower() == repo.lower()
        ):
            add(s["number"], "xref_willclose_merged", s["merged"], s.get("mergedAt"))
    for pr in refs.get(n, []):
        if pr in prs:
            add(pr, "pr_closing_refs", prs[pr]["merged"], None)
    return list(out.values())


def fix_groups(links: pd.DataFrame) -> pd.DataFrame:
    """PRs that merged-fix >= 2 distinct issues."""
    g = links[links["pr_merged"]].groupby("pr")["issue"].agg(lambda s: sorted(set(s)))
    g = g[g.map(len) >= 2]
    return pd.DataFrame({"pr": g.index.astype("int64"), "issues": list(g)})


def iter_records(kind_dir: Path, limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Records from <kind_dir>/<YYYY>/*.jsonl.gz (checkpoint.json / *.tmp never match)."""
    for f in sorted(kind_dir.glob("*/*.jsonl.gz"))[:limit]:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def build_tables(repo: str, base: Path, limit: int | None = None) -> dict[str, pd.DataFrame]:
    """Read raw files under ``base`` (= <data>/raw/<owner>__<repo>) and derive every table."""
    prs_rows = {
        r["number"]: r for r in (pr_row(x, repo) for x in iter_records(base / "prs", limit))
    }
    prs = {k: {"merged": v["merged"], "merged_at": v["merged_at"]} for k, v in prs_rows.items()}
    refs: dict[int, list[int]] = {}
    for k, v in prs_rows.items():
        for i in v["closing_issue_numbers"]:
            refs.setdefault(i, []).append(k)
    issues: dict[int, dict[str, Any]] = {}
    events: list[dict[str, Any]] = []
    dups: list[dict[str, Any]] = []
    fixes: list[dict[str, Any]] = []
    recs = {r["number"]: r for r in iter_records(base / "issues", limit)}  # last copy wins
    if not recs:
        raise SystemExit(f"no issue files under {base / 'issues'}")
    for n, rec in recs.items():
        row, ev = issue_row(rec)
        d = extract_duplicates(rec, repo)
        row["dup_label_unresolved"] = has_dup_label(rec) and not d
        issues[n] = row
        events += ev
        dups += d
        fixes += extract_fix_links(rec, repo, prs, refs)
    iss = pd.DataFrame(list(issues.values()))
    cutoff = max(
        [iss["scraped_at"].max()] + [v["scraped_at"] for v in prs_rows.values() if v["scraped_at"]]
    )
    iss["scraped_at_max"] = cutoff
    prs_df = pd.DataFrame(list(prs_rows.values()), columns=PR_COLS + ["scraped_at"])
    prs_df = prs_df.drop(columns="scraped_at")
    cols = {
        "label_events": ["number", "ts", "action", "label", "actor"],
        "duplicate_pairs": ["source", "canonical", "method", "ts", "evidence"],
        "fix_links": ["issue", "pr", "method", "pr_merged", "pr_merged_at"],
    }
    pr_nums = set(prs_rows)
    dup_df = pd.DataFrame(dups, columns=cols["duplicate_pairs"])
    dup_df["canonical_is_pr"] = dup_df["canonical"].isin(pr_nums)
    fix_df = pd.DataFrame(fixes, columns=cols["fix_links"])
    fix_df["pr_merged_at"] = pd.to_datetime(fix_df["pr_merged_at"], utc=True)
    return {
        "issues": iss.drop(columns="scraped_at"),
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
