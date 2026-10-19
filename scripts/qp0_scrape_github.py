from __future__ import annotations

# Resumable, checkpointed GitHub GraphQL scraper (quality-push Phase 0.1).
# Writes RAW issue / PR JSON, one <YYYY-MM-DD>.jsonl.gz per UTC creation day (the union of all
# sub-windows when a day had to be split), under <data-dir>/raw/<owner>__<repo>/<kind>/<YYYY>/,
# with an atomically-written checkpoint.json beside it. Nothing is ever written to the repo.
import argparse
import gzip
import json
import logging
import os
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger("qp0")
API = "https://api.github.com/graphql"
SEARCH_CAP = 1000  # GitHub search returns at most 1000 results per query
SCHEMA_VERSION = 2  # 2 = cross-repo targets carry repository{nameWithOwner}; 1 = rows before that
PAGE_SIZES = (100, 50, 25)
# (status, headers, parsed JSON body); transports raise TimeoutError/ConnectionError on I/O failure
Transport = Callable[[dict[str, Any]], tuple[int, dict[str, str], dict[str, Any]]]


class ScrapeError(RuntimeError):
    """Unrecoverable scrape failure (retries exhausted, completeness violated)."""


class TransientError(ScrapeError):
    """5xx / timeout persisted through the allotted tries."""


_TL_ACTOR = "createdAt actor{login}"
_NUM = "... on Issue{number repository{nameWithOwner}} ... on PullRequest{number repository{nameWithOwner}}"
TIMELINE = f"""timelineItems(first:100, after:$tl, itemTypes:[LABELED_EVENT, UNLABELED_EVENT,
MARKED_AS_DUPLICATE_EVENT, UNMARKED_AS_DUPLICATE_EVENT, CROSS_REFERENCED_EVENT, CLOSED_EVENT,
REOPENED_EVENT, CONNECTED_EVENT, DISCONNECTED_EVENT, MILESTONED_EVENT, DEMILESTONED_EVENT]){{
totalCount pageInfo{{hasNextPage endCursor}} nodes{{__typename
... on LabeledEvent{{{_TL_ACTOR} label{{name}}}}
... on UnlabeledEvent{{{_TL_ACTOR} label{{name}}}}
... on MarkedAsDuplicateEvent{{{_TL_ACTOR} canonical{{{_NUM}}}}}
... on UnmarkedAsDuplicateEvent{{{_TL_ACTOR} canonical{{{_NUM}}}}}
... on CrossReferencedEvent{{{_TL_ACTOR} willCloseTarget isCrossRepository source{{
 ... on Issue{{number repository{{nameWithOwner}}}}
 ... on PullRequest{{number merged mergedAt repository{{nameWithOwner}}}}}}}}
... on ClosedEvent{{{_TL_ACTOR} stateReason closer{{
 ... on PullRequest{{number merged mergedAt title repository{{nameWithOwner}}}} ... on Commit{{oid}}}}}}
... on ReopenedEvent{{{_TL_ACTOR}}}
... on ConnectedEvent{{{_TL_ACTOR} subject{{{_NUM}}}}}
... on DisconnectedEvent{{{_TL_ACTOR} subject{{{_NUM}}}}}
... on MilestonedEvent{{{_TL_ACTOR} milestoneTitle}}
... on DemilestonedEvent{{{_TL_ACTOR} milestoneTitle}}}}}}"""
_COMMON = "id number title body state createdAt closedAt authorAssociation author{login}"
_LABELS = "labels(first:50){nodes{name}} comments{totalCount}"
# lastComments is best-effort: the closing comment is usually, not always, among the last 5
ISSUE_FIELDS = f"""... on Issue{{{_COMMON} stateReason updatedAt milestone{{number title dueOn}}
{_LABELS} lastComments: comments(last:5){{nodes{{body createdAt authorAssociation author{{login}}}}}}
{TIMELINE}}}"""
PR_FIELDS = f"""... on PullRequest{{{_COMMON} merged mergedAt {_LABELS} changedFiles additions
deletions mergeCommit{{oid}}
closingIssuesReferences(first:20){{nodes{{number repository{{nameWithOwner}}}}}}}}"""
RATE = "rateLimit{cost remaining resetAt}"
SEARCH_Q = f"""query($q:String!, $n:Int!, $after:String, $tl:String){{ {RATE}
search(query:$q, type:ISSUE, first:$n, after:$after){{issueCount pageInfo{{hasNextPage endCursor}}
nodes{{{{FIELDS}}}}}}}}"""
TIMELINE_Q = f"""query($id:ID!, $tl:String){{ {RATE}
node(id:$id){{... on Issue{{{TIMELINE}}}}}}}"""


def _now() -> datetime:
    return datetime.now(UTC)


def _fmt(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def gh_token() -> str:
    """Fetch the token from the gh CLI. Output is never echoed (not even on failure)."""
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, check=True)  # noqa: S607
    except (subprocess.CalledProcessError, OSError):
        raise ScrapeError("`gh auth token` failed; run `gh auth login`") from None
    return out.stdout.strip()


def requests_transport(token: str) -> Transport:
    """Real transport. Exceptions are re-raised without request details (headers hold the token)."""
    sess = requests.Session()
    sess.headers["Authorization"] = f"bearer {token}"

    def post(payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        try:
            r = sess.post(API, json=payload, timeout=90)
        except requests.Timeout:
            raise TimeoutError("request timed out") from None
        except requests.RequestException as e:
            raise ConnectionError(type(e).__name__) from None
        try:
            body = r.json()
        except ValueError:
            body = {}
        return r.status_code, dict(r.headers), body

    return post


class Client:
    """GraphQL client: rate-limit pacing, secondary-limit and transient-error retries."""

    def __init__(self, transport: Transport) -> None:
        self.transport = transport
        self.cost = 0
        self.remaining: int | None = None
        self.total_cost = 0

    def _pace(self, rl: dict[str, Any]) -> None:
        self.cost = rl["cost"]
        self.total_cost += self.cost
        self.remaining = rl["remaining"]
        if rl["remaining"] < max(200, 3 * rl["cost"]):
            wait = max((_parse_ts(rl["resetAt"]) - _now()).total_seconds(), 0) + 5
            log.warning(
                "rate limit low (%s left); sleeping %.0fs until reset", rl["remaining"], wait
            )
            time.sleep(wait)

    def query(self, query: str, variables: dict[str, Any], tries: int = 6) -> dict[str, Any]:
        """POST one query; return ``data``. Raises TransientError / ScrapeError when giving up."""
        sec_tries = 6
        for attempt in range(max(tries, sec_tries)):
            try:
                status, headers, body = self.transport({"query": query, "variables": variables})
            except (TimeoutError, ConnectionError) as e:
                status, headers, body = 599, {}, {"message": str(e)}
            msg = str(body.get("message", "")) + " ".join(
                str(e.get("message", "")) for e in body.get("errors") or []
            )
            low = msg.lower()
            hinted = any(k in low for k in ("rate limit", "secondary", "abuse"))
            hinted = hinted or any(h.lower() == "retry-after" for h in headers)
            # a bare 403 (no access, SAML) is not a rate limit: fail fast, don't back off
            secondary = status == 429 or hinted
            bad_partial = (
                status == 200 and body.get("errors") and body.get("data") and not _benign(body)
            )
            if secondary:
                if attempt + 1 >= sec_tries:
                    raise ScrapeError(f"secondary rate limit persisted: HTTP {status}: {msg[:200]}")
                ra = next((v for h, v in headers.items() if h.lower() == "retry-after"), None)
                wait = float(ra or 60 * 2**attempt)
                log.warning("secondary rate limit (HTTP %s); sleeping %.0fs", status, wait)
                time.sleep(wait)
            elif status >= 500 or bad_partial:
                if attempt + 1 >= tries:
                    raise TransientError(f"HTTP {status} after {tries} tries: {msg[:200]}")
                wait = min(2 ** (attempt + 1), 60) + random.uniform(0, 1)  # noqa: S311
                log.warning("transient HTTP %s; retry in %.1fs", status, wait)
                time.sleep(wait)
            elif status != 200 or (body.get("errors") and not body.get("data")):
                raise ScrapeError(f"GraphQL failure HTTP {status}: {msg[:300]}")
            else:
                if body.get("errors"):
                    log.warning("ignored %d inaccessible-source errors", len(body["errors"]))
                data = body["data"]
                if data.get("rateLimit"):
                    self._pace(data["rateLimit"])
                return data
        raise ScrapeError("retries exhausted")


def _benign(body: dict[str, Any]) -> bool:
    """True if every error is NOT_FOUND/FORBIDDEN under timelineItems/closer/source.

    Those are cross-repo private sources GitHub cannot resolve; any other error next to
    partial ``data`` may mean silently missing fields, so the caller retries instead.
    """
    return all(
        e.get("type") in ("NOT_FOUND", "FORBIDDEN")
        and {"timelineItems", "closer", "source"} & set(map(str, e.get("path") or []))
        for e in body["errors"]
    )


def _atomic_write(path: Path, write: Callable[[Path], None]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    write(tmp)
    os.replace(tmp, path)


class Scraper:
    """Scrapes one repo/kind; one checkpointed unit per UTC day."""

    def __init__(self, client: Client, repo: str, kind: str, data_dir: Path) -> None:
        self.client, self.repo, self.kind = client, repo, kind
        self.base = data_dir / "raw" / repo.replace("/", "__") / kind
        self.ckpt_path = self.base / "checkpoint.json"
        self.ckpt: dict[str, Any] = {"windows": {}}
        if self.ckpt_path.exists():
            self.ckpt = json.loads(self.ckpt_path.read_text(encoding="utf-8"))
        fields = ISSUE_FIELDS if kind == "issues" else PR_FIELDS
        self.search_q = SEARCH_Q.replace("{FIELDS}", fields)
        if kind == "prs":  # $tl is only used by the issue timeline; unused variables are errors
            self.search_q = self.search_q.replace(", $tl:String", "")
        self.page_size = PAGE_SIZES[0]

    def _search_q(self, start: datetime, end: datetime) -> str:
        t = "issue" if self.kind == "issues" else "pr"
        return f"repo:{self.repo} is:{t} created:{_fmt(start)}..{_fmt(end)}"

    def _count(self, start: datetime, end: datetime) -> int:
        q = "query($q:String!){ rateLimit{cost remaining resetAt} search(query:$q, type:ISSUE, first:1){issueCount}}"
        data = self.client.query(q, {"q": self._search_q(start, end)})
        return int(data["search"]["issueCount"])

    def _complete_timeline(self, node: dict[str, Any]) -> None:
        """Follow timeline pagination so the saved record has every event."""
        tl = node.get("timelineItems")
        while tl and tl["pageInfo"]["hasNextPage"]:
            data = self.client.query(
                TIMELINE_Q, {"id": node["id"], "tl": tl["pageInfo"]["endCursor"]}
            )
            more = data["node"]["timelineItems"]
            tl["nodes"].extend(more["nodes"])
            tl["pageInfo"] = more["pageInfo"]

    def _fetch_leaf(self, start: datetime, end: datetime, count: int) -> list[dict[str, Any]]:
        """Fetch all rows of a window known to be under the cap; assert completeness."""
        for i, size in enumerate(PAGE_SIZES):
            if size > self.page_size:
                continue
            try:
                return self._pages(
                    start, end, count, size, tries=6 if i == len(PAGE_SIZES) - 1 else 3
                )
            except TransientError:
                if size == PAGE_SIZES[-1]:
                    raise
                log.warning("reducing page size %s -> next for %s", size, _fmt(start))
                self.page_size = PAGE_SIZES[i + 1]
        raise ScrapeError("unreachable")

    def _pages(
        self, start: datetime, end: datetime, count: int, size: int, tries: int
    ) -> list[dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        after: str | None = None
        while True:
            data = self.client.query(
                self.search_q,
                {"q": self._search_q(start, end), "n": size, "after": after},
                tries=tries,
            )
            s = data["search"]
            for node in s["nodes"]:
                if node:
                    self._complete_timeline(node)
                    node["_scraped_at"] = _fmt(_now())
                    node["_schema"] = SCHEMA_VERSION
                    rows[node["id"]] = node
            if not s["pageInfo"]["hasNextPage"]:
                break
            after = s["pageInfo"]["endCursor"]
        if len(rows) != count:
            raise ScrapeError(
                f"completeness: saved {len(rows)} rows != issueCount {count} "
                f"for {self.repo} {self.kind} {_fmt(start)}..{_fmt(end)}"
            )
        return list(rows.values())

    def fetch_window(self, start: datetime, end: datetime) -> tuple[list[dict[str, Any]], int]:
        """Rows + issueCount for [start, end]; halves the window while the search cap is hit."""
        count = self._count(start, end)
        if count < SEARCH_CAP:
            return (self._fetch_leaf(start, end, count) if count else []), count
        if start >= end:
            raise ScrapeError(f"{count} results in a single second at {_fmt(start)}")
        mid = start + (end - start + timedelta(seconds=1)) / 2
        mid = mid.replace(microsecond=0)
        left, lc = self.fetch_window(start, mid - timedelta(seconds=1))
        right, rc = self.fetch_window(mid, end)
        return left + right, lc + rc

    def _write_day(self, day: datetime, rows: list[dict[str, Any]]) -> None:
        path = self.base / f"{day:%Y}" / f"{day:%Y-%m-%d}.jsonl.gz"

        def w(tmp: Path) -> None:
            with gzip.open(tmp, "wt", encoding="utf-8") as f:
                for r in sorted(rows, key=lambda r: r["number"]):
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")

        _atomic_write(path, w)

    def _save_ckpt(self) -> None:
        _atomic_write(
            self.ckpt_path,
            lambda tmp: tmp.write_text(json.dumps(self.ckpt, indent=1), encoding="utf-8"),
        )

    def run(self, since: datetime, until: datetime, max_windows: int | None = None) -> int:
        """Scrape day windows in [since, until]; return the number of windows fetched."""
        done, day = 0, since.replace(hour=0, minute=0, second=0, microsecond=0)
        while day <= until and (max_windows is None or done < max_windows):
            end = min(day + timedelta(days=1) - timedelta(seconds=1), until)
            # a clipped (partial) day gets a distinct key so a later, longer run refetches it
            key = (
                f"{day:%Y-%m-%d}"
                if end.hour == 23 and end.minute == 59
                else f"{day:%Y-%m-%d}..{_fmt(end)}"
            )
            day_start, day = day, day + timedelta(days=1)
            if key in self.ckpt["windows"]:
                continue
            t0 = time.monotonic()
            rows, count = self.fetch_window(max(day_start, since), end)
            if rows:
                self._write_day(day_start, rows)
            self.ckpt["windows"][key] = {
                "issueCount": count,
                "rows": len(rows),
                "completed_at": _fmt(_now()),
            }
            self._save_ckpt()
            done += 1
            log.info(
                "%s rows=%d cost=%d remaining=%s %.1fs",
                key,
                len(rows),
                self.client.cost,
                self.client.remaining,
                time.monotonic() - t0,
            )
        return done


def summarize(data_dir: Path) -> dict[str, dict[str, int]]:
    """Per repo/kind/year row counts, from disk files and from checkpoints (no network)."""
    out: dict[str, dict[str, int]] = {}
    for ck in sorted((data_dir / "raw").glob("*__*/*/checkpoint.json")):
        label = f"{ck.parent.parent.name}/{ck.parent.name}"
        by_year: Counter[str] = Counter()
        for k, v in json.loads(ck.read_text(encoding="utf-8"))["windows"].items():
            by_year[f"ckpt:{k[:4]}"] += v["rows"]
        for f in ck.parent.glob("*/*.jsonl.gz"):
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                by_year[f"disk:{f.parent.name}"] += sum(1 for _ in fh)
        out[label] = dict(sorted(by_year.items()))
    return out


def _date_arg(s: str, end_of_day: bool = False) -> datetime:
    t = datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=UTC)
    return t + timedelta(days=1, seconds=-1) if end_of_day else t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GitHub GraphQL raw scraper")
    ap.add_argument("--repo")
    ap.add_argument("--kind", choices=["issues", "prs"])
    ap.add_argument("--since", default="2023-01-01")
    ap.add_argument("--until", default=None, help="YYYY-MM-DD (inclusive); default now (UTC)")
    default_dir = os.environ.get("TRIAGEIQ_DATA_DIR", r"D:\triageiq-data")
    ap.add_argument("--data-dir", type=Path, default=Path(default_dir))
    ap.add_argument("--max-windows", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.summary:
        print(json.dumps(summarize(a.data_dir), indent=1))
        return 0
    if not a.repo or not a.kind:
        ap.error("--repo and --kind are required unless --summary")
    until = _date_arg(a.until, True) if a.until else _now().replace(microsecond=0)
    scraper = Scraper(Client(requests_transport(gh_token())), a.repo, a.kind, a.data_dir)
    n = scraper.run(_date_arg(a.since), until, a.max_windows)
    log.info("done: %d windows fetched, total cost %d", n, scraper.client.total_cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
