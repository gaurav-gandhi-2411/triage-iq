from __future__ import annotations

# Comments-only re-scrape (quality-push Phase 0): the main scraper kept only comments(last:5);
# the comment-limit audit (D79 item 5) found that material, so this saves the FULL thread for a
# list of issue numbers. Raw data only, under <data-dir>/raw/<owner>__<repo>/comments/:
# <issue_number // 1000, 5 digits>.jsonl.gz (one JSON line per issue, _schema 2) plus an atomic
# checkpoint.json. Transport / token / retry / error types come from qp0_scrape_github.py, whose
# file budget is full. Sequential on purpose: the GraphQL budget is shared with other scrapes.
import argparse
import gzip
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts, as qp0_derive_*.py
import qp0_scrape_github as sg  # noqa: E402

log = logging.getLogger("qp0c")
BATCH_SIZES = (10, 5, 2, 1)  # shrink on persistent 5xx / timeouts, like the main scraper's pages
MIN_REMAINING = 800  # shared quota: stay well clear of exhausting it for the other scrape
_COMMENT = (
    "id databaseId body createdAt updatedAt authorAssociation isMinimized minimizedReason "
    "author{login __typename}"
)
_COMMENTS = f"comments(first:100{{AFTER}}){{totalCount pageInfo{{hasNextPage endCursor}} nodes{{{_COMMENT}}}}}"
_EVENTS = (
    "timelineItems(last:10, itemTypes:[CLOSED_EVENT, REOPENED_EVENT]){nodes{__typename "
    "... on ClosedEvent{createdAt stateReason} ... on ReopenedEvent{createdAt}}}"
)
PAGE_Q = (
    "query($id:ID!, $c:String){ rateLimit{cost remaining resetAt} node(id:$id){ ... on Issue{ "
    + _COMMENTS.replace("{AFTER}", ", after:$c")
    + "}}}"
)


def batch_query(numbers: list[int]) -> str:
    """One aliased `issue(number:N)` field per number (aliases i<N>), plus the rate-limit cost."""
    body = _COMMENTS.replace("{AFTER}", "")
    fields = " ".join(f"i{n}: issue(number:{n}){{number id {body} {_EVENTS}}}" for n in numbers)
    return (
        "query($owner:String!, $name:String!){ rateLimit{cost remaining resetAt} "
        f"repository(owner:$owner, name:$name){{ {fields} }}}}"
    )


class PacedClient(sg.Client):
    """Client that sleeps until the reset when fewer than MIN_REMAINING points are left."""

    def _pace(self, rl: dict[str, Any]) -> None:
        self.cost, self.remaining = rl["cost"], rl["remaining"]
        self.total_cost += self.cost
        if rl["remaining"] < MIN_REMAINING:
            wait = max((sg._parse_ts(rl["resetAt"]) - sg._now()).total_seconds(), 0) + 5
            log.warning("rate limit low (%s left); sleeping %.0fs", rl["remaining"], wait)
            time.sleep(wait)


# Consecutive clean queries at a reduced size before stepping back up one size. A burst of 5xx/499
# used to pin the batch at 1 for the rest of a run (GraphQL cost is 1 per query at any size).
REGROW_AFTER = 50


class BatchRegrow:
    """Mixin for the batch scrapers: ``sizes`` shrinks on errors, this grows it back."""

    sizes: list[int]
    max_batch: int
    _ok_streak: int

    def _note_success(self) -> None:
        """Count a clean query; after REGROW_AFTER in a row, step the batch size back up one."""
        self._ok_streak += 1
        if self._ok_streak < REGROW_AFTER or self.sizes[0] >= self.max_batch:
            return
        self._ok_streak = 0
        new = min(s for s in BATCH_SIZES if self.sizes[0] < s <= self.max_batch)
        self.sizes = [s for s in BATCH_SIZES if s <= new]
        log.info("%d clean queries in a row; regrowing batch to %d", REGROW_AFTER, new)


class CommentScraper:
    """Saves the complete comment thread + closed/reopened events for given issue numbers."""

    def __init__(self, client: sg.Client, repo: str, data_dir: Path, batch: int = 10) -> None:
        self.client, self.repo = client, repo
        self.owner, self.name = repo.split("/", 1)
        self.base = data_dir / "raw" / repo.replace("/", "__") / "comments"
        self.ckpt_path = self.base / "checkpoint.json"
        # "windows" keeps the main scraper's --summary glob (raw/*/*/checkpoint.json) working
        self.ckpt: dict[str, Any] = {"windows": {}, "done": []}
        if self.ckpt_path.exists():
            self.ckpt = json.loads(self.ckpt_path.read_text(encoding="utf-8"))
        self.done = set(self.ckpt["done"])
        self.sizes = [s for s in BATCH_SIZES if s <= batch] or [1]
        self.stats = {"batches": 0, "issues": 0, "comments": 0}

    def _complete(self, issue: dict[str, Any]) -> None:
        """Follow comment pagination; the saved thread must equal totalCount or we fail loudly."""
        c = issue["comments"]
        while c["pageInfo"]["hasNextPage"]:
            node = self.client.query(PAGE_Q, {"id": issue["id"], "c": c["pageInfo"]["endCursor"]})
            more = node["node"]["comments"]
            c["nodes"].extend(more["nodes"])
            c["pageInfo"] = more["pageInfo"]
        c["nodes"] = [n for n in c["nodes"] if n]
        if len(c["nodes"]) != c["totalCount"]:
            raise sg.ScrapeError(
                f"completeness: issue #{issue['number']} saved {len(c['nodes'])} comments "
                f"!= totalCount {c['totalCount']}"
            )

    def fetch(self, numbers: list[int]) -> list[dict[str, Any]]:
        """Complete rows for ``numbers``; shrinks the batch size on persistent transient errors."""
        try:
            data = self.client.query(
                batch_query(numbers),
                {"owner": self.owner, "name": self.name},
                tries=3 if len(self.sizes) > 1 else 6,
            )
        except sg.TransientError:
            if len(numbers) == 1:
                raise
            self.sizes = [s for s in self.sizes if s < len(numbers)] or [1]
            log.warning("batch of %d failed; reducing to %d", len(numbers), self.sizes[0])
            half = self.sizes[0]
            return [
                r for i in range(0, len(numbers), half) for r in self.fetch(numbers[i : i + half])
            ]
        rows = []
        for n in numbers:
            issue = (data.get("repository") or {}).get(f"i{n}")
            if not issue:
                raise sg.ScrapeError(f"issue #{n} not found in {self.repo} (deleted/transferred?)")
            self._complete(issue)
            events = [e for e in issue["timelineItems"]["nodes"] if e]
            rows.append(
                {
                    "number": issue["number"],
                    "id": issue["id"],
                    "comments": issue["comments"],
                    "events": events,
                    "_scraped_at": sg._fmt(sg._now()),
                    "_schema": sg.SCHEMA_VERSION,
                }  # fmt: skip
            )
        return rows

    def _path(self, number: int) -> Path:
        return self.base / f"{number // 1000:05d}.jsonl.gz"

    def _write(self, rows: list[dict[str, Any]]) -> None:
        """Merge ``rows`` into their per-thousand files (atomic rewrite), then the checkpoint."""
        for path in sorted({self._path(r["number"]) for r in rows}):
            merged = {r["number"]: r for r in self._read(path)}
            merged.update({r["number"]: r for r in rows if self._path(r["number"]) == path})

            def w(tmp: Path, merged: dict[int, dict[str, Any]] = merged) -> None:
                with gzip.open(tmp, "wt", encoding="utf-8") as f:
                    for n in sorted(merged):
                        f.write(json.dumps(merged[n], ensure_ascii=False) + "\n")

            sg._atomic_write(path, w)
        self.done |= {r["number"] for r in rows}
        self.ckpt["done"] = sorted(self.done)
        sg._atomic_write(
            self.ckpt_path,
            lambda tmp: tmp.write_text(json.dumps(self.ckpt, indent=1), encoding="utf-8"),
        )

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def run(self, numbers: list[int], max_batches: int | None = None) -> int:
        """Fetch every not-yet-done number; returns the number of batches executed."""
        todo = [n for n in sorted(set(numbers)) if n not in self.done]
        batches = 0
        while todo and (max_batches is None or batches < max_batches):
            chunk, todo = todo[: self.sizes[0]], todo[self.sizes[0] :]
            t0 = time.monotonic()
            rows = self.fetch(chunk)
            self._write(rows)
            batches += 1
            n_comments = sum(len(r["comments"]["nodes"]) for r in rows)
            self.stats["batches"] += 1
            self.stats["issues"] += len(rows)
            self.stats["comments"] += n_comments
            log.info(
                "batch %d issues=%d comments=%d cost=%d remaining=%s %.1fs",
                batches, len(rows), n_comments, self.client.cost, self.client.remaining,
                time.monotonic() - t0,
            )  # fmt: skip
        return batches


def summarize(data_dir: Path) -> dict[str, dict[str, int]]:
    """Per repo: done count from the checkpoint and issue/comment counts from disk (no network)."""
    out: dict[str, dict[str, int]] = {}
    for ck in sorted((data_dir / "raw").glob("*__*/comments/checkpoint.json")):
        issues = comments = 0
        for f in ck.parent.glob("*.jsonl.gz"):
            for r in CommentScraper._read(f):
                issues, comments = issues + 1, comments + len(r["comments"]["nodes"])
        done = len(json.loads(ck.read_text(encoding="utf-8"))["done"])
        out[ck.parent.parent.name] = {
            "ckpt_done": done,
            "disk_issues": issues,
            "comments": comments,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Full-thread comment re-scrape for given issues")
    ap.add_argument("--repo")
    ap.add_argument("--numbers-file", type=Path, help="one issue number per line")
    default_dir = os.environ.get("TRIAGEIQ_DATA_DIR", r"D:\triageiq-data")
    ap.add_argument("--data-dir", type=Path, default=Path(default_dir))
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.summary:
        print(json.dumps(summarize(a.data_dir), indent=1))
        return 0
    if not a.repo or not a.numbers_file:
        ap.error("--repo and --numbers-file are required unless --summary")
    lines = a.numbers_file.read_text(encoding="utf-8").split()
    numbers = [int(x) for x in lines]
    scraper = CommentScraper(
        PacedClient(sg.requests_transport(sg.gh_token())), a.repo, a.data_dir, a.batch
    )
    n = scraper.run(numbers, a.max_batches)
    log.info("done: %d batches, %s, total cost %d", n, scraper.stats, scraper.client.total_cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
