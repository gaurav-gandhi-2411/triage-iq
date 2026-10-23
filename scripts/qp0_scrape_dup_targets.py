from __future__ import annotations

# Duplicate-target scrape. The main scrape saved MarkedAsDuplicateEvent.canonical, which on an issue's
# own timeline is always that issue itself; the OTHER issue of the pair is in `.duplicate`, and the
# real target of a UI "close as duplicate" is ClosedEvent.duplicateOf (null for slash-command
# closures). This saves those events (with their pair partners) for every issue with a duplicate
# signal, under <data-dir>/raw/<owner>__<repo>/dup_targets/<number // 1000, 5 digits>.jsonl.gz
# (_schema 2) plus an atomic checkpoint.json, same idiom as pr_files. Reuses PrFilesScraper's
# storage / resume code, the tolerant (missing-alias) transport and PacedClient. Sequential: the
# GraphQL budget is shared with other runs.
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts, as qp0_derive_*.py
import qp0_scrape_comments as qcm  # noqa: E402
import qp0_scrape_github as sg  # noqa: E402
import qp0_scrape_pr_files as pf  # noqa: E402

log = logging.getLogger("qp0dt")
BATCH_SIZES = (10, 5, 2, 1)
# Consecutive clean queries at a reduced size before stepping back up one size. A burst of 5xx/499
# used to pin the batch at 1 for the rest of a run (GraphQL cost is 1 per query at any size).
REGROW_AFTER = 50
MAX_PAGES = 10  # 1,000 duplicate-type events on one issue would be a data problem, not a page size
_T = "__typename ... on Issue{number repository{nameWithOwner}} ... on PullRequest{number repository{nameWithOwner}}"
_NODES = (
    "nodes{__typename"
    f" ... on MarkedAsDuplicateEvent{{createdAt actor{{login}} isCrossRepository canonical{{{_T}}} duplicate{{{_T}}}}}"
    f" ... on UnmarkedAsDuplicateEvent{{createdAt canonical{{{_T}}} duplicate{{{_T}}}}}"
    f" ... on ClosedEvent{{createdAt stateReason duplicateOf{{{_T}}}}}}}"
)
_TL = (
    "timelineItems(first:100{AFTER}, itemTypes:[CLOSED_EVENT, MARKED_AS_DUPLICATE_EVENT, UNMARKED_AS_DUPLICATE_EVENT]){totalCount pageInfo{hasNextPage endCursor} "
    + _NODES
    + "}"
)
PAGE_Q = (
    "query($id:ID!, $c:String){ rateLimit{cost remaining resetAt} node(id:$id){ ... on Issue{ "
    + _TL.replace("{AFTER}", ", after:$c")
    + "}}}"
)


def batch_query(numbers: list[int]) -> str:
    """One aliased `issue(number:N)` field per number (alias p<N>, the prefix `tolerant` expects)."""
    tl = _TL.replace("{AFTER}", "")
    fields = " ".join(
        f"p{n}: issue(number:{n}){{number id state stateReason {tl}}}" for n in numbers
    )
    return (
        "query($owner:String!, $name:String!){ rateLimit{cost remaining resetAt} "
        f"repository(owner:$owner, name:$name){{ {fields} }}}}"
    )


def build_scope(derived_dir: Path) -> list[int]:
    """Issues with ANY duplicate signal: a *duplicate* label, stateReason DUPLICATE, or a
    MarkedAsDuplicate event (counted at derive time in canon_self / canon_*_repo columns)."""
    cols = ["number", "labels_final", "closed_as_duplicate", "canon_self", "canon_foreign_repo"]
    iss = pd.read_parquet(derived_dir / "issues.parquet", columns=cols)
    labelled = iss["labels_final"].map(lambda ls: any("duplicate" in x.lower() for x in ls))
    keep = (
        labelled
        | iss["closed_as_duplicate"]
        | (iss["canon_self"] > 0)
        | (iss["canon_foreign_repo"] > 0)
    )
    return sorted(map(int, iss.loc[keep, "number"]))


class DupTargetScraper(pf.PrFilesScraper):
    """Saves the duplicate-type timeline events of given issues; resumable and atomic."""

    def __init__(self, client: sg.Client, repo: str, data_dir: Path, batch: int = 10) -> None:
        self.client, self.repo = client, repo
        self.owner, self.name = repo.split("/", 1)
        self.base = data_dir / "raw" / repo.replace("/", "__") / "dup_targets"
        self.ckpt_path = self.base / "checkpoint.json"
        self.ckpt: dict[str, Any] = {"windows": {}, "done": [], "missing": []}
        if self.ckpt_path.exists():
            self.ckpt = json.loads(self.ckpt_path.read_text(encoding="utf-8"))
        self.done = set(self.ckpt["done"])
        self.sizes = [s for s in BATCH_SIZES if s <= batch] or [1]
        self.max_batch, self._ok_streak = self.sizes[0], 0
        self.stats: dict[str, int] = {"batches": 0, "issues": 0, "events": 0, "missing": 0}

    def _complete(self, issue: dict[str, Any]) -> None:
        """Follow event pagination; the saved events must equal totalCount or we fail loudly."""
        tl = issue["timelineItems"]
        tl["nodes"] = [n for n in tl["nodes"] if n]
        for _ in range(MAX_PAGES):
            if not tl["pageInfo"]["hasNextPage"]:
                break
            more = self.client.query(PAGE_Q, {"id": issue["id"], "c": tl["pageInfo"]["endCursor"]})
            m = more["node"]["timelineItems"]
            tl["nodes"].extend(n for n in m["nodes"] if n)
            tl["pageInfo"] = m["pageInfo"]
        if len(tl["nodes"]) != tl["totalCount"]:
            raise sg.ScrapeError(
                f"completeness: issue #{issue['number']} saved {len(tl['nodes'])} events "
                f"!= totalCount {tl['totalCount']}"
            )

    def _note_success(self) -> None:
        """Count a clean query; after REGROW_AFTER in a row, step the batch size back up one."""
        self._ok_streak += 1
        if self._ok_streak < REGROW_AFTER or self.sizes[0] >= self.max_batch:
            return
        self._ok_streak = 0
        new = min(s for s in BATCH_SIZES if self.sizes[0] < s <= self.max_batch)
        self.sizes = [s for s in BATCH_SIZES if s <= new]
        log.info("%d clean queries in a row; regrowing batch to %d", REGROW_AFTER, new)

    def fetch(self, numbers: list[int]) -> list[dict[str, Any]]:
        """Rows for ``numbers``; shrinks the batch size on persistent transient errors and
        regrows it (up to the configured maximum) after a run of clean queries."""
        try:
            data = self.client.query(
                batch_query(numbers),
                {"owner": self.owner, "name": self.name},
                tries=3 if len(self.sizes) > 1 else 6,
            )
        except sg.TransientError:
            self._ok_streak = 0
            if len(numbers) == 1:
                raise
            self.sizes = [s for s in self.sizes if s < len(numbers)] or [1]
            log.warning("batch of %d failed; reducing to %d", len(numbers), self.sizes[0])
            step = self.sizes[0]
            return [
                r for i in range(0, len(numbers), step) for r in self.fetch(numbers[i : i + step])
            ]
        self._note_success()
        rows = []
        for n in numbers:
            issue = (data.get("repository") or {}).get(f"p{n}")
            stamp = {"_scraped_at": sg._fmt(sg._now()), "_schema": sg.SCHEMA_VERSION}
            if not issue:
                log.warning(
                    "issue #%d not found in %s (deleted / transferred / a PR): recorded",
                    n,
                    self.repo,
                )
                rows.append({"number": n, "missing": True, **stamp})
                continue
            self._complete(issue)
            rows.append({**issue, **stamp})
        return rows

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
            events = sum(len(r["timelineItems"]["nodes"]) for r in rows if not r.get("missing"))
            miss = sum(bool(r.get("missing")) for r in rows)
            for k, v in (
                ("batches", 1),
                ("issues", len(rows)),
                ("events", events),
                ("missing", miss),
            ):
                self.stats[k] += v
            log.info("batch %d issues=%d events=%d cost=%d remaining=%s %.1fs", batches, len(rows), events,
                     self.client.cost, self.client.remaining, time.monotonic() - t0)  # fmt: skip
        return batches


def summarize(data_dir: Path) -> dict[str, dict[str, int]]:
    """Per repo: checkpoint counts and how many saved rows carry a usable target (no network)."""
    out: dict[str, dict[str, int]] = {}
    for ck in sorted((data_dir / "raw").glob("*__*/dup_targets/checkpoint.json")):
        c = dict.fromkeys(("rows", "closed_duplicate_of", "marked_with_duplicate"), 0)
        for f in ck.parent.glob("*.jsonl.gz"):
            for r in DupTargetScraper._read(f):
                c["rows"] += 1
                nodes = [] if r.get("missing") else r["timelineItems"]["nodes"]
                c["closed_duplicate_of"] += any(
                    n["__typename"] == "ClosedEvent" and n.get("duplicateOf") for n in nodes
                )
                c["marked_with_duplicate"] += any(n["__typename"] == "MarkedAsDuplicateEvent" and n.get("duplicate") for n in nodes)  # fmt: skip
        j = json.loads(ck.read_text(encoding="utf-8"))
        out[ck.parent.parent.name] = {
            "ckpt_done": len(j["done"]),
            "ckpt_missing": len(j["missing"]),
            **c,
        }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape duplicate-target events for issues")
    ap.add_argument("--repo")
    ap.add_argument("--numbers-file", type=Path, help="one issue number per line")
    ap.add_argument(
        "--scope-from-derived", action="store_true", help="issues with a duplicate signal"
    )
    default_dir = os.environ.get("TRIAGEIQ_DATA_DIR", r"D:\triageiq-data")
    ap.add_argument("--data-dir", type=Path, default=Path(default_dir))
    ap.add_argument(
        "--read-dir", type=Path, default=None, help="where derived/ is READ from (scope)"
    )
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.summary:
        print(json.dumps(summarize(a.data_dir), indent=1))
        return 0
    if not a.repo or not (a.numbers_file or a.scope_from_derived):
        ap.error(
            "--repo and (--numbers-file or --scope-from-derived) are required unless --summary"
        )
    if a.numbers_file:
        numbers = [int(x) for x in a.numbers_file.read_text(encoding="utf-8").split()]
    else:
        numbers = build_scope((a.read_dir or a.data_dir) / "derived" / a.repo.replace("/", "__"))
        log.info("scope: %d issues with a duplicate signal", len(numbers))
    client = qcm.PacedClient(pf.tolerant(sg.requests_transport(sg.gh_token())))
    scraper = DupTargetScraper(client, a.repo, a.data_dir, a.batch)
    n = scraper.run(numbers, a.max_batches)
    log.info("done: %d batches, %s, total cost %d", n, scraper.stats, client.total_cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
