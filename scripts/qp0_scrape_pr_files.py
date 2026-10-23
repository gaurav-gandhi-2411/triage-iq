from __future__ import annotations

# PR files-changed scrape (D83: the k8s retrieval positives are prior PRs touching a file that the
# issue's eventual fix PR changed). Raw data only, under <data-dir>/raw/<owner>__<repo>/pr_files/:
# <pr_number // 1000, 5 digits>.jsonl.gz (one JSON line per PR, _schema 2) plus an atomic
# checkpoint.json (done + missing lists). Client / retry / token come from qp0_scrape_github.py and
# PacedClient from qp0_scrape_comments.py. Sequential: the GraphQL budget is shared with other runs.
# Files are saved up to 300 per PR; a larger PR keeps totalCount and files_truncated=true (mostly
# generated / vendor sweeps). A missing / deleted / transferred PR is recorded as {number, missing}.
import argparse
import gzip
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts, as qp0_derive_*.py
import qp0_scrape_comments as qcm  # noqa: E402
import qp0_scrape_github as sg  # noqa: E402

log = logging.getLogger("qp0pf")
BATCH_SIZES = (10, 5, 2, 1)
MAX_FILES = 300  # three pages of 100
_FILES = "files(first:100{AFTER}){totalCount pageInfo{hasNextPage endCursor} nodes{path additions deletions changeType}}"
_FIELDS = (
    "number id state merged mergedAt createdAt baseRefName headRefName title changedFiles "
    "additions deletions mergeCommit{oid} "
)
PAGE_Q = (
    "query($id:ID!, $c:String){ rateLimit{cost remaining resetAt} node(id:$id){ ... on PullRequest{ "
    + _FILES.replace("{AFTER}", ", after:$c")
    + "}}}"
)


def batch_query(numbers: list[int]) -> str:
    """One aliased `pullRequest(number:N)` field per number (aliases p<N>), plus the rate-limit cost."""
    files = _FILES.replace("{AFTER}", "")
    fields = " ".join(f"p{n}: pullRequest(number:{n}){{{_FIELDS}{files}}}" for n in numbers)
    return (
        "query($owner:String!, $name:String!){ rateLimit{cost remaining resetAt} "
        f"repository(owner:$owner, name:$name){{ {fields} }}}}"
    )


def tolerant(transport: sg.Transport) -> sg.Transport:
    """Drop NOT_FOUND errors on pullRequest aliases so the client does not retry missing PRs.

    The shared client treats any other error next to partial data as transient; a missing PR is
    permanent and recorded as such (the alias comes back null).
    """

    def post(payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        status, headers, body = transport(payload)
        errs = body.get("errors") or []
        if errs and all(
            e.get("type") == "NOT_FOUND" and str((e.get("path") or ["", ""])[1]).startswith("p")
            for e in errs
        ):
            body = {**body, "errors": None}
        return status, headers, body

    return post


def build_scope(data_dir: Path, derived_dir: Path, repo: str) -> list[int]:
    """Merged PRs of the raw scrape UNION merged same-repo closer-link PRs, deduped and sorted."""
    nums: set[int] = set()
    for f in sorted((data_dir / "raw" / repo.replace("/", "__") / "prs").glob("*/*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            nums |= {r["number"] for r in map(json.loads, fh) if r.get("merged")}
    links = pd.read_parquet(
        derived_dir / "fix_links.parquet", columns=["pr", "method", "pr_merged"]
    )
    ok = links[(links["method"] == "closed_event_closer_pr") & links["pr_merged"]]
    return sorted(nums | set(map(int, ok["pr"])))


class PrFilesScraper(qcm.BatchRegrow):
    """Saves files-changed + metadata for given PR numbers; resumable and atomic."""

    def __init__(self, client: sg.Client, repo: str, data_dir: Path, batch: int = 10) -> None:
        self.client, self.repo = client, repo
        self.owner, self.name = repo.split("/", 1)
        self.base = data_dir / "raw" / repo.replace("/", "__") / "pr_files"
        self.ckpt_path = self.base / "checkpoint.json"
        self.ckpt: dict[str, Any] = {"windows": {}, "done": [], "missing": []}
        if self.ckpt_path.exists():
            self.ckpt = json.loads(self.ckpt_path.read_text(encoding="utf-8"))
        self.done = set(self.ckpt["done"])
        self.sizes = [s for s in BATCH_SIZES if s <= batch] or [1]
        self.max_batch, self._ok_streak = self.sizes[0], 0
        self.stats: Counter[str] = Counter()

    def _complete(self, pr: dict[str, Any]) -> None:
        """Follow file pagination up to MAX_FILES; otherwise totalCount must match what was saved."""
        f = pr["files"]
        f["nodes"] = [n for n in f["nodes"] if n]
        while f["pageInfo"]["hasNextPage"] and len(f["nodes"]) < MAX_FILES:
            more = self.client.query(PAGE_Q, {"id": pr["id"], "c": f["pageInfo"]["endCursor"]})
            m = more["node"]["files"]
            f["nodes"].extend(n for n in m["nodes"] if n)
            f["pageInfo"] = m["pageInfo"]
        pr["files_truncated"] = bool(f["pageInfo"]["hasNextPage"]) and len(f["nodes"]) >= MAX_FILES
        if pr["files_truncated"]:
            f["nodes"] = f["nodes"][:MAX_FILES]
        elif len(f["nodes"]) != f["totalCount"]:
            raise sg.ScrapeError(
                f"completeness: PR #{pr['number']} saved {len(f['nodes'])} files "
                f"!= totalCount {f['totalCount']}"
            )

    def fetch(self, numbers: list[int]) -> list[dict[str, Any]]:
        """Rows for ``numbers``; shrinks the batch size on persistent transient errors and
        regrows it (qcm.BatchRegrow) after a run of clean queries."""
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
            pr = (data.get("repository") or {}).get(f"p{n}")
            stamp = {"_scraped_at": sg._fmt(sg._now()), "_schema": sg.SCHEMA_VERSION}
            if not pr:
                log.warning(
                    "PR #%d not found in %s (deleted / transferred / not a PR): recorded",
                    n,
                    self.repo,
                )
                rows.append({"number": n, "missing": True, **stamp})
                continue
            self._complete(pr)
            rows.append({**pr, **stamp})
        return rows

    def _path(self, number: int) -> Path:
        return self.base / f"{number // 1000:05d}.jsonl.gz"

    @staticmethod
    def _read(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    def _write(self, rows: list[dict[str, Any]]) -> None:
        """Merge ``rows`` into their per-thousand files (atomic), then the checkpoint."""
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
        self.ckpt["missing"] = sorted(
            set(self.ckpt["missing"]) | {r["number"] for r in rows if r.get("missing")}
        )
        sg._atomic_write(
            self.ckpt_path,
            lambda tmp: tmp.write_text(json.dumps(self.ckpt, indent=1), encoding="utf-8"),
        )

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
            files = sum(len(r["files"]["nodes"]) for r in rows if not r.get("missing"))
            self.stats.update(
                batches=1,
                prs=len(rows),
                files=files,
                missing=sum(bool(r.get("missing")) for r in rows),
            )
            log.info("batch %d prs=%d files=%d cost=%d remaining=%s %.1fs", batches, len(rows), files,
                     self.client.cost, self.client.remaining, time.monotonic() - t0)  # fmt: skip
        return batches


def summarize(data_dir: Path) -> dict[str, dict[str, Any]]:
    """Per repo: checkpoint counts and on-disk PR / file / base-branch counts (no network)."""
    out: dict[str, dict[str, Any]] = {}
    for ck in sorted((data_dir / "raw").glob("*__*/pr_files/checkpoint.json")):
        prs = files = 0
        bases: Counter[str] = Counter()
        for f in ck.parent.glob("*.jsonl.gz"):
            for r in PrFilesScraper._read(f):
                prs += 1
                if not r.get("missing"):
                    files += len(r["files"]["nodes"])
                    bases[r["baseRefName"]] += 1
        c = json.loads(ck.read_text(encoding="utf-8"))
        out[ck.parent.parent.name] = {"ckpt_done": len(c["done"]), "ckpt_missing": len(c["missing"]),
                                      "disk_prs": prs, "files": files, "base_ref_top": bases.most_common(8)}  # fmt: skip
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape files changed for PRs")
    ap.add_argument("--repo")
    ap.add_argument("--numbers-file", type=Path, help="one PR number per line")
    ap.add_argument("--scope-from-raw", action="store_true", help="merged raw PRs + fix-link PRs")
    default_dir = os.environ.get("TRIAGEIQ_DATA_DIR", r"D:\triageiq-data")
    ap.add_argument("--data-dir", type=Path, default=Path(default_dir))
    ap.add_argument(
        "--read-dir", type=Path, default=None, help="where raw/derived are READ from (scope)"
    )
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.summary:
        print(json.dumps(summarize(a.data_dir), indent=1))
        return 0
    if not a.repo or not (a.numbers_file or a.scope_from_raw):
        ap.error("--repo and (--numbers-file or --scope-from-raw) are required unless --summary")
    if a.numbers_file:
        numbers = [int(x) for x in a.numbers_file.read_text(encoding="utf-8").split()]
    else:
        read = a.read_dir or a.data_dir
        numbers = build_scope(read, read / "derived" / a.repo.replace("/", "__"), a.repo)
        log.info("scope: %d merged PRs", len(numbers))
    client = qcm.PacedClient(tolerant(sg.requests_transport(sg.gh_token())))
    scraper = PrFilesScraper(client, a.repo, a.data_dir, a.batch)
    n = scraper.run(numbers, a.max_batches)
    log.info("done: %d batches, %s, total cost %d", n, dict(scraper.stats), client.total_cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
