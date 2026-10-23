from __future__ import annotations

# Direct per-issue timeline scrape. The main scraper embeds `timelineItems` inside a GraphQL `search`
# query, and a verifier showed those embedded timelines silently DROP events (~14% of label events
# on a control). `repository.issue(number:N)` queried directly is reliable, so this re-fetches the
# same timeline (same itemTypes and node selection, taken from qp0_scrape_github.TIMELINE so the two
# cannot drift) for every issue, under <data-dir>/raw/<owner>__<repo>/timelines/<number // 1000, 5
# digits>.jsonl.gz (_schema 2, _source 'direct') plus an atomic checkpoint.json. Storage / resume
# reuse the dup_targets scraper (-> pr_files); the client is PacedClient. `--compare` is OFFLINE:
# embedded (raw issues) vs direct (raw timelines) completeness report. Sequential by design.
import argparse
import gzip
import json
import logging
import os
import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts, as qp0_derive_*.py
import qp0_scrape_comments as qcm  # noqa: E402
import qp0_scrape_dup_targets as dt  # noqa: E402
import qp0_scrape_github as sg  # noqa: E402
import qp0_scrape_pr_files as pf  # noqa: E402

log = logging.getLogger("qp0tl")
BATCH_SIZES = (10, 5, 2, 1)
MAX_PAGES = 50  # 5,000 events on one issue would be a data problem, not a page size
_TL = sg.TIMELINE.replace(", after:$tl", "")  # first page: no cursor; everything else identical
PAGE_Q = sg.TIMELINE_Q  # node(id:$id) continuation, same selection with after:$tl
TRANSIENT_STATUS = {499: 502}  # HTTP 499 (client closed request) crashed an earlier long run


def transient_499(transport: sg.Transport) -> sg.Transport:
    """Report HTTP 499 as a 502 so the shared client retries it instead of aborting the run."""

    def post(payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        status, headers, body = transport(payload)
        return TRANSIENT_STATUS.get(status, status), headers, body

    return post


def batch_query(numbers: list[int]) -> str:
    """One aliased `issue(number:N)` field per number (alias p<N>, what `tolerant` expects)."""
    fields = " ".join(f"p{n}: issue(number:{n}){{number id {_TL}}}" for n in numbers)
    return (
        "query($owner:String!, $name:String!){ rateLimit{cost remaining resetAt} "
        f"repository(owner:$owner, name:$name){{ {fields} }}}}"
    )


def build_scope(derived_dir: Path) -> list[int]:
    """Every issue number in the derived issues table."""
    iss = pd.read_parquet(derived_dir / "issues.parquet", columns=["number"])
    return sorted(map(int, iss["number"]))


class TimelineScraper(dt.DupTargetScraper):
    """Saves the complete timeline of given issues; resumable and atomic."""

    def __init__(self, client: sg.Client, repo: str, data_dir: Path, batch: int = 10) -> None:
        self.client, self.repo = client, repo
        self.owner, self.name = repo.split("/", 1)
        self.base = data_dir / "raw" / repo.replace("/", "__") / "timelines"
        self.ckpt_path = self.base / "checkpoint.json"
        self.ckpt: dict[str, Any] = {"windows": {}, "done": [], "missing": []}
        if self.ckpt_path.exists():
            self.ckpt = json.loads(self.ckpt_path.read_text(encoding="utf-8"))
        self.done = set(self.ckpt["done"])
        self.sizes = [s for s in BATCH_SIZES if s <= batch] or [1]
        self.max_batch, self._ok_streak = self.sizes[0], 0
        self.stats: dict[str, int] = {"batches": 0, "issues": 0, "events": 0, "missing": 0}

    def _complete(self, issue: dict[str, Any]) -> None:
        """Follow event pagination; saved events must equal totalCount or we fail loudly."""
        tl = issue["timelineItems"]
        tl["nodes"] = [n for n in tl["nodes"] if n]
        for _ in range(MAX_PAGES):
            if not tl["pageInfo"]["hasNextPage"]:
                break
            more = self.client.query(PAGE_Q, {"id": issue["id"], "tl": tl["pageInfo"]["endCursor"]})
            m = more["node"]["timelineItems"]
            tl["nodes"].extend(n for n in m["nodes"] if n)
            tl["pageInfo"] = m["pageInfo"]
        if len(tl["nodes"]) != tl["totalCount"]:
            raise sg.ScrapeError(
                f"completeness: issue #{issue['number']} saved {len(tl['nodes'])} events "
                f"!= totalCount {tl['totalCount']}"
            )

    def fetch(self, numbers: list[int]) -> list[dict[str, Any]]:
        """Rows for ``numbers``; shrinks the batch on persistent transient errors (incl. 499) and
        regrows it (dt._note_success) after a run of clean queries."""
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
                    "issue #%d not found in %s (deleted / transferred): recorded", n, self.repo
                )
                rows.append({"number": n, "missing": True, "_source": "direct", **stamp})
                continue
            self._complete(issue)
            rows.append({**issue, "_source": "direct", **stamp})
        return rows


def summarize(data_dir: Path) -> dict[str, dict[str, int]]:
    """Per repo: checkpoint counts and on-disk rows / events (no network)."""
    out: dict[str, dict[str, int]] = {}
    for ck in sorted((data_dir / "raw").glob("*__*/timelines/checkpoint.json")):
        rows = events = 0
        for f in ck.parent.glob("*.jsonl.gz"):
            for r in TimelineScraper._read(f):
                rows += 1
                events += 0 if r.get("missing") else len(r["timelineItems"]["nodes"])
        c = json.loads(ck.read_text(encoding="utf-8"))
        out[ck.parent.parent.name] = {"ckpt_done": len(c["done"]), "ckpt_missing": len(c["missing"]),
                                      "disk_rows": rows, "disk_events": events}  # fmt: skip
    return out


# ---------------------------------------------------------------- offline compare


def _rows(files: list[Path]) -> Iterator[dict[str, Any]]:
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield json.loads(line)


def load_events(files: list[Path], only: set[int] | None = None) -> dict[int, dict[str, Any]]:
    """{number: {created, events}} from raw issue or timeline files (missing rows skipped)."""
    out: dict[int, dict[str, Any]] = {}
    for r in _rows(files):
        if r.get("missing") or (only is not None and r["number"] not in only):
            continue
        nodes = [n for n in (r.get("timelineItems") or {}).get("nodes", []) if n]
        out[r["number"]] = {"created": r.get("createdAt"), "events": nodes}
    return out


def _labels(events: list[dict[str, Any]], prefix: str, until: str | None = None) -> set[str]:
    """Labels (starting with ``prefix``) in place after replaying events up to ``until``."""
    cur: set[str] = set()
    for e in sorted(events, key=lambda e: e.get("createdAt") or ""):
        name = (e.get("label") or {}).get("name", "")
        if e["__typename"] not in ("LabeledEvent", "UnlabeledEvent") or not name.startswith(prefix):
            continue
        if until is not None and (e.get("createdAt") or "") > until:
            break
        cur.add(name) if e["__typename"] == "LabeledEvent" else cur.discard(name)
    return cur


def _first_label(events: list[dict[str, Any]], prefix: str) -> str | None:
    ls = [
        e
        for e in events
        if e["__typename"] == "LabeledEvent" and e["label"]["name"].startswith(prefix)
    ]
    ls.sort(key=lambda e: (e.get("createdAt") or "", e["label"]["name"]))
    return ls[0]["label"]["name"] if ls else None


def _pr_closes(events: list[dict[str, Any]]) -> int:
    return sum(
        e["__typename"] == "ClosedEvent" and "number" in (e.get("closer") or {}) for e in events
    )


def _t72(created: str | None) -> str | None:
    if not created:
        return None
    return (sg._parse_ts(created) + pd.Timedelta(hours=72)).strftime("%Y-%m-%dT%H:%M:%SZ")


def compare(
    emb: dict[int, dict[str, Any]], dire: dict[int, dict[str, Any]], prefix: str
) -> dict[str, Any]:
    """Embedded vs direct completeness; ``prefix`` selects the labels for first-label / t72h checks."""
    nums = sorted(set(emb) & set(dire))
    types: dict[str, Counter[str]] = {"embedded": Counter(), "direct": Counter()}
    keys = ("closed_by_pr", "marked_duplicate", "cross_referenced")
    foc = {k: {"embedded": 0, "direct": 0, "issues_differing": 0} for k in keys}
    miss: Counter[int] = Counter()
    more = fl_changed = t72_changed = fl_any = t72_any = 0
    by_year: dict[str, Counter[str]] = {}
    for n in nums:
        e, d = emb[n]["events"], dire[n]["events"]
        ce, cd = Counter(x["__typename"] for x in e), Counter(x["__typename"] for x in d)
        types["embedded"].update(ce)
        types["direct"].update(cd)
        yr = by_year.setdefault((emb[n]["created"] or "unknown")[:4], Counter())
        yr["n"] += 1
        yr["direct_more"] += len(d) > len(e)
        gap = max(0, cd["LabeledEvent"] - ce["LabeledEvent"])
        miss[gap] += 1
        yr["labeled_missing"] += gap > 0
        more += len(d) > len(e)
        pairs = {"closed_by_pr": (_pr_closes(e), _pr_closes(d)),
                 "marked_duplicate": (ce["MarkedAsDuplicateEvent"], cd["MarkedAsDuplicateEvent"]),
                 "cross_referenced": (ce["CrossReferencedEvent"], cd["CrossReferencedEvent"])}  # fmt: skip
        for k, (a, b) in pairs.items():
            foc[k]["embedded"] += a
            foc[k]["direct"] += b
            foc[k]["issues_differing"] += a != b
        f1, f2 = _first_label(e, prefix), _first_label(d, prefix)
        fl_changed += f1 != f2
        fl_any += bool(f1 or f2)
        u = _t72(emb[n]["created"])
        s1, s2 = _labels(e, prefix, u), _labels(d, prefix, u)
        t72_changed += s1 != s2
        t72_any += bool(s1 or s2)

    def share(a: int, b: int) -> float | None:
        return round(a / b, 6) if b else None

    return {
        "n_issues_compared": len(nums),
        "label_prefix": prefix or "(all labels)",
        "event_counts_by_type": {k: dict(sorted(v.items())) for k, v in types.items()},
        "share_issues_direct_has_more_events": share(more, len(nums)),
        "labeled_event": {
            "share_issues_missing_ge1": share(len(nums) - miss[0], len(nums)),
            "missing_count_distribution": {str(k): v for k, v in sorted(miss.items())},
        },
        "by_created_year": {
            y: {
                "n": c["n"],
                "share_direct_more": share(c["direct_more"], c["n"]),
                "share_labeled_missing_ge1": share(c["labeled_missing"], c["n"]),
            }
            for y, c in sorted(by_year.items())
        },  # fmt: skip
        "first_label_event_changed": {
            "share_of_all": share(fl_changed, len(nums)),
            "n": fl_changed,
            "share_of_issues_with_any": share(fl_changed, fl_any),
        },  # fmt: skip
        "labels_at_created_plus_72h_changed": {
            "share_of_all": share(t72_changed, len(nums)),
            "n": t72_changed,
            "share_of_issues_with_any": share(t72_changed, t72_any),
        },  # fmt: skip
        "focus_events": foc,
    }


def run_compare(a: argparse.Namespace, numbers: list[int] | None) -> int:
    slug = a.repo.replace("/", "__")
    only = set(numbers) if numbers else None
    read = a.read_dir or a.data_dir
    emb = load_events(sorted((read / "raw" / slug / "issues").glob("*/*.jsonl.gz")), only)
    dire = load_events(sorted((a.data_dir / "raw" / slug / "timelines").glob("*.jsonl.gz")), only)
    prefix = (
        a.label_prefix if a.label_prefix is not None else ("sig/" if "kubernetes" in slug else "")
    )
    rep = compare(emb, dire, prefix)
    out = a.out_dir / f"qp0_timeline_completeness_{slug}_2026-10.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repo": a.repo, **rep}, indent=1), encoding="utf-8")
    log.info("compare: %d issues -> %s", rep["n_issues_compared"], out)
    if a.print_issues:
        for n in sorted(set(emb) & set(dire)):
            print(f"#{n}: embedded={len(emb[n]['events'])} direct={len(dire[n]['events'])}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Scrape complete (direct) timelines for issues")
    ap.add_argument("--repo")
    ap.add_argument("--numbers-file", type=Path, help="one issue number per line")
    ap.add_argument("--scope-from-derived", action="store_true", help="all derived issue numbers")
    default_dir = os.environ.get("TRIAGEIQ_DATA_DIR", r"D:\triageiq-data")
    ap.add_argument("--data-dir", type=Path, default=Path(default_dir))
    ap.add_argument(
        "--read-dir", type=Path, default=None, help="where raw issues / derived are READ"
    )
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--compare", action="store_true", help="OFFLINE embedded-vs-direct report")
    ap.add_argument("--label-prefix", default=None, help="compare: default sig/ for kubernetes")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "reports")
    ap.add_argument("--print-issues", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if a.summary:
        print(json.dumps(summarize(a.data_dir), indent=1))
        return 0
    if not a.repo or not (a.numbers_file or a.scope_from_derived or a.compare):
        ap.error("--repo and (--numbers-file or --scope-from-derived or --compare) are required")
    numbers: list[int] | None = None
    if a.numbers_file:
        numbers = [int(x) for x in a.numbers_file.read_text(encoding="utf-8").split()]
    elif a.scope_from_derived:
        numbers = build_scope((a.read_dir or a.data_dir) / "derived" / a.repo.replace("/", "__"))
        log.info("scope: %d issues", len(numbers))
    if a.compare:
        return run_compare(a, numbers)
    client = qcm.PacedClient(pf.tolerant(transient_499(sg.requests_transport(sg.gh_token()))))
    scraper = TimelineScraper(client, a.repo, a.data_dir, a.batch)
    n = scraper.run(numbers or [], a.max_batches)
    log.info("done: %d batches, %s, total cost %d", n, scraper.stats, client.total_cost)
    return 0


if __name__ == "__main__":
    sys.exit(main())
