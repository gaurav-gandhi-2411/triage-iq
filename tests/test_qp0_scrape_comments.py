from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("qp0sc", _SCRIPTS / "qp0_scrape_comments.py")
qp0sc = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0sc"] = qp0sc
_SPEC.loader.exec_module(qp0sc)

TOKEN = "ghp_SECRET_TOKEN_VALUE"  # noqa: S105 - fake, to prove it never leaks
RL = {"cost": 1, "remaining": 5000, "resetAt": "2030-01-01T00:00:00Z"}


class Fake:
    """Fake GraphQL transport over {issue number: total comment count}; records every query."""

    def __init__(self, totals: dict[int, int], fail_over: int | None = None) -> None:
        self.totals, self.queries, self.fail_over = totals, [], fail_over

    def comment(self, n: int, i: int) -> dict[str, Any]:
        return {"id": f"c{n}-{i}", "body": "b", "author": {"login": "u", "__typename": "User"}}

    def __call__(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        q, v = payload["query"], payload["variables"]
        self.queries.append(q)
        nums = [int(x) for x in re.findall(r"issue\(number:(\d+)\)", q)]
        if self.fail_over is not None and len(nums) > self.fail_over:
            return 502, {}, {"message": "Bad Gateway"}
        if "node(id" in q:  # follow-up page: ids look like ID<n>; after cursor = "after<k>"
            n, k = int(v["id"][2:]), int(v["c"][5:])
            return 200, {}, {"data": {"rateLimit": RL, "node": {"comments": self.page(n, k)}}}
        repo = {}
        for n in nums:
            issue = {"number": n, "id": f"ID{n}", "comments": self.page(n, 0, True)}
            issue["timelineItems"] = {"nodes": [{"__typename": "ClosedEvent", "createdAt": "t"}]}
            repo[f"i{n}"] = issue
        return 200, {}, {"data": {"rateLimit": RL, "repository": repo}}

    def page(self, n: int, start: int, first: bool = False) -> dict[str, Any]:
        total = self.totals[n]
        end = min(start + 100, total)
        info = {"hasNextPage": end < total, "endCursor": f"after{end}"}
        out = {"pageInfo": info, "nodes": [self.comment(n, i) for i in range(start, end)]}
        return {**out, "totalCount": total} if first else out


def scraper(tmp_path: Path, fake: Fake, batch: int = 10) -> Any:
    return qp0sc.CommentScraper(qp0sc.sg.Client(fake), "o/r", tmp_path, batch)


def test_batch_query_has_one_alias_per_number() -> None:
    q = qp0sc.batch_query([5, 1007])
    assert "i5: issue(number:5)" in q and "i1007: issue(number:1007)" in q
    assert q.count("rateLimit{cost remaining resetAt}") == 1 and "CLOSED_EVENT" in q


def test_pagination_beyond_100_comments_and_storage_layout(tmp_path: Path) -> None:
    fake = Fake({5: 250, 1007: 2})
    s = scraper(tmp_path, fake)
    s.run([1007, 5])
    rows = {r["number"]: r for p in sorted(s.base.glob("*.jsonl.gz")) for r in s._read(p)}
    assert len(rows[5]["comments"]["nodes"]) == 250 and rows[5]["_schema"] == 2
    assert sorted(p.name for p in s.base.glob("*.jsonl.gz")) == ["00000.jsonl.gz", "00001.jsonl.gz"]
    assert rows[5]["events"][0]["__typename"] == "ClosedEvent"
    assert json.loads(s.ckpt_path.read_text())["done"] == [5, 1007]


def test_resume_skips_done_numbers(tmp_path: Path) -> None:
    fake = Fake({1: 1, 2: 1, 3: 1})
    scraper(tmp_path, fake).run([1, 2])
    fake2 = Fake({1: 1, 2: 1, 3: 1})
    scraper(tmp_path, fake2).run([1, 2, 3])
    assert len(fake2.queries) == 1 and "issue(number:3)" in fake2.queries[0]
    assert "issue(number:1)" not in fake2.queries[0]


def test_completeness_failure_names_the_issue(tmp_path: Path) -> None:
    class Short(Fake):
        def page(self, n: int, start: int, first: bool = False) -> dict[str, Any]:
            out = super().page(n, start, first)
            if first:
                out["totalCount"] += 1  # claims one more comment than is delivered
            return out

    with pytest.raises(qp0sc.sg.ScrapeError, match=r"issue #7 .*!= totalCount 4"):
        scraper(tmp_path, Short({7: 3})).run([7])
    assert not list(tmp_path.rglob("*.jsonl.gz"))  # nothing saved for an incomplete issue


def test_missing_issue_fails_loudly(tmp_path: Path) -> None:
    class Gone(Fake):
        def __call__(self, payload: dict[str, Any]) -> Any:
            return 200, {}, {"data": {"rateLimit": RL, "repository": {"i9": None}}}

    with pytest.raises(qp0sc.sg.ScrapeError, match="issue #9 not found"):
        scraper(tmp_path, Gone({9: 1})).run([9])


def test_batch_size_falls_back_on_502(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    fake = Fake({n: 1 for n in range(1, 11)}, fail_over=2)
    s = scraper(tmp_path, fake)
    s.run(list(range(1, 11)))
    assert s.sizes[0] == 2
    assert sorted(s.done) == list(range(1, 11))
    assert max(len(re.findall(r"issue\(number", q)) for q in fake.queries[-3:]) <= 2


def test_checkpoint_is_atomic_and_compatible(tmp_path: Path) -> None:
    s = scraper(tmp_path, Fake({1: 1}))
    s.run([1])
    assert not list(tmp_path.rglob("*.tmp"))
    assert json.loads(s.ckpt_path.read_text())["windows"] == {}  # main --summary compatibility
    with gzip.open(s.base / "00000.jsonl.gz", "rt", encoding="utf-8") as f:
        assert json.loads(f.readline())["number"] == 1


def test_no_write_outside_data_dir_and_summary(tmp_path: Path) -> None:
    data = tmp_path / "data"
    scraper(data, Fake({1: 2})).run([1])
    touched = {p.relative_to(tmp_path).parts[:4] for p in tmp_path.rglob("*") if p.is_file()}
    assert all(t[:3] == ("data", "raw", "o__r") and t[3] in ("comments",) for t in touched)
    assert qp0sc.summarize(data)["o__r"] == {"ckpt_done": 1, "disk_issues": 1, "comments": 2}


def test_token_not_in_errors_or_stored_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(payload: dict[str, Any]) -> Any:
        raise ConnectionError("ConnectError")

    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(qp0sc.sg.ScrapeError) as ei:
        scraper(tmp_path, boom, 1).run([1])  # type: ignore[arg-type]
    assert TOKEN not in str(ei.value)
    scraper(tmp_path, Fake({2: 1})).run([2])
    blob = b"".join(gzip.decompress(p.read_bytes()) for p in tmp_path.rglob("*.jsonl.gz"))
    assert TOKEN.encode() not in blob


def test_batch_regrows_after_clean_run_capped_at_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(qp0sc, "REGROW_AFTER", 10**6)  # no regrow during the burst
    fake = Fake({n: 1 for n in range(1, 200)}, fail_over=1)
    s = scraper(tmp_path, fake, batch=5)
    s.run(list(range(1, 6)))
    assert s.sizes[0] == 1
    s._ok_streak = 0  # successes during the burst must not pre-load the streak
    fake.fail_over, seen = None, len(fake.queries)
    monkeypatch.setattr(qp0sc, "REGROW_AFTER", 3)
    s.run(list(range(10, 100)))
    sizes = [len(re.findall(r"issue\(number", q)) for q in fake.queries[seen:]]
    assert s.sizes[0] == 5 and max(sizes) == 5 and sizes[:-1] == sorted(sizes[:-1])
    monkeypatch.setattr(qp0sc, "REGROW_AFTER", 10**6)  # isolate the shrink from regrow
    fake.fail_over = 2  # a failure after regrow shrinks again
    s.run(list(range(100, 120)))
    assert s.sizes[0] == 2
