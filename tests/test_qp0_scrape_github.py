from __future__ import annotations

import gzip
import importlib.util
import json
import logging
import re
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "qp0", Path(__file__).resolve().parents[1] / "scripts" / "qp0_scrape_github.py"
)
qp0 = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0"] = qp0
_SPEC.loader.exec_module(qp0)

D0 = datetime(2023, 1, 1, tzinfo=UTC)
RL = {"cost": 1, "remaining": 4000, "resetAt": "2099-01-01T00:00:00Z"}


class FakeGitHub:
    """In-memory GitHub: serves count / search-page / timeline-page queries over `items`."""

    def __init__(self, items: list[dict[str, Any]], rl: dict[str, Any] | None = None) -> None:
        self.items = items
        self.rl = rl or RL
        self.calls: list[dict[str, Any]] = []
        self.scripted: list[tuple[int, dict[str, str], dict[str, Any]]] = []

    def _match(self, q: str) -> list[dict[str, Any]]:
        lo, hi = re.search(r"created:(\S+)\.\.(\S+)", q).groups()
        return [i for i in self.items if lo <= i["createdAt"] <= hi]

    def __call__(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        self.calls.append(payload)
        if self.scripted:
            return self.scripted.pop(0)
        v, q = payload["variables"], payload["query"]
        if "node(id" in q:
            item = next(i for i in self.items if i["id"] == v["id"])
            tl = self._tl_page(item, v.get("tl"))
            return 200, {}, {"data": {"rateLimit": self.rl, "node": {"timelineItems": tl}}}
        hits = self._match(v["q"])
        if "n" not in v:
            return 200, {}, {"data": {"rateLimit": self.rl, "search": {"issueCount": len(hits)}}}
        start = int(v["after"] or 0)
        page = hits[start : start + v["n"]]
        more = start + v["n"] < len(hits)
        nodes = []
        for i in page:
            n = {k: x for k, x in i.items() if k != "_events"}
            n["timelineItems"] = self._tl_page(i, None)
            nodes.append(n)
        s = {
            "issueCount": len(hits),
            "pageInfo": {"hasNextPage": more, "endCursor": str(start + v["n"])},
            "nodes": nodes,
        }
        return 200, {}, {"data": {"rateLimit": self.rl, "search": s}}

    @staticmethod
    def _tl_page(item: dict[str, Any], cursor: str | None) -> dict[str, Any]:
        ev, s = item["_events"], int(cursor or 0)
        return {
            "totalCount": len(ev),
            "nodes": ev[s : s + 100],
            "pageInfo": {"hasNextPage": s + 100 < len(ev), "endCursor": str(s + 100)},
        }


def make_items(n_per_day: dict[int, int], events: int = 1) -> list[dict[str, Any]]:
    items, num = [], 0
    for day, n in n_per_day.items():
        for k in range(n):
            num += 1
            t = D0 + timedelta(days=day, hours=k % 24, minutes=k // 24)
            items.append(
                {
                    "id": f"I{num}",
                    "number": num,
                    "createdAt": qp0._fmt(t),
                    "_events": [{"__typename": "LabeledEvent", "n": j} for j in range(events)],
                }
            )
    return items


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    slept: list[float] = []
    monkeypatch.setattr(qp0.time, "sleep", slept.append)
    return slept


def read_rows(base: Path) -> list[dict[str, Any]]:
    rows = []
    for f in sorted(base.glob("*/*.jsonl.gz")):
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh]
    return rows


def scraper(fake: Any, tmp_path: Path, kind: str = "issues") -> Any:
    return qp0.Scraper(qp0.Client(fake), "o/r", kind, tmp_path)


UNTIL = D0 + timedelta(days=3, seconds=-1)


def test_resume_skips_completed_windows(tmp_path: Path) -> None:
    fake = FakeGitHub(make_items({0: 2, 1: 2, 2: 2}))
    assert scraper(fake, tmp_path).run(D0, UNTIL, max_windows=2) == 2  # "interrupted"
    fake.calls.clear()
    assert scraper(fake, tmp_path).run(D0, UNTIL) == 1
    searched = {
        re.search(r"created:(\S+)\.\.", c["variables"]["q"]).group(1)[:10] for c in fake.calls
    }
    assert searched == {"2023-01-03"}
    fake.calls.clear()
    assert scraper(fake, tmp_path).run(D0, UNTIL) == 0 and fake.calls == []
    rows = read_rows(tmp_path / "raw" / "o__r" / "issues")
    assert sorted(r["number"] for r in rows) == [1, 2, 3, 4, 5, 6]  # no duplicates


def test_cap_split_union_equals_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(qp0, "SEARCH_CAP", 5)
    items = make_items({0: 23})
    fake = FakeGitHub(items)
    s = scraper(fake, tmp_path)
    s.run(D0, D0 + timedelta(days=1, seconds=-1))
    rows = read_rows(tmp_path / "raw" / "o__r" / "issues")
    assert sorted(r["id"] for r in rows) == sorted(i["id"] for i in items)
    assert s.ckpt["windows"]["2023-01-01"] == {
        "issueCount": 23,
        "rows": 23,
        "completed_at": s.ckpt["windows"]["2023-01-01"]["completed_at"],
    }
    assert sum("n" in c["variables"] for c in fake.calls) > 1  # window was split


def test_rate_limit_sleeps_until_reset(tmp_path: Path, no_sleep: list[float]) -> None:
    reset = qp0._fmt(qp0._now() + timedelta(seconds=120))
    fake = FakeGitHub(make_items({0: 1}), rl={"cost": 1, "remaining": 10, "resetAt": reset})
    scraper(fake, tmp_path).run(D0, D0 + timedelta(days=1, seconds=-1))
    assert no_sleep and all(110 <= w <= 126 for w in no_sleep)  # ~reset + 5s margin


def test_secondary_rate_limit_backoff_then_gives_up(tmp_path: Path, no_sleep: list[float]) -> None:
    fake = FakeGitHub([])
    fake.scripted = [(403, {}, {"message": "You have exceeded a secondary rate limit"})] * 10
    with pytest.raises(qp0.ScrapeError, match="secondary rate limit"):
        scraper(fake, tmp_path).run(D0, D0 + timedelta(days=1, seconds=-1))
    assert no_sleep == [60, 120, 240, 480, 960]  # 6 tries -> 5 sleeps, exponential


def test_retry_after_header_honored(tmp_path: Path, no_sleep: list[float]) -> None:
    fake = FakeGitHub(make_items({0: 1}))
    fake.scripted = [(429, {"Retry-After": "7"}, {"message": "slow down"})]
    scraper(fake, tmp_path).run(D0, D0 + timedelta(days=1, seconds=-1))
    assert no_sleep == [7.0]


def test_5xx_retried_then_clear_error(tmp_path: Path, no_sleep: list[float]) -> None:
    fake = FakeGitHub([])
    fake.scripted = [(502, {}, {"message": "Bad Gateway"})] * 40
    with pytest.raises(qp0.TransientError, match="HTTP 502 after 6 tries"):
        scraper(fake, tmp_path).run(D0, D0 + timedelta(days=1, seconds=-1))
    assert len(no_sleep) == 5


def test_page_size_fallback(tmp_path: Path) -> None:
    fake = FakeGitHub(make_items({0: 3}))
    real = fake.__call__
    sizes: list[int] = []

    def flaky(payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        n = payload["variables"].get("n")
        if n:
            sizes.append(n)
            if n > 25:
                raise TimeoutError("timed out")
        return real(payload)

    s = qp0.Scraper(qp0.Client(flaky), "o/r", "issues", tmp_path)
    s.run(D0, D0 + timedelta(days=1, seconds=-1))
    assert [x for x in dict.fromkeys(sizes)] == [100, 50, 25]
    assert len(read_rows(tmp_path / "raw" / "o__r" / "issues")) == 3


def test_timeline_pagination_follows_cursor(tmp_path: Path) -> None:
    fake = FakeGitHub(make_items({0: 1}, events=250))
    scraper(fake, tmp_path).run(D0, D0 + timedelta(days=1, seconds=-1))
    (row,) = read_rows(tmp_path / "raw" / "o__r" / "issues")
    assert [e["n"] for e in row["timelineItems"]["nodes"]] == list(range(250))
    assert row["timelineItems"]["pageInfo"]["hasNextPage"] is False
    assert sum("node(id" in c["query"] for c in fake.calls) == 2


def test_completeness_assertion_fails_loudly(tmp_path: Path) -> None:
    fake = FakeGitHub(make_items({0: 4}))
    real = fake.__call__

    def lossy(payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        st, h, b = real(payload)
        if "n" in payload["variables"]:
            b["data"]["search"]["nodes"].pop()  # a row silently goes missing
        return st, h, b

    s = qp0.Scraper(qp0.Client(lossy), "o/r", "issues", tmp_path)
    with pytest.raises(qp0.ScrapeError, match="completeness: saved 3 rows != issueCount 4"):
        s.run(D0, D0 + timedelta(days=1, seconds=-1))
    assert s.ckpt["windows"] == {} and not (s.base / "checkpoint.json").exists()


def test_checkpoint_write_is_atomic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeGitHub(make_items({0: 1, 1: 1}))
    s = scraper(fake, tmp_path)
    s.run(D0, D0 + timedelta(days=1, seconds=-1))
    before = (s.base / "checkpoint.json").read_text(encoding="utf-8")

    def boom(src: Any, dst: Any) -> None:
        raise OSError("crash before rename")

    monkeypatch.setattr(qp0.os, "replace", boom)
    with pytest.raises(OSError, match="crash"):
        s.run(D0, D0 + timedelta(days=2, seconds=-1))
    # the live checkpoint is untouched / still valid JSON; only a temp file may be left behind
    assert (s.base / "checkpoint.json").read_text(encoding="utf-8") == before
    assert json.loads(before)["windows"].keys() == {"2023-01-01"}


def test_pr_query_has_no_unused_variable(tmp_path: Path) -> None:
    s = scraper(FakeGitHub([]), tmp_path, kind="prs")
    assert "$tl" not in s.search_q and "closingIssuesReferences" in s.search_q
    assert "is:pr" in s._search_q(D0, UNTIL)


def test_token_never_leaks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    secret = "gho_" + "A" * 36
    monkeypatch.setattr(
        qp0.subprocess,
        "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout=secret + "\n", stderr=""),
    )
    token = qp0.gh_token()
    assert token == secret

    def failing(payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        return 403, {}, {"message": "secondary rate limit"}

    caplog.set_level(logging.DEBUG)
    with pytest.raises(qp0.ScrapeError) as ei:
        qp0.Client(failing).query("query{x}", {})
    assert secret not in str(ei.value) and secret not in caplog.text

    def bad_run(*a: Any, **k: Any) -> Any:
        raise subprocess.CalledProcessError(1, a[0], output=secret, stderr=secret)

    monkeypatch.setattr(qp0.subprocess, "run", bad_run)
    with pytest.raises(qp0.ScrapeError) as ei2:
        qp0.gh_token()
    assert secret not in str(ei2.value) and secret not in repr(ei2.value.__cause__)


def test_summary_counts_disk_and_checkpoint(tmp_path: Path) -> None:
    scraper(FakeGitHub(make_items({0: 2, 1: 1})), tmp_path).run(
        D0, D0 + timedelta(days=2, seconds=-1)
    )
    assert qp0.summarize(tmp_path) == {"o__r/issues": {"ckpt:2023": 3, "disk:2023": 3}}
