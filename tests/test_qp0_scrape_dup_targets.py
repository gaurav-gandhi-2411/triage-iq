from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp0dt", _S / "qp0_scrape_dup_targets.py")
dt = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0dt"] = dt
_SPEC.loader.exec_module(dt)

TOKEN = "ghp_SECRET_TOKEN_VALUE"  # noqa: S105 - fake, to prove it never leaks
RL = {"cost": 1, "remaining": 5000, "resetAt": "2030-01-01T00:00:00Z"}


class Fake:
    """Fake GraphQL transport over {issue number: event count | None (missing)}."""

    def __init__(self, totals: dict[int, int | None], fail_over: int | None = None) -> None:
        self.totals, self.queries, self.fail_over = totals, [], fail_over

    def events(self, n: int, start: int, end: int) -> list[dict[str, Any]]:
        return [
            {"__typename": "MarkedAsDuplicateEvent", "createdAt": "t", "canonical": {"number": n}, "duplicate": {"number": 1000 + i}}
            for i in range(start, end)
        ]  # fmt: skip

    def page(self, n: int, start: int, first: bool = False) -> dict[str, Any]:
        total = self.totals[n]
        end = min(start + 100, total)
        out = {
            "pageInfo": {"hasNextPage": end < total, "endCursor": f"after{end}"},
            "nodes": self.events(n, start, end),
        }
        return {**out, "totalCount": total} if first else out

    def __call__(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        q, v = payload["query"], payload["variables"]
        self.queries.append(q)
        nums = [int(x) for x in re.findall(r"issue\(number:(\d+)\)", q)]
        if self.fail_over is not None and len(nums) > self.fail_over:
            return 502, {}, {"message": "Bad Gateway"}
        if "node(id" in q:
            n, k = int(v["id"][2:]), int(v["c"][5:])
            return 200, {}, {"data": {"rateLimit": RL, "node": {"timelineItems": self.page(n, k)}}}
        repo, errors = {}, []
        for n in nums:
            if self.totals[n] is None:
                repo[f"p{n}"] = None
                errors.append(
                    {"type": "NOT_FOUND", "path": ["repository", f"p{n}"], "message": "x"}
                )
                continue
            repo[f"p{n}"] = {"number": n, "id": f"IS{n}", "state": "CLOSED", "stateReason": "DUPLICATE", "timelineItems": self.page(n, 0, True)}  # fmt: skip
        body: dict[str, Any] = {"data": {"rateLimit": RL, "repository": repo}}
        if errors:
            body["errors"] = errors
        return 200, {}, body


def scraper(tmp_path: Path, fake: Fake, batch: int = 10):  # noqa: ANN201
    return dt.DupTargetScraper(dt.qcm.PacedClient(dt.pf.tolerant(fake)), "o/r", tmp_path, batch)


def rows_on_disk(s) -> dict[int, dict]:  # noqa: ANN001
    return {r["number"]: r for p in sorted(s.base.glob("*.jsonl.gz")) for r in s._read(p)}


def test_query_shape_has_all_three_event_kinds_and_both_sides() -> None:
    q = dt.batch_query([5, 1007])
    assert "p5: issue(number:5)" in q and "p1007: issue(number:1007)" in q
    for field in ("CLOSED_EVENT", "MARKED_AS_DUPLICATE_EVENT", "UNMARKED_AS_DUPLICATE_EVENT", "duplicateOf", "canonical{", "duplicate{", "isCrossRepository", "stateReason", "totalCount"):  # fmt: skip
        assert field in q
    assert q.count("rateLimit{cost remaining resetAt}") == 1 and q.count("{") == q.count("}")
    assert dt.PAGE_Q.count("{") == dt.PAGE_Q.count("}") and "after:$c" in dt.PAGE_Q


def test_pagination_over_100_events_and_layout(tmp_path: Path) -> None:
    s = scraper(tmp_path, Fake({5: 250, 1007: 2}))
    s.run([1007, 5])
    rows = rows_on_disk(s)
    assert len(rows[5]["timelineItems"]["nodes"]) == 250 and rows[5]["_schema"] == 2
    assert sorted(p.name for p in s.base.glob("*.jsonl.gz")) == ["00000.jsonl.gz", "00001.jsonl.gz"]
    assert s.base.name == "dup_targets" and json.loads(s.ckpt_path.read_text())["done"] == [5, 1007]


def test_resume_skips_done_numbers(tmp_path: Path) -> None:
    scraper(tmp_path, Fake({1: 1, 2: 1, 3: 1})).run([1, 2])
    fake2 = Fake({1: 1, 2: 1, 3: 1})
    scraper(tmp_path, fake2).run([1, 2, 3])
    assert len(fake2.queries) == 1 and "issue(number:3)" in fake2.queries[0]
    assert "issue(number:1)" not in fake2.queries[0]


def test_missing_issue_recorded_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    fake = Fake({1: 2, 9: None})
    s = scraper(tmp_path, fake)
    s.run([1, 9])
    rows = rows_on_disk(s)
    assert rows[9]["missing"] is True and len(rows[1]["timelineItems"]["nodes"]) == 2
    assert (
        json.loads(s.ckpt_path.read_text())["missing"] == [9]
        and len(fake.queries) == 1
        and not sleeps
    )


def test_completeness_failure_names_the_issue(tmp_path: Path) -> None:
    class Short(Fake):
        def page(self, n: int, start: int, first: bool = False) -> dict[str, Any]:
            out = super().page(n, start, first)
            if first:
                out["totalCount"] += 1
            return out

    with pytest.raises(dt.sg.ScrapeError, match=r"issue #7 .*!= totalCount 4"):
        scraper(tmp_path, Short({7: 3})).run([7])
    assert not list(tmp_path.rglob("*.jsonl.gz"))


def test_batch_fallback_atomic_checkpoint_and_no_foreign_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    s = scraper(tmp_path / "data", Fake({n: 1 for n in range(1, 11)}, fail_over=2))
    s.run(list(range(1, 11)))
    assert s.sizes[0] == 2 and sorted(s.done) == list(range(1, 11))
    assert (
        not list(tmp_path.rglob("*.tmp")) and json.loads(s.ckpt_path.read_text())["windows"] == {}
    )
    assert {p.relative_to(tmp_path).parts[:4] for p in tmp_path.rglob("*") if p.is_file()} == {("data", "raw", "o__r", "dup_targets")}  # fmt: skip


def test_token_not_in_errors_or_stored_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(payload: dict[str, Any]) -> Any:
        raise ConnectionError("ConnectError")

    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(dt.sg.ScrapeError) as ei:
        dt.DupTargetScraper(dt.qcm.PacedClient(dt.pf.tolerant(boom)), "o/r", tmp_path, 1).run([1])
    assert TOKEN not in str(ei.value)
    scraper(tmp_path, Fake({2: 1})).run([2])
    blob = b"".join(gzip.decompress(p.read_bytes()) for p in tmp_path.rglob("*.jsonl.gz"))
    assert TOKEN.encode() not in blob


def test_scope_has_any_duplicate_signal_and_summary(tmp_path: Path) -> None:
    derived = tmp_path / "derived"
    derived.mkdir()
    pd.DataFrame(
        {
            "number": [1, 2, 3, 4, 5, 6],
            "labels_final": [["*duplicate"], ["bug"], ["Possible-Duplicate"], [], ["x"], []],
            "closed_as_duplicate": [False, False, False, True, False, False],
            "canon_self": [0, 0, 0, 0, 1, 0],
            "canon_foreign_repo": [0, 0, 0, 0, 0, 0],
        }
    ).to_parquet(derived / "issues.parquet")
    assert dt.build_scope(derived) == [1, 3, 4, 5]  # label, label (case-insens.), reason, event
    data = tmp_path / "data"
    scraper(data, Fake({1: 2})).run([1])
    assert dt.summarize(data)["o__r"] == {"ckpt_done": 1, "ckpt_missing": 0, "rows": 1, "closed_duplicate_of": 0, "marked_with_duplicate": 1}  # fmt: skip


def test_batch_regrows_after_clean_run_capped_at_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(dt, "REGROW_AFTER", 10**6)  # no regrow during the burst
    fake = Fake({n: 1 for n in range(1, 200)}, fail_over=1)
    s = scraper(tmp_path, fake, batch=5)
    s.run(list(range(1, 6)))
    assert s.sizes[0] == 1
    s._ok_streak = 0  # successes during the burst must not pre-load the streak
    fake.fail_over, seen = None, len(fake.queries)
    monkeypatch.setattr(dt, "REGROW_AFTER", 3)
    s.run(list(range(10, 100)))
    sizes = [len(re.findall(r"issue\(number:", q)) for q in fake.queries[seen:]]
    assert s.sizes[0] == 5 and max(sizes) == 5 and sizes[:-1] == sorted(sizes[:-1])
    monkeypatch.setattr(dt, "REGROW_AFTER", 10**6)  # isolate the shrink from regrow
    fake.fail_over = 2  # a failure after regrow shrinks again
    s.run(list(range(100, 120)))
    assert s.sizes[0] == 2
