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
_SPEC = importlib.util.spec_from_file_location("qp0pf", _S / "qp0_scrape_pr_files.py")
pf = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0pf"] = pf
_SPEC.loader.exec_module(pf)

TOKEN = "ghp_SECRET_TOKEN_VALUE"  # noqa: S105 - fake, to prove it never leaks
RL = {"cost": 1, "remaining": 5000, "resetAt": "2030-01-01T00:00:00Z"}


class Fake:
    """Fake GraphQL transport over {PR number: file count | None (missing)}."""

    def __init__(self, totals: dict[int, int | None], fail_over: int | None = None) -> None:
        self.totals, self.queries, self.fail_over = totals, [], fail_over

    def page(self, n: int, start: int, first: bool = False) -> dict[str, Any]:
        total = self.totals[n]
        end = min(start + 100, total)
        out = {
            "pageInfo": {"hasNextPage": end < total, "endCursor": f"after{end}"},
            "nodes": [{"path": f"f{n}/{i}", "additions": 1, "deletions": 0, "changeType": "MODIFIED"} for i in range(start, end)],
        }  # fmt: skip
        return {**out, "totalCount": total} if first else out

    def __call__(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        q, v = payload["query"], payload["variables"]
        self.queries.append(q)
        nums = [int(x) for x in re.findall(r"pullRequest\(number:(\d+)\)", q)]
        if self.fail_over is not None and len(nums) > self.fail_over:
            return 502, {}, {"message": "Bad Gateway"}
        if "node(id" in q:
            n, k = int(v["id"][2:]), int(v["c"][5:])
            return 200, {}, {"data": {"rateLimit": RL, "node": {"files": self.page(n, k)}}}
        repo, errors = {}, []
        for n in nums:
            if self.totals[n] is None:
                repo[f"p{n}"] = None
                errors.append(
                    {
                        "type": "NOT_FOUND",
                        "path": ["repository", f"p{n}"],
                        "message": "Could not resolve",
                    }
                )
                continue
            repo[f"p{n}"] = {
                "number": n, "id": f"PR{n}", "state": "MERGED", "merged": True, "baseRefName": "release-1.30" if n % 2 else "master",
                "files": self.page(n, 0, True),
            }  # fmt: skip
        body: dict[str, Any] = {"data": {"rateLimit": RL, "repository": repo}}
        if errors:
            body["errors"] = errors
        return 200, {}, body


def scraper(tmp_path: Path, fake: Fake, batch: int = 10):  # noqa: ANN201
    return pf.PrFilesScraper(pf.qcm.PacedClient(pf.tolerant(fake)), "o/r", tmp_path, batch)


def rows_on_disk(s) -> dict[int, dict]:  # noqa: ANN001
    return {r["number"]: r for p in sorted(s.base.glob("*.jsonl.gz")) for r in s._read(p)}


def test_batch_query_shape() -> None:
    q = pf.batch_query([5, 1007])
    assert "p5: pullRequest(number:5)" in q and "p1007: pullRequest(number:1007)" in q
    for field in (
        "baseRefName",
        "headRefName",
        "mergeCommit{oid}",
        "files(first:100)",
        "changeType",
        "changedFiles",
    ):
        assert field in q
    assert q.count("rateLimit{cost remaining resetAt}") == 1


def test_pagination_over_100_and_storage_layout(tmp_path: Path) -> None:
    s = scraper(tmp_path, Fake({5: 250, 1007: 2}))
    s.run([1007, 5])
    rows = rows_on_disk(s)
    assert (
        len(rows[5]["files"]["nodes"]) == 250
        and rows[5]["files_truncated"] is False
        and rows[5]["_schema"] == 2
    )
    assert sorted(p.name for p in s.base.glob("*.jsonl.gz")) == ["00000.jsonl.gz", "00001.jsonl.gz"]
    assert rows[5]["baseRefName"] == "release-1.30"
    assert json.loads(s.ckpt_path.read_text())["done"] == [5, 1007]


def test_truncation_at_300_keeps_total_count(tmp_path: Path) -> None:
    s = scraper(tmp_path, Fake({7: 1234}))
    s.run([7])
    r = rows_on_disk(s)[7]
    assert (
        len(r["files"]["nodes"]) == 300
        and r["files"]["totalCount"] == 1234
        and r["files_truncated"] is True
    )


def test_exactly_300_files_is_complete_not_truncated(tmp_path: Path) -> None:
    s = scraper(tmp_path, Fake({7: 300}))
    s.run([7])
    r = rows_on_disk(s)[7]
    assert len(r["files"]["nodes"]) == 300 and r["files_truncated"] is False


def test_resume_skips_done_numbers(tmp_path: Path) -> None:
    scraper(tmp_path, Fake({1: 1, 2: 1, 3: 1})).run([1, 2])
    fake2 = Fake({1: 1, 2: 1, 3: 1})
    scraper(tmp_path, fake2).run([1, 2, 3])
    assert len(fake2.queries) == 1 and "pullRequest(number:3)" in fake2.queries[0]
    assert "pullRequest(number:1)" not in fake2.queries[0]


def test_missing_pr_is_recorded_not_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    fake = Fake({1: 2, 9: None, 3: 1})
    s = scraper(tmp_path, fake)
    s.run([1, 9, 3])
    rows = rows_on_disk(s)
    assert (
        rows[9]["missing"] is True
        and "files" not in rows[9]
        and len(rows[1]["files"]["nodes"]) == 2
    )
    ck = json.loads(s.ckpt_path.read_text())
    assert ck["missing"] == [9] and ck["done"] == [1, 3, 9]
    assert len(fake.queries) == 1 and not sleeps  # a missing PR is not retried as a transient error


def test_completeness_failure_names_the_pr(tmp_path: Path) -> None:
    class Short(Fake):
        def page(self, n: int, start: int, first: bool = False) -> dict[str, Any]:
            out = super().page(n, start, first)
            if first:
                out["totalCount"] += 1
            return out

    with pytest.raises(pf.sg.ScrapeError, match=r"PR #7 .*!= totalCount 4"):
        scraper(tmp_path, Short({7: 3})).run([7])
    assert not list(tmp_path.rglob("*.jsonl.gz"))


def test_batch_size_falls_back_on_502(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    fake = Fake({n: 1 for n in range(1, 11)}, fail_over=2)
    s = scraper(tmp_path, fake)
    s.run(list(range(1, 11)))
    assert s.sizes[0] == 2 and sorted(s.done) == list(range(1, 11))


def test_atomic_checkpoint_summary_and_no_write_outside(tmp_path: Path) -> None:
    data = tmp_path / "data"
    s = scraper(data, Fake({1: 2}))
    s.run([1])
    assert not list(tmp_path.rglob("*.tmp"))
    assert json.loads(s.ckpt_path.read_text())["windows"] == {}  # main --summary compatibility
    assert {p.relative_to(tmp_path).parts[:4] for p in tmp_path.rglob("*") if p.is_file()} == {("data", "raw", "o__r", "pr_files")}  # fmt: skip
    summ = pf.summarize(data)["o__r"]
    assert (
        summ["ckpt_done"] == 1
        and summ["files"] == 2
        and summ["base_ref_top"] == [("release-1.30", 1)]
    )


def test_token_not_in_errors_or_stored_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(payload: dict[str, Any]) -> Any:
        raise ConnectionError("ConnectError")

    monkeypatch.setattr(time, "sleep", lambda s: None)
    with pytest.raises(pf.sg.ScrapeError) as ei:
        pf.PrFilesScraper(pf.qcm.PacedClient(pf.tolerant(boom)), "o/r", tmp_path, 1).run([1])
    assert TOKEN not in str(ei.value)
    scraper(tmp_path, Fake({2: 1})).run([2])
    blob = b"".join(gzip.decompress(p.read_bytes()) for p in tmp_path.rglob("*.jsonl.gz"))
    assert TOKEN.encode() not in blob


def test_scope_builder_unions_merged_raw_and_fix_link_prs(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "o__r" / "prs" / "2024"
    raw.mkdir(parents=True)
    with gzip.open(raw / "2024-01-01.jsonl.gz", "wt", encoding="utf-8") as f:
        for n, merged in ((30, True), (10, True), (20, False), (30, True)):
            f.write(json.dumps({"number": n, "merged": merged}) + "\n")
    derived = tmp_path / "derived" / "o__r"
    derived.mkdir(parents=True)
    pd.DataFrame(
        {
            "issue": [1, 2, 3, 4], "pr": [5, 10, 6, 7],
            "method": ["closed_event_closer_pr", "closed_event_closer_pr", "pr_closing_refs", "closed_event_closer_pr"],
            "pr_merged": [True, True, True, False],
        }
    ).to_parquet(derived / "fix_links.parquet")  # fmt: skip
    assert pf.build_scope(tmp_path, derived, "o/r") == [
        5,
        10,
        30,
    ]  # 20 unmerged, 6 wrong method, 7 unmerged


def test_batch_regrows_after_clean_run_capped_at_max(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    monkeypatch.setattr(pf.qcm, "REGROW_AFTER", 10**6)  # no regrow during the burst
    fake = Fake({n: 1 for n in range(1, 200)}, fail_over=1)
    s = scraper(tmp_path, fake, batch=5)
    s.run(list(range(1, 6)))
    assert s.sizes[0] == 1
    s._ok_streak = 0  # successes during the burst must not pre-load the streak
    fake.fail_over, seen = None, len(fake.queries)
    monkeypatch.setattr(pf.qcm, "REGROW_AFTER", 3)
    s.run(list(range(10, 100)))
    sizes = [len(re.findall(r"pullRequest\(number:", q)) for q in fake.queries[seen:]]
    assert s.sizes[0] == 5 and max(sizes) == 5 and sizes[:-1] == sorted(sizes[:-1])
    monkeypatch.setattr(pf.qcm, "REGROW_AFTER", 10**6)  # isolate the shrink from regrow
    fake.fail_over = 2  # a failure after regrow shrinks again
    s.run(list(range(100, 120)))
    assert s.sizes[0] == 2
