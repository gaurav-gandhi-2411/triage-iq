from __future__ import annotations

import gzip
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp0tl", _S / "qp0_scrape_timelines.py")
tl = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0tl"] = tl
_SPEC.loader.exec_module(tl)

TOKEN = "ghp_SECRET_TOKEN_VALUE"  # noqa: S105 - fake, to prove it never leaks
RL = {"cost": 1, "remaining": 5000, "resetAt": "2030-01-01T00:00:00Z"}


class Fake:
    """Fake GraphQL transport over {issue number: event count | None (missing)}."""

    def __init__(
        self, totals: dict[int, int | None], fail_over: int | None = None, status: int = 502
    ):
        self.totals, self.queries, self.fail_over, self.status = totals, [], fail_over, status
        self.total_override: int | None = None

    def page(self, n: int, start: int, first: bool = False) -> dict[str, Any]:
        total = self.totals[n]
        end = min(start + 100, total)
        nodes = [
            {"__typename": "LabeledEvent", "createdAt": f"2024-01-01T00:00:{i % 60:02d}Z", "label": {"name": f"l{i}"}}
            for i in range(start, end)
        ]  # fmt: skip
        out = {"pageInfo": {"hasNextPage": end < total, "endCursor": f"after{end}"}, "nodes": nodes}
        if not first:
            return out
        return {**out, "totalCount": self.total_override or total}

    def __call__(self, payload: dict[str, Any]) -> tuple[int, dict[str, str], dict[str, Any]]:
        q, v = payload["query"], payload["variables"]
        self.queries.append(q)
        nums = [int(x) for x in re.findall(r"issue\(number:(\d+)\)", q)]
        if self.fail_over is not None and len(nums) > self.fail_over:
            return self.status, {}, {"message": "Bad Gateway"}
        if "node(id" in q:
            n, k = int(v["id"][2:]), int(v["tl"][5:])
            return 200, {}, {"data": {"rateLimit": RL, "node": {"timelineItems": self.page(n, k)}}}
        repo, errors = {}, []
        for n in nums:
            if self.totals[n] is None:
                repo[f"p{n}"] = None
                errors.append(
                    {"type": "NOT_FOUND", "path": ["repository", f"p{n}"], "message": "x"}
                )
                continue
            repo[f"p{n}"] = {"number": n, "id": f"IS{n}", "timelineItems": self.page(n, 0, True)}
        body: dict[str, Any] = {"data": {"rateLimit": RL, "repository": repo}}
        if errors:
            body["errors"] = errors
        return 200, {}, body


def scraper(tmp_path: Path, fake: Any, batch: int = 10):  # noqa: ANN201
    client = tl.qcm.PacedClient(tl.pf.tolerant(tl.transient_499(fake)))
    return tl.TimelineScraper(client, "o/r", tmp_path, batch)


def on_disk(s) -> dict[int, dict]:  # noqa: ANN001
    return {r["number"]: r for p in sorted(s.base.glob("*.jsonl.gz")) for r in s._read(p)}


def test_selection_identical_to_main_scraper() -> None:
    q = tl.batch_query([5, 1007])
    assert "p5: issue(number:5)" in q and "p1007: issue(number:1007)" in q
    assert q.count(tl.sg.TIMELINE.replace(", after:$tl", "")) == 2  # itemTypes + node fields
    assert "after:" not in q and q.count("{") == q.count("}")
    # continuation pages reuse the main scraper's node(id:) query verbatim
    assert tl.PAGE_Q is tl.sg.TIMELINE_Q and "after:$tl" in tl.PAGE_Q
    assert q.count("repository{nameWithOwner}") >= 2 * tl.sg.TIMELINE.count(
        "repository{nameWithOwner}"
    )


def test_pagination_over_100_events_layout_and_stamps(tmp_path: Path) -> None:
    s = scraper(tmp_path, Fake({5: 250, 1007: 2}))
    assert s.run([1007, 5]) == 1
    rows = on_disk(s)
    assert len(rows[5]["timelineItems"]["nodes"]) == 250
    assert len(rows[1007]["timelineItems"]["nodes"]) == 2
    assert {p.name for p in s.base.glob("*.jsonl.gz")} == {"00000.jsonl.gz", "00001.jsonl.gz"}
    assert all(r["_schema"] == 2 and r["_source"] == "direct" for r in rows.values())
    assert all(r["_scraped_at"] for r in rows.values())
    assert s.stats["events"] == 252


def test_resume_is_idempotent(tmp_path: Path) -> None:
    fake = Fake({n: 1 for n in range(1, 25)})
    s = scraper(tmp_path, fake)
    assert s.run(list(range(1, 25)), max_batches=1) == 1
    s2 = scraper(tmp_path, fake)
    assert len(s2.done) == 10
    assert s2.run(list(range(1, 25))) == 2
    assert s2.run(list(range(1, 25))) == 0
    assert sorted(on_disk(s2)) == list(range(1, 25))


def test_completeness_failure_names_the_issue(tmp_path: Path) -> None:
    fake = Fake({7: 3})
    fake.total_override = 4
    with pytest.raises(tl.sg.ScrapeError, match="issue #7 saved 3 events != totalCount 4"):
        scraper(tmp_path, fake).run([7])
    assert not list(tmp_path.rglob("*.jsonl.gz"))


def test_missing_issue_recorded(tmp_path: Path) -> None:
    s = scraper(tmp_path, Fake({1: 2, 2: None}))
    s.run([1, 2])
    rows = on_disk(s)
    assert rows[2]["missing"] is True and rows[2]["_source"] == "direct"
    assert s.ckpt["missing"] == [2] and s.ckpt["done"] == [1, 2]


@pytest.mark.parametrize("status", [502, 499])
def test_batch_fallback_10_5_2_1_including_499(tmp_path: Path, monkeypatch, status: int) -> None:  # noqa: ANN001
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = Fake({n: 1 for n in range(1, 11)}, fail_over=2, status=status)
    s = scraper(tmp_path, fake)
    s.run(list(range(1, 11)))
    assert sorted(on_disk(s)) == list(range(1, 11))
    assert s.sizes[0] == 2
    sizes = {len(re.findall(r"issue\(number:", q)) for q in fake.queries}
    assert {10, 5, 2} <= sizes


def test_token_never_logged(tmp_path: Path, caplog) -> None:  # noqa: ANN001
    def fake(payload: dict[str, Any]):  # noqa: ANN202
        assert TOKEN not in json.dumps(payload)
        return Fake({1: 1})(payload)

    caplog.set_level("DEBUG")
    scraper(tmp_path, fake).run([1])
    assert TOKEN not in caplog.text


def _ev(t: str, created: str, label: str | None = None, **kw: Any) -> dict[str, Any]:
    e: dict[str, Any] = {"__typename": t, "createdAt": created, **kw}
    if label:
        e["label"] = {"name": label}
    return e


def test_compare_on_fixtures() -> None:
    c = "2024-03-01T00:00:00Z"
    a1 = _ev("LabeledEvent", "2024-03-01T01:00:00Z", "kind/bug")
    a2 = _ev("LabeledEvent", "2024-03-01T02:00:00Z", "sig/node")
    a3 = _ev("LabeledEvent", "2024-03-01T03:00:00Z", "sig/apps")
    pr = _ev("ClosedEvent", "2024-03-02T00:00:00Z", closer={"number": 9})
    emb = {  # 1: dropped label; 2: identical; 3: earliest sig label changes
        1: {"created": c, "events": [a2]},
        2: {"created": c, "events": [a2, pr]},
        3: {"created": c, "events": [a3]},
    }
    dire = {
        1: {"created": c, "events": [a1, a2]},
        2: {"created": c, "events": [a2, pr]},
        3: {"created": c, "events": [a2, a3]},
    }
    rep = tl.compare(emb, dire, "sig/")
    assert rep["n_issues_compared"] == 3
    assert rep["event_counts_by_type"]["embedded"]["LabeledEvent"] == 3
    assert rep["event_counts_by_type"]["direct"]["LabeledEvent"] == 5
    assert rep["share_issues_direct_has_more_events"] == pytest.approx(2 / 3, abs=1e-6)
    assert rep["labeled_event"]["missing_count_distribution"] == {"0": 1, "1": 2}
    assert rep["first_label_event_changed"]["n"] == 1  # only #3 (sig/apps -> sig/node)
    assert rep["labels_at_created_plus_72h_changed"]["n"] == 1
    assert rep["focus_events"]["closed_by_pr"] == {
        "embedded": 1,
        "direct": 1,
        "issues_differing": 0,
    }
    assert rep["by_created_year"]["2024"]["n"] == 3
    assert tl.compare(emb, dire, "")["first_label_event_changed"]["n"] == 2  # vscode: all labels


def test_compare_cli_reads_raw_files(tmp_path: Path) -> None:
    def w(p: Path, rows: list[dict]) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8") as f:
            f.writelines(json.dumps(r) + "\n" for r in rows)

    ev = _ev("LabeledEvent", "2024-03-01T01:00:00Z", "sig/node")
    base = tmp_path / "raw" / "o__r"
    issue = {"number": 1, "createdAt": "2024-03-01T00:00:00Z", "timelineItems": {"nodes": []}}
    w(base / "issues" / "2024" / "2024-03-01.jsonl.gz", [issue])
    direct = [{"number": 1, "timelineItems": {"nodes": [ev]}}, {"number": 2, "missing": True}]
    w(base / "timelines" / "00000.jsonl.gz", direct)
    out = tmp_path / "out"
    argv = ["--repo", "o/r", "--compare", "--data-dir", str(tmp_path), "--out-dir", str(out)]
    assert tl.main(argv) == 0
    rep = json.loads((out / "qp0_timeline_completeness_o__r_2026-10.json").read_text())
    assert rep["n_issues_compared"] == 1
    assert rep["labeled_event"]["share_issues_missing_ge1"] == 1.0
