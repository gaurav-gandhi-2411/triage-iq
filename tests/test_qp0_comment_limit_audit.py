from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("qp0c", _SCRIPTS / "qp0_comment_limit_audit.py")
qp0c = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0c"] = qp0c
_SPEC.loader.exec_module(qp0c)

T0 = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)
TOKEN = "ghp_SECRET_TOKEN_VALUE"  # noqa: S105 - fake, to prove it never leaks


def cm(i: int, body: str = "x", assoc: str = "NONE", kind: str = "User", mins: int | None = None):
    t = T0 + timedelta(minutes=i if mins is None else mins)
    return {
        "id": f"c{i}", "body": body, "createdAt": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "authorAssociation": assoc, "author": {"login": f"u{i}", "__typename": kind},
    }  # fmt: skip


def meta(**kw: Any) -> dict[str, Any]:
    base = {"number": 1, "year": 2024, "state_reason": "COMPLETED", "labels": [], "n_comments": 9}
    return base | {"samples": ["A", "B"], "repo": "o/r"} | kw


def filler(n: int, start: int = 0) -> list[dict[str, Any]]:
    return [cm(start + i) for i in range(n)]


def test_marker_only_in_full_list_is_missed() -> None:
    comments = [cm(0, "/close\nDuplicate of #55", "MEMBER")] + filler(7, 1)
    row = qp0c.analyse(meta(), comments, [T0 + timedelta(minutes=1)])
    assert row["pairs_full"] == [55] and row["pairs_saved"] == []
    assert row["pairs_only_full"] == [55]


def test_marker_inside_last_five_is_not_missed() -> None:
    comments = filler(6) + [cm(6, "/close\nDuplicate of #55", "MEMBER")]
    row = qp0c.analyse(meta(), comments, [T0 + timedelta(minutes=6)])
    assert row["pairs_only_full"] == [] and row["pairs_saved"] == [55]


def test_explanation_outside_last_five_is_missed() -> None:
    comments = [cm(0, "fixed by #3", "MEMBER")] + filler(7, 1)
    row = qp0c.analyse(meta(), comments, [T0 + timedelta(minutes=2)])
    assert row["explanation_id"] == "c0" and row["explanation_missed"] is True


def test_explanation_inside_last_five_and_after_close_window() -> None:
    comments = filler(3) + [cm(3, "done", "OWNER"), cm(4, "later", "OWNER", mins=60)]
    row = qp0c.analyse(meta(), comments, [T0 + timedelta(minutes=3)])
    assert row["explanation_id"] == "c3" and row["explanation_missed"] is False  # c4 is > +5 min


def test_bots_and_non_staff_are_not_explanations() -> None:
    comments = [cm(0, "x", "MEMBER", kind="Bot"), cm(1, "x", "CONTRIBUTOR"), cm(2, "x", "NONE")]
    row = qp0c.analyse(meta(), comments, [T0 + timedelta(minutes=5)])
    assert row["explanation_id"] is None and row["explanation_missed"] is None


def test_wilson_known_values() -> None:
    assert qp0c.wilson(95, 100) == [0.8882, 0.9785]
    assert qp0c.wilson(0, 10) == [0.0, 0.2775]
    assert qp0c.wilson(0, 0) is None


def _row(i: int, bucket: str, missed: bool) -> dict[str, Any]:
    return {
        "issue": i, "year": 2024, "state_reason": "COMPLETED", "kind": "none", "bucket": bucket,
        "total_comments": 8, "samples": ["A"], "pairs_full": [1], "marker_method": "m",
        "pairs_only_full": [1] if missed else [], "pairs_saved": [], "explanation_id": None,
        "explanation_assoc": None, "explanation_missed": None,
    }  # fmt: skip


def test_bucket_flagging_rule() -> None:
    rows = [_row(i, "11-20", i < 5) for i in range(25)]  # 20% in one bucket, n >= 20
    rows += [_row(100 + i, "6-10", False) for i in range(200)]
    flags = qp0c.summarise(rows, "A")["pattern_flags"]["markers"]
    assert [(f["dimension"], f["value"]) for f in flags] == [("bucket", "11-20")]
    small = [_row(i, "11-20", i < 5) for i in range(19)] + [
        _row(100 + i, "6-10", False) for i in range(200)
    ]
    assert qp0c.summarise(small, "A")["pattern_flags"]["markers"] == []  # n < 20: no flag
    zero = [_row(i, "11-20", False) for i in range(30)]
    assert qp0c.summarise(zero, "A")["pattern_flags"]["markers"] == []  # no misses: no flag


def test_sampling_deterministic_and_verdict_blind() -> None:
    items = [(i, 2023 + i % 4) for i in range(400)]
    a = qp0c.sample_b(items, 10, 42)
    assert a == qp0c.sample_b(list(reversed(items)), 10, 42) and len(a) == 40
    assert a != qp0c.sample_b(items, 10, 7)
    nums = list(range(300))
    assert qp0c.sample_a(nums, 150, 42) == qp0c.sample_a(nums[::-1], 150, 42)
    assert qp0c.sample_a(nums[:20], 150, 42) == nums[:20]


def _issue_df(n: int) -> pd.DataFrame:
    cc = [{"body": "Duplicate of #5", "created_at": None, "author_assoc": "MEMBER", "author": "m"}]
    return pd.DataFrame(
        {
            "number": list(range(1, n + 1)), "node_id": [f"N{i}" for i in range(1, n + 1)],
            "state": "CLOSED", "state_reason": "COMPLETED", "created_year": 2024,
            "labels_final": [["kind/bug"]] * n, "n_comments": 9, "closed_as_duplicate": False,
            "closing_comments": [cc] * n,
        }
    )  # fmt: skip


def test_run_end_to_end_with_fake_transport_and_no_token_leak(tmp_path: Path) -> None:
    _issue_df(3).to_parquet(tmp_path / "issues.parquet")
    pd.DataFrame({"source": [], "method": []}).to_parquet(tmp_path / "duplicate_pairs.parquet")
    nodes = [cm(0, "/close\nDuplicate of #5", "MEMBER")] + filler(7, 1)
    data = {
        "rateLimit": {"cost": 1, "remaining": 5000, "resetAt": "2030-01-01T00:00:00Z"},
        "node": {
            "comments": {"nodes": nodes, "pageInfo": {"hasNextPage": False, "endCursor": None}},
            "closedEvents": {"nodes": [{"createdAt": "2024-03-01T12:01:00Z"}]},
        },
    }
    client = qp0c.sg.Client(lambda payload: (200, {}, {"data": data}))
    rep = qp0c.run(client, "o/r", tmp_path, 2, 42)
    a = rep["sample_A_markers"]
    assert a["markers_pair_level"]["k"] == 3 and a["markers_pair_level"]["n"] == 3
    assert rep["verdict"]["material_markers"] and rep["verdict"]["rescrape_comments_recommended"]
    assert rep["rescrape_scope"] == [1, 2, 3]
    assert TOKEN not in str(rep)


def test_token_not_in_transport_errors() -> None:
    def boom(payload: dict[str, Any]) -> Any:
        raise ConnectionError("ConnectionError")

    client = qp0c.sg.Client(boom)
    with pytest.raises(qp0c.sg.ScrapeError) as ei:
        client.query("q", {}, tries=1)
    assert TOKEN not in str(ei.value)
