from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
_SPEC = importlib.util.spec_from_file_location("qp0a", _SCRIPTS / "qp0_fixlink_audit.py")
qp0a = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0a"] = qp0a
_SPEC.loader.exec_module(qp0a)

REPO, ISSUE, PR, SHA = "o/r", 7, 99, "abc123"
MERGED = "2024-05-01T12:00:00Z"
TOKEN = "ghp_SECRET_TOKEN_VALUE"  # noqa: S105 - fake, to prove it never leaks


def fake_api(routes: dict[str, Any], calls: list[str] | None = None) -> Any:
    """Api over a dict transport: path -> (status, body) | Exception."""

    def transport(url: str, params: dict[str, Any]) -> tuple[int, dict[str, str], Any]:
        path = url.removeprefix(qp0a.API)
        if calls is not None:
            calls.append(path)
        r = routes.get(path, (200, []))
        if isinstance(r, Exception):
            raise r
        # one page only: later pages are empty
        return r[0], {}, r[1] if params.get("page", 1) == 1 else []

    return qp0a.Api(transport, sleep=lambda s: None)


def routes(
    *,
    closed_at: str = MERGED,
    commit: str | None = None,
    xref: bool = False,
    body: str = "",
    merged: bool = True,
    extra_events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pr = {
        "merged": merged,
        "merged_at": MERGED,
        "merge_commit_sha": SHA,
        "title": "t",
        "body": body,
    }
    ev = [{"event": "closed", "created_at": closed_at, "commit_id": commit}]
    tl = [{"event": "closed", "created_at": closed_at}]
    if xref:
        src = {"issue": {"number": PR, "pull_request": {}}}
        tl.append({"event": "cross-referenced", "source": src})
    return {
        f"/repos/{REPO}/pulls/{PR}": (200, pr),
        f"/repos/{REPO}/issues/{ISSUE}/events": (200, ev + (extra_events or [])),
        f"/repos/{REPO}/issues/{ISSUE}/timeline": (200, tl),
    }


def judge(r: dict[str, Any]) -> dict[str, Any]:
    return qp0a.judge(fake_api(r), REPO, ISSUE, PR)


def test_commit_match_short_circuits_timeline() -> None:
    calls: list[str] = []
    res = qp0a.judge(
        fake_api(routes(commit=SHA, closed_at="2020-01-01T00:00:00Z"), calls), REPO, 7, 99
    )
    assert (res["verdict"], res["evidence"]) == ("CONFIRMED", "commit_match")
    assert not any("timeline" in c for c in calls)


def test_cross_ref_with_timing() -> None:
    res = judge(routes(closed_at="2024-05-01T12:01:30Z", xref=True))
    assert (res["verdict"], res["evidence"]) == ("CONFIRMED", "cross_ref_timing")


def test_keyword_with_timing() -> None:
    res = judge(routes(closed_at="2024-05-01T11:58:30Z", body="Fixes #7\n"))
    assert (res["verdict"], res["evidence"]) == ("CONFIRMED", "keyword_timing")


def test_keyword_must_match_exact_issue_number() -> None:
    assert qp0a.has_keyword("fixes #7", 7, REPO)
    assert qp0a.has_keyword("Closes o/r#7.", 7, REPO)
    assert qp0a.has_keyword("resolves https://github.com/o/r/issues/7", 7, REPO)
    assert not qp0a.has_keyword("fixes #77", 7, REPO)
    assert not qp0a.has_keyword("see #7", 7, REPO)


def test_pr_404_is_wrong() -> None:
    r = routes()
    r[f"/repos/{REPO}/pulls/{PR}"] = (404, {})
    assert (judge(r)["verdict"], judge(r)["reason"]) == ("WRONG", "pr_not_in_repo")


def test_unmerged_is_wrong() -> None:
    res = judge(routes(merged=False))
    assert (res["verdict"], res["reason"]) == ("WRONG", "pr_not_merged")


def test_closed_before_merge_is_wrong() -> None:
    res = judge(routes(closed_at="2024-05-01T10:00:00Z", xref=True))
    assert (res["verdict"], res["reason"]) == ("WRONG", "closed_before_pr_merged")


def test_closed_before_merge_but_reopened_is_unverifiable() -> None:
    ev = [{"event": "reopened", "created_at": "2024-05-01T10:30:00Z"}]
    res = judge(routes(closed_at="2024-05-01T10:00:00Z", extra_events=ev))
    assert (res["verdict"], res["reason"]) == ("UNVERIFIABLE", "reopened_after_close")


def test_no_relation_is_wrong() -> None:
    res = judge(routes(closed_at="2024-05-03T12:00:00Z"))
    assert (res["verdict"], res["reason"]) == ("WRONG", "no_relation")


def test_ambiguous_timing_is_unverifiable() -> None:
    res = judge(routes(closed_at="2024-05-01T12:20:00Z"))
    assert (res["verdict"], res["reason"]) == ("UNVERIFIABLE", "ambiguous_timing")


def test_xref_but_late_close_is_unverifiable() -> None:
    res = judge(routes(closed_at="2024-05-01T12:20:00Z", xref=True))
    assert res["reason"] == "structural_evidence_but_timing"


def test_transferred_issue_is_unverifiable() -> None:
    res = judge(routes(extra_events=[{"event": "transferred", "created_at": MERGED}]))
    assert (res["verdict"], res["reason"]) == ("UNVERIFIABLE", "issue_transferred")


def test_api_error_after_retries_is_unverifiable_and_leaks_no_token() -> None:
    r = routes()
    r[f"/repos/{REPO}/issues/{ISSUE}/events"] = ConnectionError("ConnectError")
    res = judge(r)
    assert res["verdict"] == "UNVERIFIABLE" and res["reason"].startswith("api_error")
    assert TOKEN not in json.dumps(res)
    r[f"/repos/{REPO}/pulls/{PR}"] = (500, {})
    assert judge(r)["reason"] == "api_error:retries_exhausted"


def test_http_error_message_has_no_token() -> None:
    api = fake_api({"/x": (401, {})})
    with pytest.raises(qp0a.ApiError) as ei:
        api.get("/x")
    assert TOKEN not in str(ei.value) and str(ei.value) == "http_401"


def test_rate_limit_sleeps_until_reset() -> None:
    sleeps: list[float] = []
    seq = iter([(403, {"X-RateLimit-Reset": "1100"}, {}), (200, {}, {"ok": 1})])
    api = qp0a.Api(lambda u, p: next(seq), sleep=sleeps.append, now=lambda: 1000.0)
    assert api.get("/x") == (200, {"ok": 1})
    assert sleeps == [102.0]


def test_wilson_known_values() -> None:
    lo, hi = qp0a.wilson(95, 100)
    assert (round(lo, 4), round(hi, 4)) == (0.8882, 0.9785)
    lo, hi = qp0a.wilson(0, 10)
    assert lo == 0.0 and round(hi, 4) == 0.2775
    assert qp0a.wilson(1, 1)[1] == 1.0
    assert qp0a.wilson(0, 0) is None


def test_sampling_deterministic_and_verdict_blind() -> None:
    links = [(i, 1000 + i, y) for i, y in enumerate([2023, 2024, 2025, 2026] * 20)]
    a = qp0a.stratified_sample(links, 5, 42)
    assert a == qp0a.stratified_sample(list(reversed(links)), 5, 42)
    assert a != qp0a.stratified_sample(links, 5, 43)
    assert len(a) == 20 and all(len(x) == 3 for x in a)
    assert len(qp0a.stratified_sample(links[:6], 5, 42)) == 6  # min(N, available)


def test_run_audit_end_to_end(tmp_path: Path) -> None:
    pd.DataFrame({"issue": [ISSUE, 8], "pr": [PR, 100], "method": [qp0a.METHOD] * 2}).to_parquet(
        tmp_path / "fix_links.parquet"
    )
    pd.DataFrame({"number": [ISSUE, 8], "created_year": [2024, 2022]}).to_parquet(
        tmp_path / "issues.parquet"
    )
    api = fake_api(routes(commit=SHA))
    rep = qp0a.run_audit(api, REPO, tmp_path, 30, 42)
    assert rep["overall"]["n_sampled"] == 1  # the 2022 issue is outside the strata
    assert rep["acceptance"]["result"] == "pass"
    assert rep["links"][0]["pr_url"] == f"https://github.com/{REPO}/pull/{PR}"


def test_keyword_adversarial_prefix_and_foreign_repo() -> None:
    assert not qp0a.has_keyword("prefix #7", 7, REPO)  # "fix" inside a word
    assert not qp0a.has_keyword("we disclose #7", 7, REPO)  # "close" inside a word
    assert not qp0a.has_keyword("fixes https://github.com/x/y/issues/7", 7, REPO)
    assert not qp0a.has_keyword("fixes x/y#7", 7, REPO)
    assert qp0a.has_keyword("Fixes: #7", 7, REPO)


def test_closed_events_without_timestamps_do_not_crash() -> None:
    r = routes()
    r[f"/repos/{REPO}/issues/{ISSUE}/events"] = (200, [{"event": "closed"}])
    r[f"/repos/{REPO}/issues/{ISSUE}/timeline"] = (200, [{"event": "closed"}])
    res = judge(r)
    assert (res["verdict"], res["reason"]) == ("UNVERIFIABLE", "no_closed_event_time")
