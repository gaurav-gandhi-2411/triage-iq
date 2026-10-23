from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp3b_build_evidence as ev  # noqa: E402


def c(author: str | None, body: str) -> dict[str, str | None]:
    return {"author": author, "body": body, "created_at": "2024-01-01T00:00:00Z"}


def test_is_bot_variants() -> None:
    bots = ["k8s-ci-robot", "Kubernetes-Prow", "k8s-triage-robot", "dependabot[bot]",
            "redhat-chai-bot", "vscode-triage-bot"]  # fmt: skip
    assert all(ev.is_bot(x) for x in bots)
    assert not any(ev.is_bot(x) for x in ["aojea", "robertbotez", "vbottu", None, ""])


def test_last_human_comment_skips_bots_and_commands() -> None:
    cs = [c("aojea", "first human"), c("liggitt", "fixed in the apiserver"),
          c("k8s-ci-robot", "Closing this issue"), c("dims", "/lgtm\n/approve")]  # fmt: skip
    assert ev.last_human_comment(cs) == ("liggitt", "fixed in the apiserver")


def test_last_human_comment_empty_cases() -> None:
    assert ev.last_human_comment([]) == ("", "")
    assert ev.last_human_comment(None) == ("", "")
    assert ev.last_human_comment([c("k8s-ci-robot", "x"), c("a", "   ")]) == ("", "")


def test_truncate_collapses_whitespace_and_marks_cut() -> None:
    assert ev.truncate("a \n\n b") == "a b"
    out = ev.truncate("x" * 500)
    assert len(out) == ev.MAX_CHARS and out.endswith("…")
    assert ev.truncate("x" * ev.MAX_CHARS) == "x" * ev.MAX_CHARS


def test_clean_pr_body_uses_what_this_pr_does_section() -> None:
    body = ("<!-- tips -->\n#### What type of PR is this?\n/kind bug\n"
            "#### What this PR does / why we need it:\nFixes the leak in\n```go\ncode\n```\n"
            "kubelet.\n/sig node\n")  # fmt: skip
    assert ev.clean_pr_body(body) == "Fixes the leak in kubelet."
    assert ev.clean_pr_body(None) == "" and ev.clean_pr_body("<!-- only -->") == ""


def _tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    iss = pd.DataFrame({
        "number": [1, 2, 3, 4], "state": ["CLOSED", "CLOSED", "OPEN", "CLOSED"],
        "closing_comments": [[c("a", "done")], [c("k8s-ci-robot", "bot")], [c("a", "tail")], []],
    })  # fmt: skip
    fl = pd.DataFrame({
        "issue": [1, 2, 2, 3], "pr": [10, 11, 12, 10],
        "method": ["closed_event_closer_pr", "closed_event_closer_pr", "pr_closing_refs",
                   "closed_event_closer_pr"],
        "pr_merged": [True, False, True, True],
    })  # fmt: skip
    prs = pd.DataFrame({"number": [10, 11, 12], "title": ["fix A", "fix B", "fix C"],
                        "body": ["body A", "body B", "body C"]})  # fmt: skip
    return iss, fl, prs


def test_build_evidence_audited_links_only_and_closed_comments_only() -> None:
    out = ev.build_evidence(*_tables()).set_index("number")
    assert [p["number"] for p in out.loc[1, "fix_prs"]] == [10]
    assert out.loc[1, "closing_comment"] == "done"
    # issue 2: unmerged closer link and an unaudited method are both excluded; bot comment too
    assert list(out.loc[2, "fix_prs"]) == [] and not out.loc[2, "has_evidence"]
    # issue 3: audited merged link counts, but an OPEN issue's comment tail is not evidence
    assert out.loc[3, "has_fix_evidence"] and not out.loc[3, "has_comment_evidence"]
    assert not out.loc[4, "has_evidence"]


def test_build_evidence_link_to_missing_pr_is_no_evidence() -> None:
    iss, fl, prs = _tables()
    out = ev.build_evidence(iss, fl, prs[prs["number"] != 10]).set_index("number")
    assert not out.loc[1, "has_fix_evidence"]


def test_iso_utc_normalises_and_fails_to_empty() -> None:
    assert ev.iso_utc("2024-01-02T03:04:05Z") == "2024-01-02T03:04:05Z"
    assert ev.iso_utc(pd.Timestamp("2024-01-02 05:04:05+02:00")) == "2024-01-02T03:04:05Z"
    assert ev.iso_utc(None) == "" and ev.iso_utc(pd.NaT) == "" and ev.iso_utc("not a date") == ""


def test_build_evidence_records_item_dates() -> None:
    iss, fl, prs = _tables()
    prs = prs.assign(merged_at=pd.to_datetime(["2023-05-06T07:08:09Z"] * 3, utc=True))
    out = ev.build_evidence(iss, fl, prs).set_index("number")
    assert out.loc[1, "fix_prs"][0]["merged_at"] == "2023-05-06T07:08:09Z"
    assert out.loc[1, "closing_comment_at"] == "2024-01-01T00:00:00Z"
    assert out.loc[4, "closing_comment_at"] == ""


def test_build_evidence_falls_back_to_link_merged_at_then_empty() -> None:
    iss, fl, prs = _tables()
    out = ev.build_evidence(iss, fl, prs).set_index("number")  # no merged_at anywhere
    assert out.loc[1, "fix_prs"][0]["merged_at"] == ""
    fl = fl.assign(pr_merged_at=pd.to_datetime(["2022-01-01T00:00:00Z"] * 4, utc=True))
    out = ev.build_evidence(iss, fl, prs).set_index("number")
    assert out.loc[1, "fix_prs"][0]["merged_at"] == "2022-01-01T00:00:00Z"
