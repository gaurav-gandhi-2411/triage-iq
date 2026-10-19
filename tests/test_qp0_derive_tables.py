from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # qp0_derive_tables imports its sibling qp0_derive_stats
_SPEC = importlib.util.spec_from_file_location("qp0d", _SCRIPTS / "qp0_derive_tables.py")
qp0d = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0d"] = qp0d
_SPEC.loader.exec_module(qp0d)

REPO = "o/r"


def CANON(n: int, repo: str | None = "o/r") -> dict[str, Any]:  # noqa: N802
    return {"number": n, **({"repository": {"nameWithOwner": repo}} if repo else {})}


T0 = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


def ts(hours: float) -> str:
    return (T0 + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def lab(h: float, name: str, on: bool = True) -> dict[str, Any]:
    typ = "LabeledEvent" if on else "UnlabeledEvent"
    return {"__typename": typ, "createdAt": ts(h), "actor": {"login": "u"}, "label": {"name": name}}


def comment(body: str, login: str = "human", h: float = 1) -> dict[str, Any]:
    return {
        "body": body,
        "createdAt": ts(h),
        "authorAssociation": "NONE",
        "author": {"login": login},
    }


def issue(
    number: int = 1,
    labels: list[str] | None = None,
    nodes: list[dict[str, Any]] | None = None,
    comments: list[dict[str, Any]] | None = None,
    state: str = "CLOSED",
) -> dict[str, Any]:
    return {
        "id": f"I_{number}",
        "number": number,
        "title": "T",
        "body": "B",
        "state": state,
        "stateReason": "COMPLETED" if state == "CLOSED" else None,
        "createdAt": ts(0),
        "closedAt": ts(48) if state == "CLOSED" else None,
        "authorAssociation": "NONE",
        "author": {"login": "a"},
        "milestone": None,
        "labels": {"nodes": [{"name": n} for n in labels or []]},
        "comments": {"totalCount": len(comments or [])},
        "lastComments": {"nodes": comments or []},
        "timelineItems": {"nodes": nodes or []},
        "_scraped_at": "2026-10-19T00:00:00Z",
        "_schema": 2,
    }


def pr_rec(number: int, merged: bool, refs: list[int]) -> dict[str, Any]:
    return {
        "id": f"P_{number}",
        "number": number,
        "title": "pr",
        "body": "",
        "state": "MERGED" if merged else "CLOSED",
        "createdAt": ts(0),
        "closedAt": ts(5),
        "authorAssociation": "MEMBER",
        "author": {"login": "p"},
        "merged": merged,
        "mergedAt": ts(5) if merged else None,
        "labels": {"nodes": []},
        "comments": {"totalCount": 0},
        "changedFiles": 1,
        "additions": 2,
        "deletions": 3,
        "closingIssuesReferences": {
            "nodes": [{"number": i, "repository": {"nameWithOwner": REPO}} for i in refs]
        },
        "_scraped_at": "2026-10-19T01:00:00Z",
    }


def closed_by(pr: int, merged: bool, repo: str | None = REPO) -> dict[str, Any]:
    closer = {"number": pr, "merged": merged, "mergedAt": ts(5) if merged else None}
    if repo:
        closer["repository"] = {"nameWithOwner": repo}
    return {"__typename": "ClosedEvent", "createdAt": ts(6), "actor": None, "closer": closer}


def events(rec: dict[str, Any]) -> list[dict[str, Any]]:
    return qp0d.label_events(rec)


def test_replay_removed_and_readded() -> None:
    rec = issue(
        nodes=[lab(1, "a"), lab(2, "a", False), lab(3, "a"), lab(4, "b"), lab(5, "b", False)]
    )
    ev = events(rec)
    assert qp0d.replay_labels(ev, T0 + timedelta(hours=2.5)) == []
    assert qp0d.replay_labels(ev, T0 + timedelta(hours=3.5)) == ["a"]
    assert qp0d.replay_labels(ev, T0 + timedelta(hours=10)) == ["a"]


def test_label_after_72h_cutoff_excluded_from_t72h() -> None:
    rec = issue(labels=["early", "late"], nodes=[lab(1, "early"), lab(73, "late")])
    row, ev = qp0d.issue_row(rec, REPO)
    assert row["labels_t72h"] == ["early"]
    assert row["labels_final"] == ["early", "late"]
    assert row["first_label_ts"] == T0 + timedelta(hours=1)
    assert len(ev) == 2


def test_replay_orders_by_timestamp_not_list_order() -> None:
    ev = [
        {"ts": T0 + timedelta(hours=2), "action": "unlabeled", "label": "x"},
        {"ts": T0 + timedelta(hours=1), "action": "labeled", "label": "x"},
    ]
    assert qp0d.replay_labels(ev, T0 + timedelta(hours=3)) == []


def test_marked_as_duplicate_event() -> None:
    node = {"__typename": "MarkedAsDuplicateEvent", "createdAt": ts(2), "actor": {"login": "m"}}
    rec = issue(5, nodes=[{**node, "canonical": CANON(9)}])
    [d] = qp0d.extract_duplicates(rec, REPO)
    assert (d["source"], d["canonical"], d["method"]) == (5, 9, "marked_as_duplicate_event")
    unmark = {**node, "__typename": "UnmarkedAsDuplicateEvent", "canonical": CANON(9)}
    rec["timelineItems"]["nodes"].append(unmark)
    assert qp0d.extract_duplicates(rec, REPO) == []


def test_vscode_bot_comment() -> None:
    c = comment("Marked as a duplicate of #77 by the bot", login="vscode-triage-bot")
    [d] = qp0d.extract_duplicates(issue(5, comments=[c]), REPO)
    assert (d["canonical"], d["method"]) == (77, "vscode_bot_comment")
    assert "#77" in d["evidence"] and len(d["evidence"]) <= 200


def test_k8s_close_comment_and_robot_echo() -> None:
    human = comment("duplicate of\r\nhttps://github.com/o/r/issues/12\r\n\r\n/close\r\n")
    [d] = qp0d.extract_duplicates(issue(5, comments=[human]), REPO)
    assert (d["canonical"], d["method"]) == (12, "k8s_close_comment")
    echo = comment(
        "@h: Closing this issue.\n\n<details>\n\nIn response to [this](x):\n\n"
        ">Dup with #13\r\n>/close\n\n\nInstructions",
        login="k8s-ci-robot",
    )
    [d] = qp0d.extract_duplicates(issue(5, comments=[echo]), REPO)
    assert (d["canonical"], d["method"]) == (13, "k8s_close_comment")


def test_cross_repo_url_and_bare_mention_ignored() -> None:
    cross = comment("duplicate of https://github.com/o/other/issues/3\n/close")
    mention = comment("this is a duplicate of #4")  # no /close, no label, not a bot
    assert qp0d.extract_duplicates(issue(5, comments=[cross, mention]), REPO) == []


def test_duplicate_label_with_ref_and_unresolved() -> None:
    rec = issue(
        5,
        labels=["triage/duplicate"],
        comments=[comment("Looks like a dup of #20\n/triage duplicate")],
    )
    [d] = qp0d.extract_duplicates(rec, REPO)
    assert (d["canonical"], d["method"]) == (20, "dup_label_with_ref")
    bare = issue(6, labels=["*duplicate"], comments=[comment("thanks")])
    assert qp0d.extract_duplicates(bare, REPO) == []
    assert qp0d.has_dup_label(bare)


def test_self_reference_excluded_and_best_method_wins() -> None:
    selfref = comment("duplicate of #5\n/close")
    assert qp0d.extract_duplicates(issue(5, comments=[selfref]), REPO) == []
    ev = {"__typename": "MarkedAsDuplicateEvent", "createdAt": ts(2), "actor": None}
    ev["canonical"] = CANON(9)
    both = issue(5, nodes=[ev], comments=[comment("duplicate of #9\n/close")])
    [d] = qp0d.extract_duplicates(both, REPO)
    assert d["method"] == "marked_as_duplicate_event"


def _links(
    rec: dict[str, Any], prs: dict[int, dict[str, Any]], refs: dict[int, list[int]]
) -> pd.DataFrame:
    return pd.DataFrame(
        qp0d.extract_fix_links(rec, REPO, prs, refs),
        columns=["issue", "pr", "method", "pr_merged", "pr_merged_at"],
    )


def test_unmerged_closer_pr_is_not_a_fix() -> None:
    df = _links(issue(1, nodes=[closed_by(30, merged=False)]), {}, {})
    assert df["pr_merged"].tolist() == [False]
    assert qp0d.fix_groups(df).empty
    merged = _links(issue(1, nodes=[closed_by(31, merged=True)]), {}, {})
    assert merged["pr_merged"].tolist() == [True]


def test_pr_table_overrides_closer_flag_and_closing_refs() -> None:
    prs = {40: {"merged": True, "merged_at": T0}}
    df = _links(issue(1, nodes=[closed_by(40, merged=False)]), prs, {1: [40]})
    assert sorted(df["method"]) == ["closed_event_closer_pr", "pr_closing_refs"]
    assert df["pr_merged"].all()


def test_same_pr_group_of_two_issues() -> None:
    a = _links(issue(1, nodes=[closed_by(60, True)]), {}, {})
    b = _links(issue(2, nodes=[closed_by(60, True)]), {}, {})
    g = qp0d.fix_groups(pd.concat([a, b]))
    assert g["pr"].tolist() == [60] and g["issues"].tolist() == [[1, 2]]


def test_text_hash_normalises_case_and_whitespace() -> None:
    assert qp0d.text_hash("Foo  BAR", "a\n b") == qp0d.text_hash("foo bar", "A B")
    assert qp0d.text_hash("foo", "x") != qp0d.text_hash("foo", "y")


def test_censored_open_issue_and_resolution_hours() -> None:
    row, _ = qp0d.issue_row(issue(1, state="OPEN"), REPO)
    assert row["censored"] is True and row["resolution_hours"] is None
    closed, _ = qp0d.issue_row(issue(2, state="CLOSED"), REPO)
    assert closed["censored"] is False and closed["resolution_hours"] == 48


def _write_raw(base: Path, kind: str, recs: list[dict[str, Any]]) -> None:
    d = base / kind / "2024"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "2024-03-01.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in recs)
    (base / kind / "checkpoint.json").write_text("{}")  # must be ignored by the reader


def test_end_to_end_idempotent_rerun(tmp_path: Path) -> None:
    base = tmp_path / "raw" / "o__r"
    recs = [
        issue(
            1,
            labels=["area/x", "area/y"],
            nodes=[lab(1, "area/x"), lab(2, "area/y"), closed_by(30, True)],
        ),
        issue(2, labels=["triage/duplicate"], comments=[comment("duplicate of #1\n/close")]),
        issue(3, state="OPEN"),
    ]
    _write_raw(base, "issues", recs)
    _write_raw(base, "prs", [pr_rec(30, True, [1, 3]), pr_rec(31, True, [3])])
    out = tmp_path / "out"
    for _ in range(2):  # second run overwrites the first without error or duplication
        t = qp0d.build_tables(REPO, base)
        stats = qp0d.write_outputs(t, REPO, base, out)
    assert len(pd.read_parquet(out / "issues.parquet")) == 3
    assert stats["censored_open_issues"] == 1
    assert stats["duplicates"]["by_method"] == {"k8s_close_comment": 1}
    assert stats["multilabel"]["labels_final"] == {"n_ge1": 1, "n_ge2": 1}
    assert stats["fix_links"]["issues_with_merged_fix"] == 2  # issue 1 (closer) and 3 (refs)
    assert pd.read_parquet(out / "fix_groups.parquet")["issues"].map(list).tolist() == [[1, 3]]
    assert (out / "label_frequency.csv").exists() and (out / "priority_signals.csv").exists()
    assert not list(tmp_path.glob("raw/**/*.tmp"))
    assert qp0d.stats_from_parquet(REPO, base, out)["repo"] == REPO  # --stats-only path


def _closed(*comments: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return issue(5, comments=list(comments), **kw)


def test_closer_must_be_same_repo_schema2() -> None:
    nodes = [closed_by(1, True, "x/y"), closed_by(2, True, None), closed_by(3, True)]
    rec = issue(9, nodes=nodes)
    row, _ = qp0d.issue_row(rec, REPO)
    assert row["closed_by_prs"] == [3]
    assert (row["closer_foreign_repo"], row["closer_repo_unknown"]) == (1, 1)
    assert _links(rec, {}, {})["pr"].tolist() == [3]
    v1 = {**rec}
    del v1["_schema"]  # schema-1 rows: closer repo cannot be trusted at all
    row1, _ = qp0d.issue_row(v1, REPO)
    assert row1["closed_by_prs"] == [] and row1["closer_repo_unknown"] == 3
    assert _links(v1, {}, {}).empty


def test_foreign_or_unknown_canonical_dropped_and_counted() -> None:
    mk = {"__typename": "MarkedAsDuplicateEvent", "createdAt": ts(2), "actor": None}
    nodes = [{**mk, "canonical": CANON(7, "x/y")}, {**mk, "canonical": CANON(8, None)}]
    diag: Counter[str] = Counter()
    assert qp0d.extract_duplicates(issue(5, nodes=nodes), REPO, diag) == []
    assert (diag["canon_foreign_repo"], diag["canon_repo_unknown"]) == (1, 1)


def test_xref_events_make_no_fix_links() -> None:
    # willCloseTarget is cleared by GitHub on merge (1 of 375 xrefs true, 0 merged in schema-2 data)
    src = {"number": 50, "merged": True, "repository": {"nameWithOwner": REPO}}
    x = {"__typename": "CrossReferencedEvent", "willCloseTarget": True, "source": src}
    assert _links(issue(1, nodes=[{**x, "createdAt": ts(5), "actor": None}]), {}, {}).empty


def _dups(body: str, **kw: Any) -> list[dict[str, Any]]:
    return qp0d.extract_duplicates(_closed(comment(body), **kw), REPO)


def test_false_positive_probes_rejected() -> None:
    assert _dups("This is not a duplicate of #123.\n/close") == []
    assert _dups("/close\nNot a dup of #5, but related to #6") == []
    assert _dups("> duplicate of #77 from someone else\n\nno, different. /close") == []
    assert _dups("/close\nthe duplicate issue #44 was fixed in PR") == []
    assert _dups("It isn" + "'" + "t a duplicate of #8\n/close") == []


def test_target_taken_from_same_line_and_n_targets() -> None:
    [d] = _dups("Fixed by #99 earlier.\nDuplicate of #12 and #13\n/close")
    assert (d["canonical"], d["n_targets"]) == (12, 2)


def test_ref_forms_and_slash_duplicate() -> None:
    assert _dups("/duplicate 12\n/close")[0]["canonical"] == 12
    assert _dups("duplicate of o/r#14\n/close")[0]["canonical"] == 14
    assert _dups("dup of other/repo#14\n/close") == []


def test_comment_pairs_require_closed_issue() -> None:
    assert _dups("duplicate of #12\n/close", state="OPEN") == []


def test_state_reason_duplicate_counted_unresolved() -> None:
    rec = issue(5)
    rec["stateReason"] = "DUPLICATE"
    row, _ = qp0d.issue_row(rec, REPO)
    assert row["closed_as_duplicate"] is True
    assert qp0d.extract_duplicates(rec, REPO) == []


def _tables(tmp_path: Path, recs: list[dict[str, Any]]) -> tuple[Path, dict[str, pd.DataFrame]]:
    base = tmp_path / "raw" / "o__r"
    _write_raw(base, "issues", recs)
    _write_raw(base, "prs", [])
    return base, qp0d.build_tables(REPO, base)


def test_canonical_newer_and_in_table(tmp_path: Path) -> None:
    older = issue(1)
    newer = issue(3)
    newer["createdAt"] = ts(10)
    mk = {"__typename": "MarkedAsDuplicateEvent", "createdAt": ts(2), "actor": None}
    src = issue(2, nodes=[{**mk, "canonical": CANON(3)}, {**mk, "canonical": CANON(404)}])
    _, t = _tables(tmp_path, [older, src, newer])
    d = t["duplicate_pairs"].set_index("canonical")
    assert bool(d.loc[3, "canonical_newer"]) is True and bool(d.loc[3, "canonical_in_table"])
    assert pd.isna(d.loc[404, "canonical_newer"]) and not d.loc[404, "canonical_in_table"]


def test_stats_with_empty_links_and_dups(tmp_path: Path) -> None:
    base, t = _tables(tmp_path, [issue(1, state="OPEN")])
    stats = qp0d.write_outputs(t, REPO, base, tmp_path / "out")
    assert stats["duplicates"]["n_pairs"] == 0 and stats["fix_links"]["issues_with_merged_fix"] == 0
    assert qp0d.stats_from_parquet(REPO, base, tmp_path / "out")["duplicates"]["n_pairs"] == 0


def test_malformed_input_names_file_line_and_issue(tmp_path: Path) -> None:
    base = tmp_path / "raw" / "o__r"
    good = issue(1)
    bad = issue(2)
    bad["timelineItems"] = None
    _write_raw(base, "issues", [good, bad])
    with pytest.raises(qp0d.RawDataError, match=r"2024-03-01\.jsonl\.gz:2 .*issue #2.*timeline"):
        qp0d.build_tables(REPO, base)
    del bad["timelineItems"]
    _write_raw(base, "issues", [good, bad])
    with pytest.raises(qp0d.RawDataError, match="issue #2"):
        qp0d.build_tables(REPO, base)
    bad["timelineItems"] = {"nodes": [{"createdAt": ts(1)}]}
    _write_raw(base, "issues", [good, bad])
    with pytest.raises(qp0d.RawDataError, match="__typename"):
        qp0d.build_tables(REPO, base)
    with gzip.open(base / "issues" / "2024" / "2024-03-01.jsonl.gz", "wt") as f:
        f.write(json.dumps(good) + "\n{not json\n")
    with pytest.raises(qp0d.RawDataError, match=r"2024-03-01\.jsonl\.gz:2"):
        qp0d.build_tables(REPO, base)


def test_unknown_typename_ignored_and_closed_without_closed_at(tmp_path: Path) -> None:
    rec = issue(1, nodes=[{"__typename": "SomethingNew", "createdAt": ts(1)}])
    rec["closedAt"] = None
    row, _ = qp0d.issue_row(rec, REPO)
    assert row["resolution_hours"] is None and row["anomaly_closed_no_closed_at"] is True
    base, t = _tables(tmp_path, [rec])
    stats = qp0d.write_outputs(t, REPO, base, tmp_path / "out")
    assert stats["anomalies"] == {"closed_without_closed_at": 1}


def test_stats_deterministic_apart_from_volatile(tmp_path: Path) -> None:
    base, t = _tables(tmp_path, [issue(1, labels=["a"]), issue(2, state="OPEN")])
    s1 = qp0d.write_outputs(t, REPO, base, tmp_path / "o1")
    s2 = qp0d.write_outputs(t, REPO, base, tmp_path / "o2")
    s1.pop("volatile")  # storage size varies with the out dir contents
    s2.pop("volatile")
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    assert "scraped_at" in t["issues"].columns
