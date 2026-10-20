from __future__ import annotations

import gzip
import importlib.util
import json
import sys
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


def test_xref_events_make_no_fix_links() -> None:
    # willCloseTarget is cleared by GitHub on merge (1 of 375 xrefs true, 0 merged in schema-2 data)
    src = {"number": 50, "merged": True, "repository": {"nameWithOwner": REPO}}
    x = {"__typename": "CrossReferencedEvent", "willCloseTarget": True, "source": src}
    assert _links(issue(1, nodes=[{**x, "createdAt": ts(5), "actor": None}]), {}, {}).empty


def _dups(body: str, **kw: Any) -> list[dict[str, Any]]:
    return qp0d.extract_duplicates(_closed(comment(body), **kw), REPO)


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


HEDGED = [
    "possible duplicate of #12 but not sure\n/close",
    "this may be a dup of #3\n/close",
    "Is this a duplicate of #8? I doubt it.\n/close",
    "If this is a duplicate of #10 please close\n/close",
    "Can you check whether this is a duplicate of #15?\n/close",
    "Someone said this was a duplicate of #30 but it is not\n/close",
    "would be a duplicate of #32 unless the fix landed\n/close",
    "/close\nI think it is a duplicate of #33",
    "duplicate of #34, nope\n/close",
]


def _echo(quoted: str) -> dict[str, Any]:
    body = "@h: Closing this issue.\n\n<details>\n\nIn response to [this](x):\n\n" + quoted
    return comment(body + "\n\n\nInstructions for interacting with me", login="k8s-ci-robot")


def test_label_frequency_with_multi_label_issues() -> None:
    # exploding lists repeats the issue index; pandas 3 crosstab refuses duplicate-label reindexing
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from qp0_derive_stats import label_frequency

    iss = pd.DataFrame(
        {
            "created_year": [2023, 2024, 2024],
            "labels_final": [["a", "b"], ["a"], ["a", "b", "c"]],
            "labels_t72h": [["a"], [], ["a", "b"]],
        }
    )
    f = label_frequency(iss).set_index("label")
    assert f["n_issues_final"].to_dict() == {"a": 3, "b": 2, "c": 1}
    assert f["n_issues_2023"].to_dict() == {"a": 1, "b": 1, "c": 0}
    assert f["n_issues_2024"].to_dict() == {"a": 2, "b": 1, "c": 1}
    assert f["n_issues_t72h"].to_dict() == {"a": 2, "b": 1, "c": 0}
    assert f["first_seen_year"].to_dict() == {"a": 2023, "b": 2023, "c": 2024}


def _write_dup_targets(base: Path, rows: list[dict[str, Any]]) -> None:
    d = base / "dup_targets"
    d.mkdir(parents=True, exist_ok=True)
    with gzip.open(d / "00000.jsonl.gz", "wt", encoding="utf-8") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    (d / "checkpoint.json").write_text("{}")  # must not be read as data


def test_dup_targets_add_pairs_resolve_direction_and_flags(tmp_path: Path) -> None:
    dup_labelled, plain = issue(5, labels=["*duplicate"]), issue(9)
    ui_closed = issue(7, labels=["*duplicate"])
    ui_closed["stateReason"] = "DUPLICATE"
    base = tmp_path / "raw" / "o__r"
    _write_raw(base, "issues", [dup_labelled, plain, ui_closed, issue(3)])
    _write_raw(base, "prs", [])
    marked = {
        "__typename": "MarkedAsDuplicateEvent", "createdAt": ts(3), "actor": {"login": "m"},
        "canonical": CANON(5), "duplicate": CANON(9),
    }  # fmt: skip
    closed = {
        "__typename": "ClosedEvent", "createdAt": ts(4), "stateReason": "DUPLICATE",
        "duplicateOf": CANON(3),
    }  # fmt: skip
    _write_dup_targets(
        base,
        [
            {"number": 5, "timelineItems": {"totalCount": 1, "nodes": [marked]}, "_schema": 2},
            {"number": 7, "timelineItems": {"totalCount": 1, "nodes": [closed]}, "_schema": 2},
            {"number": 9, "missing": True, "_schema": 2},
        ],
    )
    t = qp0d.build_tables(REPO, base)
    pairs = {(r.source, r.canonical): r.method for r in t["duplicate_pairs"].itertuples()}
    assert pairs == {(5, 9): "marked_as_duplicate_event", (7, 3): "closed_event_duplicate_of"}
    iss = t["issues"].set_index("number")
    assert not iss.at[5, "dup_label_unresolved"] and not iss.at[7, "dup_label_unresolved"]
    assert not iss.at[7, "closed_as_duplicate_unresolved"]


def test_dup_targets_pair_from_other_side_clears_that_issues_flag(tmp_path: Path) -> None:
    # the event sits on X's (5) timeline but Y (9) is the closed duplicate: the pair is 9 -> 5
    base = tmp_path / "raw" / "o__r"
    _write_raw(base, "issues", [issue(5), issue(9, labels=["*duplicate"])])
    _write_raw(base, "prs", [])
    marked = {
        "__typename": "MarkedAsDuplicateEvent", "createdAt": ts(3), "actor": {"login": "m"},
        "canonical": CANON(5), "duplicate": CANON(9),
    }  # fmt: skip
    _write_dup_targets(
        base, [{"number": 5, "timelineItems": {"totalCount": 1, "nodes": [marked]}, "_schema": 2}]
    )
    t = qp0d.build_tables(REPO, base)
    assert [(r.source, r.canonical) for r in t["duplicate_pairs"].itertuples()] == [(9, 5)]
    assert not t["issues"].set_index("number").at[9, "dup_label_unresolved"]


# --- labels present from creation (templates / forms apply them without any LabeledEvent) -------
def test_label_without_events_is_present_from_creation() -> None:
    rec = issue(labels=["template", "later"], nodes=[lab(80, "later")])
    row, _ = qp0d.issue_row(rec, REPO)
    assert row["labels_t72h"] == [
        "template"
    ]  # never touched -> there since creation; "later" came after 72 h
    assert row["labels_final"] == ["template", "later"]


def test_creation_label_removed_and_readded_is_present_until_the_first_unlabel() -> None:
    ev = events(issue(nodes=[lab(10, "x", False), lab(100, "x")]))
    init = qp0d.creation_labels(ev, ["x"])
    assert init == {"x"}  # its first event is an unlabel, so it existed before that
    assert qp0d.replay_labels(ev, T0 + timedelta(hours=5), init) == ["x"]
    assert qp0d.replay_labels(ev, T0 + timedelta(hours=50), init) == []  # removed at 10 h
    assert qp0d.replay_labels(ev, T0 + timedelta(hours=101), init) == ["x"]  # re-added at 100 h


def test_label_with_only_an_unlabel_event_was_there_until_removed() -> None:
    ev = events(issue(nodes=[lab(10, "gone", False)]))
    init = qp0d.creation_labels(ev, [])  # not in the final set
    assert init == {"gone"}
    assert qp0d.replay_labels(ev, T0 + timedelta(hours=9), init) == ["gone"]
    assert qp0d.replay_labels(ev, T0 + timedelta(hours=11), init) == []
    row, _ = qp0d.issue_row(issue(nodes=[lab(10, "gone", False)]), REPO)
    assert row["labels_t72h"] == [] and row["labels_final"] == []


def test_label_first_labeled_then_removed_is_not_a_creation_label() -> None:
    ev = events(issue(nodes=[lab(1, "a"), lab(2, "a", False)]))
    assert qp0d.creation_labels(ev, []) == set()
