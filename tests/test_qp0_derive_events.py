from __future__ import annotations

import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))  # scripts/ is not a package
import qp0_derive_events as qp0d  # noqa: E402

REPO = "o/r"


def CANON(n: int, repo: str | None = "o/r") -> dict[str, Any]:  # noqa: N802
    return {"number": n, **({"repository": {"nameWithOwner": repo}} if repo else {})}


T0 = datetime(2024, 3, 1, 12, 0, tzinfo=UTC)


def ts(hours: float) -> str:
    return (T0 + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


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


def _closed(*comments: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return issue(5, comments=list(comments), **kw)


def _dups(body: str, **kw: Any) -> list[dict[str, Any]]:
    return qp0d.extract_duplicates(_closed(comment(body), **kw), REPO)


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


def test_foreign_or_unknown_canonical_dropped_and_counted() -> None:
    mk = {"__typename": "MarkedAsDuplicateEvent", "createdAt": ts(2), "actor": None}
    nodes = [{**mk, "canonical": CANON(7, "x/y")}, {**mk, "canonical": CANON(8, None)}]
    diag: Counter[str] = Counter()
    assert qp0d.extract_duplicates(issue(5, nodes=nodes), REPO, diag) == []
    assert (diag["canon_foreign_repo"], diag["canon_repo_unknown"]) == (1, 1)


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


def test_hedged_question_and_conditional_phrasings_rejected() -> None:
    for body in HEDGED:
        assert _dups(body) == [], body


def test_true_positive_forms_keep_recall() -> None:
    cases = {
        "Duplicate of #12 /close": 12,
        "Closing as duplicate of #16": 16,
        "dup of o/r#88\n/close": 88,
        "duplicate of [#9](https://github.com/o/r/issues/9)\n/close": 9,
        "/duplicate #19\n/close": 19,
        "Looks good.\nDuplicate of #12\n/close\nThanks, it would help to retest.": 12,
    }
    for body, want in cases.items():
        got = _dups(body)
        assert [d["canonical"] for d in got] == [want], body
    bot = comment("Closed this one as a duplicate of #21", login="vscode-triage-bot")
    assert qp0d.extract_duplicates(_closed(bot), REPO)[0]["canonical"] == 21


def test_robot_echo_nested_quotes_and_missing_close_ignored() -> None:
    assert (
        qp0d.extract_duplicates(_closed(_echo(">/close\n>> bob said: duplicate of #50")), REPO)
        == []
    )
    hedged = ">/close\n>duplicate of #51 (bob" + "'" + "s suggestion, I disagree)"
    assert qp0d.extract_duplicates(_closed(_echo(hedged)), REPO) == []
    # the closer's own /close line must be inside the echoed block
    assert qp0d.extract_duplicates(_closed(_echo(">duplicate of #52")), REPO) == []
    [d] = qp0d.extract_duplicates(_closed(_echo(">Dup of #53\n>/close")), REPO)
    assert d["canonical"] == 53
    labeled = _closed(_echo(">duplicate of #54"), labels=["triage/duplicate"])
    assert qp0d.extract_duplicates(labeled, REPO) == []


def test_label_pair_needs_unhedged_unquoted_comment() -> None:
    guess = comment("/remove-lifecycle stale\n>duplicate of #60 was my guess")
    assert qp0d.extract_duplicates(_closed(guess, labels=["triage/duplicate"]), REPO) == []
    maybe = comment("maybe a dup of #61 /triage duplicate")
    assert qp0d.extract_duplicates(_closed(maybe, labels=["triage/duplicate"]), REPO) == []


# --- vscode wording (the extractor was first built on kubernetes only) ---------------------------
def _staff(body: str, login: str = "maint") -> dict[str, Any]:
    return {**comment(body, login), "authorAssociation": "MEMBER"}


def test_slash_duplicate_command_forms_seen_in_vscode() -> None:
    for body, want in [
        ("/duplicate of https://github.com/o/r/issues/12", 12),
        ("/duplicate\r\n\r\nUse Profiles feature - https://github.com/o/r/issues/116", 116),
        ("\duplicate #170073 ", 170073),
        ("/duplicate 44", 44),
    ]:
        got = qp0d.extract_duplicates(_closed(_staff(body)), REPO)
        assert [(p["canonical"], p["method"]) for p in got] == [(want, "slash_duplicate_cmd")], body


def test_slash_duplicate_needs_staff_and_same_repo() -> None:
    assert _dups("/duplicate https://github.com/o/r/issues/12") == []  # not staff, no label
    got = qp0d.extract_duplicates(
        _closed(_staff("/duplicate https://github.com/x/y/issues/9")), REPO
    )
    assert got == []  # a foreign repo is not a usable pair
    labelled = _dups("/duplicate #12", labels=["*duplicate"])
    assert [p["method"] for p in labelled] == ["dup_label_with_ref"]  # unchanged k8s-era path


def test_dupe_wording_is_a_duplicate_phrase() -> None:
    body = _staff("dupe of https://github.com/o/r/issues/7")
    got = qp0d.extract_duplicates(issue(5, labels=["*duplicate"], comments=[body]), REPO)
    assert [p["canonical"] for p in got] == [7]


def test_close_as_duplicate_prose_needs_staff_and_exactly_one_ref() -> None:
    one = "Tracked here - https://github.com/o/r/issues/597. I will go ahead and close this issue as a duplicate."
    got = qp0d.extract_duplicates(_closed(_staff(one)), REPO)
    assert [(p["canonical"], p["method"]) for p in got] == [(597, "close_as_duplicate_prose")]
    two = one + " See also #12."
    assert qp0d.extract_duplicates(_closed(_staff(two)), REPO) == []
    assert qp0d.extract_duplicates(_closed(comment(one)), REPO) == []  # not staff
    other = "Tracked in #597, closing."
    assert qp0d.extract_duplicates(_closed(_staff(other)), REPO) == []  # no duplicate wording


def test_self_canonical_marked_event_is_counted_not_paired() -> None:
    # vscode: every MarkedAsDuplicateEvent has canonical == the issue itself (the target is in the
    # `duplicate` field, which the scrape did not request); it must be visible, not silent
    node = {
        "__typename": "MarkedAsDuplicateEvent",
        "createdAt": ts(2),
        "actor": {"login": "m"},
        "canonical": CANON(5),
    }
    diag: Counter[str] = Counter()
    assert qp0d.extract_duplicates(issue(5, nodes=[node]), REPO, diag) == []
    assert diag["canon_self"] == 1


def test_closed_event_duplicate_of_target_is_a_pair() -> None:
    # GraphQL ClosedEvent.duplicateOf carries the target of a UI "close as duplicate" (not yet scraped)
    closed = {
        "__typename": "ClosedEvent",
        "createdAt": ts(5),
        "actor": {"login": "m"},
        "stateReason": "DUPLICATE",
        "duplicateOf": CANON(9),
    }
    got = qp0d.extract_duplicates(issue(5, nodes=[closed]), REPO)
    assert [(p["canonical"], p["method"]) for p in got] == [(9, "closed_event_duplicate_of")]
    foreign = {**closed, "duplicateOf": CANON(9, "x/y")}
    assert qp0d.extract_duplicates(issue(5, nodes=[foreign]), REPO) == []
    assert qp0d.extract_duplicates(issue(5, nodes=[{**closed, "duplicateOf": None}]), REPO) == []


def test_staff_plain_duplicate_comment_without_of() -> None:
    for body in ("Duplicate https://github.com/o/r/issues/164", "Dup: #164", "duplicate\r\n#164"):
        got = qp0d.extract_duplicates(_closed(_staff(body)), REPO)
        assert [(p["canonical"], p["method"]) for p in got] == [(164, "slash_duplicate_cmd")], body
    assert (
        qp0d.extract_duplicates(
            _closed(comment("Duplicate https://github.com/o/r/issues/164")), REPO
        )
        == []
    )
    assert qp0d.extract_duplicates(_closed(_staff("This is not a duplicate. See #164")), REPO) == []


# --- review round: hedge / negation, adjacency, human-only, PR targets --------------------------
def _staff_pairs(body: str, **kw: Any) -> list[tuple[int, str, int]]:
    got = qp0d.extract_duplicates(_closed(_staff(body)), REPO, **kw)
    return [(p["canonical"], p["method"], p["n_targets"]) for p in got]


def test_command_paths_reject_hedged_and_negated_wording() -> None:
    for body in [
        "not a duplicate: /duplicate #5",
        "this is not a /duplicate #12",
        "could this be a /duplicate #9?",
        "/duplicate #26 but it is not",
        "/duplicate #27 is wrong, it is not a dup",
        "Duplicate #28 maybe",
        "is this a Duplicate #29",
    ]:
        assert _staff_pairs(body) == [], body


def test_real_staff_command_survives_unrelated_negation() -> None:
    # the "dont" belongs to another clause, not to the target sentence
    assert _staff_pairs("pls dont open duplicate issues /duplicate #212322") == [
        (212322, "slash_duplicate_cmd", 1)
    ]
    assert _staff_pairs("please don't open duplicates.\n/duplicate #3") == [
        (3, "slash_duplicate_cmd", 1)
    ]


def test_command_target_must_be_adjacent_and_unambiguous() -> None:
    for body in [
        "/duplicate fixed in #5, canonical is #9",
        "/duplicate\nThanks for the report, this is similar to #44 maybe",
        "/duplicate\n2 of the above",
        "/duplicate-check #31",
        "/duplicate one two three four five #8",
    ]:
        assert _staff_pairs(body) == [], body
    assert _staff_pairs("Duplicate #34 and #35") == [(34, "slash_duplicate_cmd", 2)]
    assert _staff_pairs("/duplicate #1 & #2") == [(1, "slash_duplicate_cmd", 2)]
    assert _staff_pairs("/duplicate https://github.com/o/r/issues/7 Related: #8") == [
        (7, "slash_duplicate_cmd", 1)
    ]
    assert _staff_pairs("/duplicate\r\n\r\nUse Profiles feature - #116") == [
        (116, "slash_duplicate_cmd", 1)
    ]


def test_staff_gated_methods_need_a_human_user() -> None:
    bot = {**_staff("/duplicate #12", login="some-bot"), "author": {"login": "some-bot"}}
    assert "slash_duplicate_cmd" not in {
        p["method"] for p in qp0d.extract_duplicates(_closed(bot), REPO)
    }
    app = {**_staff("/duplicate #12"), "author": {"login": "octo", "__typename": "Bot"}}
    assert qp0d.extract_duplicates(_closed(app), REPO) == []
    suffix = {
        **_staff("I will close this as a duplicate, tracked in #12"),
        "author": {"login": "dependabot[bot]"},
    }
    assert qp0d.extract_duplicates(_closed(suffix), REPO) == []
    user = {**_staff("/duplicate #12"), "author": {"login": "octo", "__typename": "User"}}
    assert len(qp0d.extract_duplicates(_closed(user), REPO)) == 1


def test_prose_rejects_pull_requests() -> None:
    body = "The fix is in #77, I will close this as a duplicate."
    assert _staff_pairs(body) == [(77, "close_as_duplicate_prose", 1)]  # unknown to be a PR
    assert _staff_pairs(body, pr_numbers={77}) == []  # known PR number
    url = "Fixed by https://github.com/o/r/pull/77 and #9, I will close this as a duplicate."
    assert _staff_pairs(url) == []  # a /pull/ URL is a second, non-issue reference
    assert _staff_pairs("closing as a duplicate of https://github.com/o/r/pull/77") == []


def test_command_accepts_markdown_link_query_urls_and_trailing_explanation() -> None:
    cmd = "slash_duplicate_cmd"
    assert _staff_pairs("/duplicate [#194841](https://github.com/o/r/issues/194841)") == [
        (194841, cmd, 1)
    ]
    assert _staff_pairs(
        "/duplicate https://github.com/o/r/issues/246652?dupRef=o%2Fr%2F229301"
    ) == [(246652, cmd, 1)]
    assert _staff_pairs(
        "/duplicate https://github.com/o/r/issues/122951 , if you need admin mode see there"
    ) == [(122951, cmd, 1)]


# --- dup_targets re-scrape: MarkedAsDuplicateEvent.duplicate / ClosedEvent.duplicateOf ----------
def _dt(number: int, nodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "number": number,
        "timelineItems": {"totalCount": len(nodes), "nodes": nodes},
        "_schema": 2,
    }


def _marked(self_n: int, other: int, repo: str = "o/r") -> dict[str, Any]:
    return {
        "__typename": "MarkedAsDuplicateEvent",
        "createdAt": ts(3),
        "actor": {"login": "m"},
        "canonical": CANON(self_n),
        "duplicate": CANON(other, repo),
    }


def _dupish(number: int, dup: bool) -> dict[str, Any]:
    return issue(number, labels=["*duplicate"] if dup else [])


def _tp(dt: dict[str, Any], recs: list[dict[str, Any]], diag: Counter[str] | None = None):  # noqa: ANN202
    lookup = {r["number"]: r for r in recs}
    return qp0d.target_pairs(dt, lookup, REPO, diag if diag is not None else Counter())


def test_marked_event_direction_follows_the_closed_duplicate_side() -> None:
    nodes = [_marked(5, 9)]
    # X (5) carries the dup label, Y (9) does not: source = 5, canonical = 9
    got = _tp(_dt(5, nodes), [_dupish(5, True), _dupish(9, False)])
    assert [(p["source"], p["canonical"], p["method"]) for p in got] == [
        (5, 9, "marked_as_duplicate_event")
    ]
    # Y is the duplicate side: the pair is Y -> X even though the event sits on X's timeline
    got = _tp(_dt(5, nodes), [_dupish(5, False), _dupish(9, True)])
    assert [(p["source"], p["canonical"]) for p in got] == [(9, 5)]
    # Y outside the table (older than the window): X is the source
    assert [(p["source"], p["canonical"]) for p in _tp(_dt(5, nodes), [_dupish(5, True)])] == [
        (5, 9)
    ]


def test_marked_event_ambiguous_direction_is_skipped_and_counted() -> None:
    for x_dup, y_dup in ((True, True), (False, False)):
        diag: Counter[str] = Counter()
        assert _tp(_dt(5, [_marked(5, 9)]), [_dupish(5, x_dup), _dupish(9, y_dup)], diag) == []
        assert diag["dup_target_ambiguous"] == 1


def test_marked_event_foreign_repo_and_unmark_retraction() -> None:
    assert _tp(_dt(5, [_marked(5, 9, "x/y")]), [_dupish(5, True)]) == []
    unmark = {
        "__typename": "UnmarkedAsDuplicateEvent",
        "createdAt": ts(4),
        "canonical": CANON(5),
        "duplicate": CANON(9),
    }
    assert _tp(_dt(5, [_marked(5, 9), unmark]), [_dupish(5, True)]) == []


def test_closed_event_duplicate_of_in_dup_targets_and_missing_rows() -> None:
    closed = {
        "__typename": "ClosedEvent",
        "createdAt": ts(5),
        "stateReason": "DUPLICATE",
        "duplicateOf": CANON(7),
    }
    got = _tp(_dt(5, [closed]), [_dupish(5, True)])
    assert [(p["source"], p["canonical"], p["method"]) for p in got] == [
        (5, 7, "closed_event_duplicate_of")
    ]
    assert _tp(_dt(5, [{**closed, "duplicateOf": None}]), [_dupish(5, True)]) == []  # slash closure
    assert _tp({"number": 5, "missing": True, "_schema": 2}, []) == []
    assert (
        qp0d.METHOD_RANK["closed_event_duplicate_of"]
        < qp0d.METHOD_RANK["marked_as_duplicate_event"]
    )
    assert qp0d.METHOD_RANK["marked_as_duplicate_event"] < qp0d.METHOD_RANK["slash_duplicate_cmd"]
