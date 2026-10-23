"""Phase 3b flag 2 plumbing: evidence rendering caps + artifact loader. Network-free."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from triage_iq.models.resolution_evidence import (
    EVIDENCE_BLOCK_MAX_CHARS,
    EVIDENCE_HEADER,
    EVIDENCE_ITEM_MAX_CHARS,
    EvidenceRecord,
    ResolutionEvidence,
    render_item,
)

LONG = "word " * 400


OLD = "2020-01-01T00:00:00Z"
AS_OF = pd.Timestamp("2030-01-01", tz="UTC")  # after every OLD-dated item


def _rec(prs=(), comment: str = "") -> EvidenceRecord:  # noqa: ANN001
    """Record whose items are all dated OLD (prs given as (number, title, excerpt))."""
    return EvidenceRecord(tuple((*p, OLD) for p in prs), comment, OLD if comment else "")


def test_item_without_evidence_is_empty() -> None:
    assert render_item(5, _rec()) == ""


def test_fix_item_names_pr_title_and_excerpt() -> None:
    line = render_item(7, _rec([(900, "Fix kubelet crash", "Handles nil pod in sync loop")]))
    assert line == '#7: fix PR 900 "Fix kubelet crash": Handles nil pod in sync loop'


def test_comment_only_item() -> None:
    line = render_item(7, _rec(comment="Fixed by bumping etcd."))
    assert line == '#7: closing comment "Fixed by bumping etcd."'


def test_item_with_both_stays_within_cap_and_keeps_both_parts() -> None:
    line = render_item(1234, _rec([(99, "T" * 300, LONG)], LONG))
    assert len(line) <= EVIDENCE_ITEM_MAX_CHARS
    assert "fix PR 99" in line and "closing comment" in line


def test_hash_references_and_quotes_are_neutralised() -> None:
    line = render_item(7, _rec([(9, 'Fixes #123 "x"', "see #456 and #78")], "dup of #999"))
    assert line.count("#") == 1  # only the neighbour's own "#7"
    assert '"x"' not in line  # inner double quotes replaced


def test_block_is_hard_capped_and_header_first() -> None:
    ev = ResolutionEvidence({n: _rec([(n, "T" * 300, LONG)], LONG) for n in (1, 2, 3, 4)})
    block, used = ev.block_for([{"number": n} for n in (1, 2, 3, 4)], AS_OF)
    assert len(block) <= EVIDENCE_BLOCK_MAX_CHARS
    assert block.startswith(EVIDENCE_HEADER)
    assert used == [1, 2, 3]  # only the top-3 neighbours are considered
    assert block.count("\n#") == 3  # no line was cut off by the global cap


def test_block_skips_neighbours_without_evidence_and_is_empty_when_none() -> None:
    ev = ResolutionEvidence({2: _rec(comment="done")})
    block, used = ev.block_for([{"number": 1}, {"number": 2}, {"number": 3}], AS_OF)
    assert used == [2] and "#1" not in block
    assert ev.block_for([{"number": 1}, {"number": 3}], AS_OF) == ("", [])
    assert ev.block_for([], AS_OF) == ("", [])


def test_load_keeps_only_rows_with_evidence(tmp_path: Path) -> None:
    p = tmp_path / "e.parquet"
    pd.DataFrame(
        [
            {
                "number": 1,
                "fix_prs": [
                    {"number": 10, "title": "Fix A", "body_excerpt": "x", "merged_at": OLD}
                ],
                "closing_comment": "",
                "closing_comment_at": "",
                "has_evidence": True,
            },
            {
                "number": 2,
                "fix_prs": [],
                "closing_comment": "done",
                "closing_comment_at": OLD,
                "has_evidence": True,
            },
            {
                "number": 3,
                "fix_prs": [],
                "closing_comment": "",
                "closing_comment_at": "",
                "has_evidence": False,
            },
        ]
    ).to_parquet(p, index=False)
    ev = ResolutionEvidence.load(p)
    assert sorted(ev.records) == [1, 2]
    assert ev.dated
    assert ev.records[1].fix_prs == ((10, "Fix A", "x", OLD),)
    assert (ev.records[2].closing_comment, ev.records[2].closing_comment_at) == ("done", OLD)


def _v1_frame() -> pd.DataFrame:
    """The v1 artifact shape: no merged_at / closing_comment_at."""
    return pd.DataFrame(
        [
            {
                "number": 2,
                "fix_prs": [{"number": 10, "title": "Fix A", "body_excerpt": "x"}],
                "closing_comment": "done",
                "has_evidence": True,
            }
        ]
    )


def test_undated_artifact_fails_closed_in_block_for(tmp_path: Path) -> None:
    p = tmp_path / "v1.parquet"
    _v1_frame().to_parquet(p, index=False)
    ev = ResolutionEvidence.load(p)
    assert not ev.dated
    with pytest.raises(ValueError, match="no item dates"):
        ev.block_for([{"number": 2}], AS_OF)


# --- as-of filter: strictly before -----------------------------------------------------------

T = pd.Timestamp("2024-05-01T12:00:00Z")
SEC = pd.Timedelta(seconds=1)


def _dated(merged: pd.Timestamp | None, commented: pd.Timestamp | None) -> EvidenceRecord:
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return EvidenceRecord(
        ((9, "Fix it", "exc", merged.strftime(fmt) if merged is not None else ""),),
        "closed by upgrade",
        commented.strftime(fmt) if commented is not None else "",
    )


def test_fix_pr_dated_one_second_before_is_kept() -> None:
    rec = _dated(T - SEC, None).before(T)
    assert [p[0] for p in rec.fix_prs] == [9]


def test_fix_pr_dated_exactly_at_as_of_is_dropped() -> None:
    assert _dated(T, None).before(T).fix_prs == ()


def test_fix_pr_dated_one_second_after_is_dropped() -> None:
    assert _dated(T + SEC, None).before(T).fix_prs == ()


def test_comment_boundaries_are_strict_too() -> None:
    assert _dated(None, T - SEC).before(T).closing_comment == "closed by upgrade"
    assert _dated(None, T).before(T).closing_comment == ""
    assert _dated(None, T + SEC).before(T).closing_comment == ""


def test_undated_item_in_a_dated_artifact_is_dropped() -> None:
    rec = EvidenceRecord(((9, "Fix it", "exc", ""),), "c", "")
    assert not rec.before(T).has_evidence


def test_naive_as_of_is_taken_as_utc() -> None:
    assert _dated(T - SEC, None).before(T.tz_localize(None)).fix_prs != ()
    assert _dated(T, None).before(T.tz_localize(None)).fix_prs == ()


def test_block_keeps_early_item_and_neighbour_with_nothing_left_contributes_nothing() -> None:
    ev = ResolutionEvidence(
        {
            1: EvidenceRecord(((11, "Early fix", "", (T - SEC).strftime("%Y-%m-%dT%H:%M:%SZ")),)),
            2: EvidenceRecord(((12, "Late fix", "", (T + SEC).strftime("%Y-%m-%dT%H:%M:%SZ")),)),
        }
    )
    block, used = ev.block_for([{"number": 1}, {"number": 2}], T)
    assert used == [1] and "Early fix" in block and "Late fix" not in block
    assert ev.block_for([{"number": 2}], T) == ("", [])
