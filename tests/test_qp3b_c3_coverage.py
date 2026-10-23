from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp3b_c3_coverage as c3  # noqa: E402


def rec(fix: bool, comment: bool) -> dict[str, Any]:
    return {
        "fix_prs": [{"title": "tttt", "body_excerpt": "bb"}] if fix else [],
        "closing_comment": "cccccc" if comment else "",
        "has_fix_evidence": fix,
        "has_comment_evidence": comment,
        "has_evidence": fix or comment,
    }


def test_evidence_chars() -> None:
    assert c3.evidence_chars(rec(True, True)) == 12
    assert c3.evidence_chars(rec(False, False)) == 0


def test_coverage_shares_and_missing_neighbours() -> None:
    rows = [
        [rec(True, False), None, rec(False, False)],  # fix only
        [rec(False, True)],  # comment only, fewer than 3 neighbours
        [rec(False, False), None],  # nothing
        [rec(True, True), rec(False, True)],  # both types
    ]
    r = c3.coverage(rows)
    assert r["n_rows"] == 4 and r["rows_ge1_neighbour_with_evidence"] == 3
    assert r["share_ge1_neighbour_with_evidence"] == 0.75
    assert r["share_with_fix_pr_evidence"] == 0.5
    assert r["share_with_closing_comment_evidence"] == 0.5
    assert r["share_with_both_types"] == 0.25
    # per-row chars: 6, 6, 0, 18 -> mean 7.5; evidenced neighbours: 6, 6, 12, 6 -> mean 7.5
    assert r["mean_evidence_chars_per_row_top3"] == 7.5
    assert r["mean_evidence_chars_per_evidenced_neighbour"] == 7.5
