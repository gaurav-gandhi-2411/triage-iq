from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import adr0073_label_table as gen  # noqa: E402


def _counts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "label": ["bug", "git", "recent-regression", "UBB", "performance"],
            "n_final": [9, 7, 5, 3, 2],
            "n_event_issues": [10, 8, 6, 4, 3],
            "frozen_excluded": [True, False, False, False, False],
        }
    )


def test_render_on_tiny_fixture() -> None:
    decisions = {
        "recent-regression": ("EXCLUDE", "r1"),
        "performance": ("BORDERLINE-EXCLUDE", "r2"),
        "UBB": ("BORDERLINE-KEEP", "r3"),
    }
    rows = gen.build_rows(_counts(), decisions)
    text = gen.render(rows)
    assert "Distinct labels: 5. Decisions: BORDERLINE-EXCLUDE 1, BORDERLINE-KEEP 1" in text
    assert "EXCLUDE 2, KEEP 1." in text
    assert "NEW exclusions for retraining (EXCLUDE or BORDERLINE-EXCLUDE and not already" in text
    assert "ADR-0070): 2." in text
    lines = text.splitlines()
    table = [x for x in lines if x.startswith("| ") and not x.startswith("| label")]
    # EXCLUDE first, then BORDERLINE-EXCLUDE, then BORDERLINE-KEEP; KEEP/frozen only in the lists.
    assert [x.split(" | ")[0] for x in table] == ["| recent-regression", "| performance", "| UBB"]
    assert "## Already excluded by ADR-0070 (EXCLUDE, NEW = no): 1" in text
    assert "bug (9)" in text
    assert "## KEEP: 1" in text
    assert text.endswith("git (7)\n")
    assert gen.render(rows) == text  # deterministic


def test_decisions_json_is_well_formed() -> None:
    dec = gen.load_decisions()
    assert len(dec) == 43
    assert {d for d, _ in dec.values()} <= set(gen.ORDER)
