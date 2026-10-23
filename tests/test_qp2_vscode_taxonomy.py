from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_spec = importlib.util.spec_from_file_location("qp2vt", _S / "qp2_vscode_taxonomy.py")
vt = importlib.util.module_from_spec(_spec)
sys.modules["qp2vt"] = vt
_spec.loader.exec_module(vt)


def utc(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def make_issues(rows: list[tuple[int, str, list[str]]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": [r[0] for r in rows],
            "created_at": [utc(r[1]) for r in rows],
            "labels_final": [r[2] for r in rows],
            "labels_t72h": [r[2] for r in rows],
            "text_hash": [str(r[0]) for r in rows],
        }
    )


@pytest.mark.parametrize(
    "label",
    ["bug", "info-needed", "*duplicate", "~info-needed", "verified", "Windows 11", "linux",
     "translation-required-german", "electron-39-update", "testplan-item", "accessibility",
     "error-telemetry", "upstream", "python"],
)  # fmt: skip
def test_non_component_labels_are_excluded(label: str) -> None:
    assert vt.is_excluded(label)


@pytest.mark.parametrize(
    "label", ["terminal", "chat-mcp", "git", "editor-hover", "notebook-diff", "NES", "typescript"]
)
def test_component_labels_are_kept(label: str) -> None:
    assert not vt.is_excluded(label)


def test_primary_label_rule_skips_excluded_and_ranks_creation_first() -> None:
    t = utc("2024-03-01")
    iss = make_issues(
        [
            (1, "2024-03-01", ["bug", "terminal", "git"]),  # bug is excluded, never primary
            (2, "2024-03-01", ["git", "scm"]),  # git has no event: applied at creation, wins
        ]
    )
    ev = pd.DataFrame(
        {
            "number": [1, 1, 1, 2],
            "ts": [
                t,
                t + pd.Timedelta(hours=5),
                t + pd.Timedelta(hours=2),
                t + pd.Timedelta(hours=1),
            ],
            "action": ["labeled"] * 4,
            "label": ["bug", "terminal", "git", "scm"],
            "actor": ["a"] * 4,
        }
    )
    d = vt.cl.per_issue(iss, ev, vt.not_excluded).set_index("number")
    assert d.loc[1, "primary"] == "git"  # earliest non-excluded labeled event
    assert d.loc[2, "primary"] == "git"  # applied at creation, ahead of scm


def test_class_family_uses_window_counts_only() -> None:
    rows = [(i, "2024-01-10", ["terminal"]) for i in range(30)]
    rows += [(100 + i, "2025-08-10", ["chat-billing"]) for i in range(200)]  # validation-only label
    rows += [(400 + i, "2024-02-10", ["scm", "bug"]) for i in range(29)]  # one short of 30
    iss = make_issues(rows)
    win = iss[iss["created_at"] < vt.VAL_START]
    assert vt.family(vt.window_counts(win)) == ["terminal"]
    assert vt.family(vt.window_counts(iss)) == ["chat-billing", "terminal"]  # the leak we avoid


def test_parent_of_maps_prefix_families_and_unseen_labels() -> None:
    assert vt.parent_of("terminal-suggest") == "terminal"
    assert vt.parent_of("chat-subagents") == "chat"  # unseen label, same rule
    assert vt.parent_of("extension-host") == vt.parent_of("extensions") == "extensions"
    assert vt.parent_of("tree-sitter") == "tree-sitter"  # not merged by a first-token rule
    assert vt.parent_of("git") == "git"


def test_test_period_guard_raises() -> None:
    ok = make_issues([(1, "2025-12-31", ["git"])])
    vt.assert_no_test_rows(ok)
    bad = make_issues([(1, "2025-12-31", ["git"]), (2, "2026-01-01", ["git"])])
    with pytest.raises(AssertionError, match="test period"):
        vt.assert_no_test_rows(bad)
    with pytest.raises(AssertionError, match="test period"):
        vt.run(bad, pd.DataFrame(columns=["number", "ts", "action", "label", "actor"]))


def test_loader_never_returns_test_rows(tmp_path: Path) -> None:
    iss = make_issues(
        [(1, "2025-12-31", ["git"]), (2, "2026-01-01", ["git"]), (3, "2026-06-01", ["scm"])]
    )
    iss.to_parquet(tmp_path / "issues.parquet")
    ev = pd.DataFrame(
        {
            "number": [1, 2, 3],
            "ts": [utc("2025-12-31")] * 3,
            "action": ["labeled"] * 3,
            "label": ["git", "git", "scm"],
            "actor": ["a"] * 3,
        }
    )
    ev.to_parquet(tmp_path / "label_events.parquet")
    got_i, got_e = vt.load_train_val(tmp_path)
    assert got_i["number"].tolist() == [1]
    assert got_e["number"].tolist() == [1]
