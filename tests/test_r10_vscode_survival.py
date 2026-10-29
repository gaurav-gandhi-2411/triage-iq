from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import qp1_eval as ev  # noqa: E402
import r10_vscode_survival as r10  # noqa: E402


def test_pop_rows_test_guard_and_population() -> None:
    tab = pd.DataFrame({
        "split": ["train", "validation", "test", "test"],
        "eligible": [True, True, True, False], "leak": [False, False, False, False],
    })  # fmt: skip
    with pytest.raises(ev.TestGuardError):
        r10.pop_rows(tab, "test", "eligible", False)
    assert len(r10.pop_rows(tab, "test", "eligible", True)) == 1
    assert len(r10.pop_rows(tab, "test", "nonleaky", True)) == 2


def test_frozen_split_boundaries_are_reused() -> None:
    manifest = Path(__file__).resolve().parents[1] / "reports" / f"qp0_split_manifest_{r10.SLUG}_2026-10.json"  # fmt: skip
    b = json.loads(manifest.read_text(encoding="utf-8"))["boundaries"]
    assert b == {"test_start": "2026-01-01 00:00:00+00:00", "validation_start": "2025-07-01 00:00:00+00:00"}  # fmt: skip


def test_final_test_refused_when_already_read_or_unfrozen(tmp_path: Path) -> None:
    tab = pd.DataFrame({"split": ["train"], "eligible": [True], "leak": [False]})
    rep = tmp_path / "r.json"
    for body in (
        {"freeze": {"test_read": True}},  # test already read
        {"freeze": {"test_read": False}, "test": {}},  # a test entry exists
        {},  # no freeze record at all: fail closed
    ):
        rep.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(r10.FinalTestError):
            r10.run("final", tab, rep, tmp_path)


def test_criteria_requires_all_three_against_both_baselines() -> None:
    def gain(lo: float) -> dict:
        ci = {"point": 1.0, "ci95": [lo, 2.0], "n": 5}
        return {"median_ae_gain_days": ci, "ipcw_mae_gain_days": ci}

    champ = {"harrell_c": {"ci95": [0.55, 0.6]}}
    ok = r10.criteria({"served": gain(0.1), "naive": gain(0.2)}, champ)
    assert ok["all_met"] and ok["C1"]
    assert not r10.criteria({"served": gain(0.1), "naive": gain(-0.1)}, champ)["all_met"]
    assert not r10.criteria({"served": gain(0.1)}, {"harrell_c": {"ci95": [0.49, 0.6]}})["all_met"]
