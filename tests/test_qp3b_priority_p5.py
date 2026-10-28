from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_S = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_S))
_spec = importlib.util.spec_from_file_location("qp3bp5", _S / "qp3b_priority_p5.py")
p5 = importlib.util.module_from_spec(_spec)
sys.modules["qp3bp5"] = p5
_spec.loader.exec_module(p5)

LEVELS = ["priority/critical-urgent", "priority/important-longterm", "priority/backlog"]
T0 = pd.Timestamp("2026-01-01", tz="UTC")


def make_data(tmp_path: Path):
    """Issue n: n%4==3 unprioritised ('plain' text); else level n%4 with matching words."""
    rows, split_rows = [], []
    for n in range(1, 241):
        k = n % 4
        labels = [] if k == 3 else [LEVELS[k]]
        rows.append({
            "number": n, "title": f"{'plain' if k == 3 else 'prio'} word{k}",
            "body": f"body word{k} filler{n % 7}", "labels_final": labels,
            "labels_t72h": labels[:1] if n % 5 == 0 else [], "author_assoc": "NONE",
            "created_at": T0,
        })  # fmt: skip
        name = "train" if n <= 140 else "validation" if n <= 190 else "test"
        split_rows.append({"number": n, "split": name})
    derived, splits, models = tmp_path / "derived", tmp_path / "splits", tmp_path / "models"
    derived.mkdir()
    splits.mkdir()
    pd.DataFrame(rows).to_parquet(derived / "issues.parquet")
    sp = pd.DataFrame(split_rows)
    for name in ("train", "validation", "test"):
        sp[sp["split"] == name][["number"]].to_parquet(splits / f"{name}.parquet")
    # P4 artifacts as the real run leaves them (refit with the same code path)
    pr = p5.pr
    out = models / "priority" / "p4"
    out.mkdir(parents=True)
    trp = pr.load_split(derived, splits, "train")[0]
    m4 = pr.Model(False, 1).fit(trp)
    for name in ("validation", "test"):
        d = pr.load_split(derived, splits, name)[0]
        p5.t1.save_proba(out / f"{name}.parquet", d["number"], m4.proba(d), pr.CLASSES)
    (out / "selection.json").write_text(json.dumps({"C": 1}), encoding="utf-8")
    return derived, splits, models, tmp_path / "report.json"


def test_best_threshold_prefers_precision_and_higher_tie() -> None:
    y = np.array([1, 1, 0, 0, 1, 0], dtype=bool)
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    t, f = p5.best_threshold(y, p)
    assert t == 0.8 and f == pytest.approx(0.9090909)  # P=1, R=2/3: F0.5 = 1.25*2/3 / (0.25 + 2/3)


def test_predict_weighted_reweights_high_down() -> None:
    p = np.array([[0.2, 0.3, 0.5]])
    assert p5.predict_weighted(p, p5.weights(1.0, 1.0))[0] == 2
    assert p5.predict_weighted(p, p5.weights(1.3, 0.5))[0] == 1


def test_stage1_population_is_all_issues(tmp_path: Path) -> None:
    derived, splits, *_ = make_data(tmp_path)
    d = p5.load_all(derived, splits, "validation")
    assert len(d) == 50 and 0 < d["positive"].sum() < 50
    stage2_rows = p5.pr.load_split(derived, splits, "validation")[0]["number"]
    assert list(d[d["positive"]]["number"]) == list(stage2_rows)  # exactly P4's population


def test_validate_freeze_then_test_once(tmp_path: Path) -> None:
    derived, splits, models, report = make_data(tmp_path)
    frozen = p5.validate(derived, splits, models, report)
    assert frozen["frozen"]["model_id"] == p5.P5_ID
    assert "test" not in json.loads(report.read_text(encoding="utf-8"))  # test untouched
    res = p5.final_test(derived, splits, models, report, n_boot=50)
    assert res["stage1"]["roc_auc"] > 0.9 and res["n_prioritised"] > 0
    assert set(res["criteria_met"]) == {"C1", "C2", "C3", "C4"}
    with pytest.raises(p5.ev.TestGuardError):
        p5.final_test(derived, splits, models, report, n_boot=50)


def test_final_test_rejects_changed_freeze(tmp_path: Path) -> None:
    derived, splits, models, report = make_data(tmp_path)
    p5.validate(derived, splits, models, report)
    fz = models / "priority" / "p5" / "frozen.json"
    cfg = json.loads(fz.read_text(encoding="utf-8"))
    cfg["h_high"] = 0.5 if cfg["h_high"] != 0.5 else 0.6
    fz.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(AssertionError, match="frozen config changed"):
        p5.final_test(derived, splits, models, report, n_boot=50)
