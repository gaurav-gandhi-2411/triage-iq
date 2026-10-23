from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp3bp", _S / "qp3b_priority.py")
pr = importlib.util.module_from_spec(_SPEC)
sys.modules["qp3bp"] = pr
_SPEC.loader.exec_module(pr)

TRAIN, VAL, TEST = range(1, 161), range(161, 221), range(221, 281)
LEVEL_OF = {0: "priority/backlog", 1: "priority/important-longterm", 2: "priority/important-soon"}


def make_data(tmp_path: Path):
    """Synthetic corpus: the level is signalled by a word; every 4th issue has no level."""
    rows, split_rows = [], []
    for n in [*TRAIN, *VAL, *TEST]:
        k = n % 3
        final = ["sig/x", "needs-triage"]
        t72 = ["kind/bug", "sig/x"] + (["priority/backlog"] if n % 5 == 0 else [])
        if n % 4:
            final.append(LEVEL_OF[k])
        else:
            final.append("priority/awaiting-more-evidence")
        rows.append({
            "number": n, "title": f"word{k} title", "body": f"body word{k} filler{n % 7}",
            "labels_final": final, "labels_t72h": t72, "author_assoc": "MEMBER",
        })  # fmt: skip
        name = "train" if n in TRAIN else "validation" if n in VAL else "test"
        split_rows.append({"number": n, "split": name})
    derived, splits = tmp_path / "derived", tmp_path / "splits"
    derived.mkdir()
    splits.mkdir()
    pd.DataFrame(rows).to_parquet(derived / "issues.parquet")
    pd.DataFrame({"x": [1]}).to_parquet(derived / "label_events.parquet")
    sp = pd.DataFrame(split_rows)
    for name in ("train", "validation", "test"):
        sp[sp["split"] == name][["number"]].to_parquet(splits / f"{name}.parquet")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return derived, splits, tmp_path / "models", tmp_path / "report.json", manifest


def test_gold_mapping_and_most_urgent() -> None:
    assert pr.gold4(["priority/backlog", "priority/important-soon"]) == "priority/important-soon"
    assert (
        pr.gold4(["priority/backlog", "priority/critical-urgent", "x"])
        == "priority/critical-urgent"
    )
    assert [pr.gold3([lv]) for lv in pr.LEVELS] == ["high", "high", "medium", "low"]
    assert pr.gold3(["priority/important-longterm", "priority/backlog"]) == "medium"


def test_awaiting_more_evidence_alone_is_excluded() -> None:
    assert pr.gold4(["priority/awaiting-more-evidence", "kind/bug"]) is None
    assert pr.gold3(["priority/awaiting-more-evidence"]) is None
    assert pr.gold3(["priority/awaiting-more-evidence", "priority/backlog"]) == "low"


def test_priority_labels_never_reach_inputs() -> None:
    labels = [
        "kind/bug",
        "priority/critical-urgent",
        "sig/node",
        "priority/backlog",
        "triage/accepted",
    ]
    assert pr.input_labels(labels) == ["kind/bug", "sig/node"]
    assert not any("priority" in k for k in pr.meta_dict(labels, "MEMBER"))


def test_majority_from_train_only_with_class_order_tiebreak() -> None:
    assert pr.majority(["low", "high", "high"]) == "high"
    assert pr.majority(["low", "high"]) == "low"  # tie -> CLASSES order, deterministic


def test_population_and_features_exclude_priority(tmp_path: Path) -> None:
    derived, splits, *_ = make_data(tmp_path)
    tr, n = pr.load_split(derived, splits, "train")
    assert n == len(TRAIN) and len(tr) == sum(1 for i in TRAIN if i % 4)
    assert tr["prio_t72h"].sum() > 0  # some issues had a priority label at t72h ...
    assert not any("priority" in k for m in tr["meta"] for k in m)  # ... never a feature


def test_validate_then_final_test_once(tmp_path: Path) -> None:
    derived, splits, models, report, manifest = make_data(tmp_path)
    sel = pr.validate(derived, splits, models, report, manifest)
    assert sel["val_accuracy"] > 0.9  # synthetic words fully determine the level
    assert "test" not in json.loads(report.read_text(encoding="utf-8"))
    res = pr.final_test(derived, splits, models, report, manifest, n_boot=50)
    assert res["population"]["n_population"] == sum(1 for i in TEST if i % 4)
    assert res["primary"]["difference_paired"]["point"] > 0
    data = json.loads(report.read_text(encoding="utf-8"))
    assert set(data["models"]) == {pr.P0_ID, sel["model_id"]}
    with pytest.raises(pr.ev.TestGuardError):
        pr.final_test(derived, splits, models, report, manifest, n_boot=50)
