from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp3bp3", _S / "qp3b_priority_p3.py")
p3 = importlib.util.module_from_spec(_SPEC)
sys.modules["qp3bp3"] = p3
_SPEC.loader.exec_module(p3)

T0 = pd.Timestamp("2025-03-01T00:00:00Z")
TRAIN, VAL, TEST = range(1, 161), range(161, 221), range(221, 281)
LEVEL_OF = {0: "priority/backlog", 1: "priority/important-longterm", 2: "priority/important-soon"}


def make_data(tmp_path: Path):
    """kind/bug labeled +30 s (kept), sig/x labeled +10 min (dropped), kind/feature no event (kept)."""
    rows, events, split_rows = [], [], []
    for n in [*TRAIN, *VAL, *TEST]:
        k = n % 3
        rows.append({
            "number": n, "title": f"word{k} title", "body": f"body word{k} filler{n % 7}",
            "labels_final": ["sig/x", LEVEL_OF[k]],
            "labels_t72h": ["kind/bug", "sig/x", "kind/feature", *(["priority/backlog"] if n % 5 == 0 else [])],
            "author_assoc": "MEMBER", "created_at": T0,
        })  # fmt: skip
        events += [
            {"number": n, "ts": T0 + pd.Timedelta(seconds=30), "action": "labeled", "label": "kind/bug"},
            {"number": n, "ts": T0 + pd.Timedelta(minutes=10), "action": "labeled", "label": "sig/x"},
            {"number": n, "ts": T0 + pd.Timedelta(minutes=1), "action": "unlabeled", "label": "kind/feature"},
        ]  # fmt: skip
        name = "train" if n in TRAIN else "validation" if n in VAL else "test"
        split_rows.append({"number": n, "split": name})
    derived, splits = tmp_path / "derived", tmp_path / "splits"
    derived.mkdir()
    splits.mkdir()
    pd.DataFrame(rows).to_parquet(derived / "issues.parquet")
    pd.DataFrame(events).to_parquet(derived / "label_events.parquet")
    sp = pd.DataFrame(split_rows)
    for name in ("train", "validation", "test"):
        sp[sp["split"] == name][["number"]].to_parquet(splits / f"{name}.parquet")
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return derived, splits, tmp_path / "models", tmp_path / "report.json", manifest


def test_creation_rule_unit() -> None:
    first = {"kind/bug": T0 + pd.Timedelta(seconds=60), "sig/x": T0 + pd.Timedelta(seconds=61)}
    got = p3.creation_labels(
        ["kind/bug", "sig/x", "kind/feature", "priority/backlog"], T0, first, 60
    )
    assert got == ["kind/bug", "kind/feature"]  # <=60 s kept, later dropped, no event kept
    assert p3.creation_labels(["sig/x"], T0, first, 300) == ["sig/x"]


@pytest.mark.parametrize("split", ["train", "validation", "test"])
def test_same_rule_for_every_split(tmp_path: Path, split: str) -> None:
    derived, splits, *_ = make_data(tmp_path)
    d, _ = p3.load_p3(derived, splits, split)
    assert all(set(m) == {"l:kind/bug", "l:kind/feature", "a:MEMBER"} for m in d["meta"])


def test_validate_then_test_once(tmp_path: Path) -> None:
    derived, splits, models, report, manifest = make_data(tmp_path)
    p3.pr.validate(derived, splits, models, report, manifest)
    sel = p3.validate_p3(derived, splits, models, report, manifest)
    data = json.loads(report.read_text(encoding="utf-8"))
    assert "p3_validation_table" in data and "test" not in data
    p3.pr.final_test(derived, splits, models, report, manifest, n_boot=50)
    res = p3.final_test_p3(derived, splits, models, report, manifest, n_boot=50)
    assert res["vs_P0_majority"]["difference_paired"]["point"] > 0 and sel["model_id"] == p3.P3_ID
    assert "primary" in json.loads(report.read_text(encoding="utf-8"))["test"]  # P2 kept
    with pytest.raises(p3.ev.TestGuardError):
        p3.final_test_p3(derived, splits, models, report, manifest, n_boot=50)
