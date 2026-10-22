from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_SCRIPTS))
_SPEC = importlib.util.spec_from_file_location("qp1_gold_v2", _SCRIPTS / "qp1_gold_v2_rescore.py")
g = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1_gold_v2"] = g
_SPEC.loader.exec_module(g)

CLASSES = ["sig/a", "sig/b", "other"]


def table(rows: list[tuple[int, str, str, set[str]]]) -> pd.DataFrame:
    """(number, split, primary, final) -> gold table with every issue classifier_eligible."""
    return pd.DataFrame(
        {
            "number": [r[0] for r in rows],
            "split": [r[1] for r in rows],
            "primary": [r[2] for r in rows],
            "final": [frozenset(r[3]) for r in rows],
            "classifier_eligible": True,
        }
    )


OLD = table(
    [
        (1, "test", "sig/a", {"sig/a", "sig/b"}),
        (2, "test", "sig/a", {"sig/a"}),
        (3, "test", "sig/b", {"sig/b"}),
        (4, "validation", "sig/a", {"sig/a"}),
    ]
)
NEW = table(
    [
        (1, "test", "sig/b", {"sig/a", "sig/b"}),  # primary flips a -> b
        (2, "test", "sig/a", {"sig/a", "sig/b"}),  # label set grows, primary same
        (5, "test", "sig/b", {"sig/b"}),  # newly labelled (no saved probabilities)
        (4, "validation", "sig/a", {"sig/a"}),
    ]
)  # issue 3 is unlabelled in NEW


def test_compare_gold_counts_every_kind_of_change() -> None:
    c = g.compare_gold(OLD, NEW, {"sig/a", "sig/b"})["test"]
    assert c["n_old_labelled"] == 3 and c["n_new_labelled"] == 3 and c["n_both"] == 2
    assert (c["newly_labelled"], c["unlabelled"]) == (1, 1)
    assert (c["label_set_changed"], c["primary_changed"]) == (1, 1)
    assert c["share_primary_changed"] == 0.5
    assert g.compare_gold(OLD, NEW, {"sig/a"})["test"]["primary_changed_modelled_class"] == 1


def test_class_counts_map_unmodelled_labels_to_other() -> None:
    counts = g.class_counts(OLD, {"sig/a"})["test"]
    assert counts == {"sig/a": 2, "other": 1}


def _write_probs(models: Path, split: str, numbers: list[int], favour: list[int]) -> None:
    p = np.full((len(numbers), 3), 0.1)
    p[np.arange(len(numbers)), favour] = 0.8
    df = pd.DataFrame(p, columns=CLASSES).assign(number=numbers)
    (models / "tfidf_lr").mkdir(parents=True, exist_ok=True)
    df.to_parquet(models / "tfidf_lr" / f"{split}.parquet", index=False)


def test_rescore_changes_only_with_gold_and_skips_unscored_issues(tmp_path: Path) -> None:
    _write_probs(tmp_path, "test", [1, 2, 3], [0, 0, 1])  # favours a, a, b
    _write_probs(tmp_path, "validation", [4], [0])
    out: dict[str, Any] = g.rescore(OLD, NEW, tmp_path, {"classes": CLASSES})
    t = out["test"]
    assert t["old_gold"]["top1"]["point"] == 1.0 and t["old_gold"]["n"] == 3
    # NEW test gold has 3 issues; #5 has no probabilities so only #1, #2 are scored
    assert t["gold_v2"]["n"] == 2 and t["gold_v2"]["top1"]["point"] == 0.5
    assert t["n_new_labelled_without_probabilities"] == 1
    # on the common issues (#1, #2) the old gold scores 1.0, v2 gold 0.5: the gold alone moved it
    assert t["common_issues_old_gold"]["top1"]["point"] == 1.0
    assert t["common_issues_gold_v2"]["top1"]["point"] == 0.5
    assert t["gold_v2"]["anyvalid_top1"]["point"] == 1.0  # a is in both final sets
    json.dumps(out)  # serialisable


def test_load_old_per_issue_from_saved_script(tmp_path: Path) -> None:
    s = tmp_path / "old.py"
    s.write_text("def per_issue(issues, events, family):\n    return 'old'\n", encoding="utf-8")
    assert g.load_old_per_issue(s)(None, None, None) == "old"
