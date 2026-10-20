from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_S = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(_S))
_SPEC = importlib.util.spec_from_file_location("qp1vf", _S / "qp1_vscode_taxonomy_freeze.py")
fz = importlib.util.module_from_spec(_SPEC)
sys.modules["qp1vf"] = fz
_SPEC.loader.exec_module(fz)

FAM = {"a", "b", "c", "newlab", "unseen", "tie1", "tie2"}.__contains__


def build(spec: list[tuple[str, str, list[str]]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """spec rows: (id prefix, created ISO date, final labels); every label is applied 1 h after creation."""
    rows, ev, n = [], [], 0
    for date, count, labels in [(s[1], int(s[0].split(":")[0]), s[2]) for s in spec]:
        for _ in range(count):
            n += 1
            t = pd.Timestamp(date, tz="UTC")
            rows.append({"number": n, "created_at": t, "created_year": t.year, "labels_final": labels, "labels_t72h": labels, "text_hash": f"h{n}"})  # fmt: skip
            ev += [
                {"number": n, "ts": t + pd.Timedelta(hours=1), "action": "labeled", "label": x}
                for x in labels
            ]
    events = pd.DataFrame(ev).assign(ts=lambda x: pd.to_datetime(x["ts"], utc=True))
    return pd.DataFrame(rows), events


SPEC = [
    ("6:", "2023-03-01", ["a"]), ("6:", "2024-03-01", ["a"]), ("20:", "2025-03-01", ["a"]),  # stable class: 32 train
    ("40:", "2023-04-01", ["b"]),  # >= 30 but absent in 2024: unstable -> other
    ("3:", "2023-05-01", ["c"]), ("3:", "2024-05-01", ["c"]),  # sparse
    ("40:", "2025-04-01", ["newlab"]),  # appears only in 2025: other
    ("6:", "2023-06-01", ["tie1", "tie2"]),  # same event time: alphabetical primary
    ("3:", "2023-07-01", ["x-not-in-family"]),  # unlabelled by the family
    ("10:", "2025-09-01", ["a"]), ("4:", "2025-10-01", ["b"]),  # validation
    ("10:", "2026-02-01", ["a"]), ("3:", "2026-03-01", ["unseen"]), ("2:", "2026-04-01", ["c"]), ("1:", "2026-05-01", []),  # test
]  # fmt: skip


def run() -> dict[str, Any]:
    issues, events = build(SPEC)
    return fz.freeze(issues, events, FAM)


def test_class_rule_requires_train_volume_and_stability_in_2023_and_2024() -> None:
    rep = run()
    assert rep["classes"] == ["a"] and rep["n_classes"] == 1
    assert rep["class_train_counts_per_year"]["a"] == {
        "2023": 6,
        "2024": 6,
        "2025": 20,
        "train_total": 32,
    }
    assert set(rep["f_labels_not_classes"]) >= {
        "b",
        "c",
        "newlab",
        "tie1",
        "tie2",
    }  # unstable / sparse / new
    assert rep["f_labels_not_classes"]["b"]["2024"] == 0


def test_primary_ties_alphabetical_and_mapping_to_other() -> None:
    issues, events = build(SPEC)
    d = fz.cl.per_issue(issues, events, FAM)
    tie = d[d["final"] == frozenset({"tie1", "tie2"})]
    assert set(tie["primary"]) == {"tie1"}


def test_per_split_counts_unseen_vs_sparse_and_majority_baseline() -> None:
    rep = run()["per_split"]
    tr = rep["train"]
    assert tr["labelled"] == 6 + 6 + 20 + 40 + 6 + 40 + 6 and tr["primary_is_class"] == 32
    assert tr["issues"] == 127  # 3 train issues carry no family label
    te = rep["test"]
    assert (te["issues"], te["labelled"]) == (16, 15)  # one test issue has no label
    assert te["primary_is_class"] == 10 and te["primary_is_other"] == 5
    assert (
        te["other_never_seen_in_train"] == 3 and te["other_sparse_seen_in_train"] == 2
    )  # unseen vs c (sparse)
    assert te["coverage_f_final"] == 15 / 16 and "always_majority_accuracy" in te
    # train priors: class a 32 vs other 92 -> the majority is `other`, so test accuracy = share of other
    out = run()
    assert out["train_majority"]["class"] == "other"
    assert te["always_majority_accuracy"] == 5 / 15


def test_deterministic_and_hash_stable() -> None:
    a, b = run(), run()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert len(a["classes_sha256"]) == 64


def test_vscode_family_excludes_non_ownership_labels() -> None:
    rows = [
        {"labels_final": ["chat", "error-telemetry", "chat-ext-issue", "api-proposal", "notebook"]}
        for _ in range(100)
    ]
    fam = fz.vscode_family(pd.DataFrame(rows))
    assert fam("chat") and fam("notebook")
    assert not fam("error-telemetry") and not fam("chat-ext-issue") and not fam("api-proposal")
    assert {"error-telemetry", "chat-ext-issue", "api-proposal"} == fz.NON_OWNERSHIP
