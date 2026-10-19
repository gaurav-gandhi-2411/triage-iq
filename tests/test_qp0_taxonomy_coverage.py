from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "qp0t", _ROOT / "scripts" / "qp0_taxonomy_coverage.py"
)
qp0t = importlib.util.module_from_spec(_SPEC)
sys.modules["qp0t"] = qp0t
_SPEC.loader.exec_module(qp0t)

T0 = pd.Timestamp("2024-03-01T00:00:00Z")
AREA = lambda x: x.startswith("area/")  # noqa: E731


def issues(rows: list[tuple[int, list[str], list[str]]], year: int = 2024) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": [r[0] for r in rows], "created_at": T0, "created_year": year,
            "labels_t72h": [r[1] for r in rows], "labels_final": [r[2] for r in rows],
        }
    )  # fmt: skip


def events(rows: list[tuple[int, float, str, str | None, str]]) -> pd.DataFrame:
    """(number, hours after T0, label, actor, action)."""
    return pd.DataFrame(
        {
            "number": [r[0] for r in rows], "ts": pd.to_datetime([T0 + pd.Timedelta(hours=r[1]) for r in rows], utc=True),
            "label": [r[2] for r in rows], "actor": [r[3] for r in rows],
            "action": [r[4] for r in rows],
        }
    )  # fmt: skip


def test_coverage_exactly_one_ge2_and_churn() -> None:
    df = issues(
        [
            (1, [], ["area/a"]),  # labelled only after 72 h
            (2, ["area/a"], ["area/a"]),
            (3, ["area/a"], ["area/a", "area/b"]),  # churn + >= 2
            (4, ["kind/bug"], ["kind/bug"]),  # no family label
        ]
    )
    m = qp0t.metrics(df, AREA, qp0t.first_events(events([]), AREA, df))
    assert m["coverage_t72h"] == 0.5 and m["coverage_final"] == 0.75
    assert m["exactly_1"] == 0.5 and m["ge_2"] == 0.25
    assert m["churn"] == pytest.approx(2 / 3)  # issues 1 and 3 changed, among 3 with a final label


def test_time_to_first_label_and_bot_vs_human() -> None:
    df = issues([(1, [], ["area/a"]), (2, [], ["area/a"]), (3, [], ["area/a"]), (4, [], [])])
    ev = events(
        [
            (1, 2.0, "area/a", "k8s-ci-robot", "labeled"),
            (1, 1.0, "kind/bug", "human1", "labeled"),  # not a family label: ignored
            (2, 10.0, "area/a", "alice", "labeled"),
            (3, 4.0, "area/a", "dependabot[bot]", "labeled"),
            (3, 1.0, "area/a", "x", "unlabeled"),  # unlabel events never count
            (4, 5.0, "area/zzz", None, "labeled"),
        ]
    )
    fe = qp0t.first_events(ev, AREA, df)
    m = qp0t.metrics(df, AREA, fe)
    assert m["first_label_actor"] == {"bot": 2, "human": 1, "unknown": 1}
    h = m["hours_to_first_label"]
    assert h["median"] == 4.5 and h["n_with_label"] == 4 and h["n_never_labelled"] == 0
    only = issues([(5, [], [])])
    assert qp0t.metrics(only, AREA, fe)["hours_to_first_label"]["n_never_labelled"] == 1


def test_class_thresholds_and_share() -> None:
    rows = [(i, [], ["area/big"]) for i in range(30)] + [
        (100 + i, [], ["area/small"]) for i in range(5)
    ]
    rows += [(200, [], ["kind/bug"])]
    df = issues(rows)
    c = qp0t.metrics(df, AREA, qp0t.first_events(events([]), AREA, df))["classes"]
    assert (c["classes_ge30"], c["classes_ge100"], c["n_classes"]) == (1, 0, 2)
    assert c["share_labelled_in_ge30"] == pytest.approx(30 / 35)


def test_per_year_columns_and_stability() -> None:
    parts = []
    for y, lbls in ((2023, ["area/a"] * 6), (2024, ["area/a"] * 6), (2025, ["area/a"] * 6),
                    (2026, ["area/a"] * 3)):  # fmt: skip
        parts.append(issues([(y * 100 + i, [], [lbl]) for i, lbl in enumerate(lbls)], y))
    df = pd.concat(parts)
    df.loc[df["created_year"] == 2026, "labels_final"] = df.loc[
        df["created_year"] == 2026, "labels_final"
    ].map(lambda x: ["area/a"])
    tab = qp0t.class_table(df, AREA)
    assert tab["classes"]["area/a"] == {"total": 21, "2023": 6, "2024": 6, "2025": 6, "2026": 3}
    assert tab["stability"]["present_ge5_some_years"] == ["area/a"]  # 3 in 2026 < 5


def test_v_new_exclusion_and_threshold() -> None:
    rows = [(i, [], ["chat", "bug", "Editor"]) for i in range(100)]
    rows += [(1000 + i, [], ["notebook"]) for i in range(99)]  # one short of the threshold
    rows += [(2000 + i, [], ["copilot-wontfix", "terminal"]) for i in range(150)]
    df = issues(rows)
    new = qp0t.v_new_labels(df, {"editor", "terminal"})
    assert new == ["chat"]  # bug/wontfix excluded, editor/terminal in V_old, notebook < 100


def test_prefix_families() -> None:
    assert [qp0t.prefix_of(x) for x in ("kind/bug", "needs-triage", "bug")] == [
        "kind/", "needs-*", "(no prefix)",
    ]  # fmt: skip


def test_cooccurrence() -> None:
    df = issues([(1, [], ["sig/a", "area/x"]), (2, [], ["sig/a"]), (3, [], ["area/x"]),
                 (4, [], ["sig/b", "area/y"])])  # fmt: skip
    c = qp0t.cooccurrence(df)
    assert c["share_S_issues_with_A"] == pytest.approx(2 / 3)
    assert c["share_A_issues_with_S"] == pytest.approx(2 / 3)
    assert c["per_sig_top3_areas"]["sig/a"]["top3_area"] == [("area/x", 1)]


def test_build_report_end_to_end(tmp_path: Path) -> None:
    issues([(1, ["sig/a"], ["sig/a", "area/x"])]).to_parquet(tmp_path / "issues.parquet")
    events([(1, 3.0, "sig/a", "k8s-ci-robot", "labeled")]).to_parquet(
        tmp_path / "label_events.parquet"
    )
    pd.DataFrame({"label": ["sig/a", "kind/bug"], "n_issues_final": [1, 1]}).to_csv(
        tmp_path / "label_frequency.csv", index=False
    )
    rep: dict[str, Any] = qp0t.build_report("kubernetes/kubernetes", tmp_path)
    assert set(rep["families"]) == {"A", "S", "SA"}
    assert rep["families"]["S"]["overall"]["coverage_t72h"] == 1.0
    assert rep["families"]["A"]["overall"]["coverage_t72h"] == 0.0
    assert "kind/" in rep["other_label_families"] and "2023" in rep["families"]["A"]["per_year"]


def test_script_imports_no_model_or_ml_modules() -> None:
    tree = ast.parse((_ROOT / "scripts" / "qp0_taxonomy_coverage.py").read_text(encoding="utf-8"))
    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            mods |= {a.name for a in n.names}
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module or "")
    banned = ("sklearn", "torch", "transformers", "xgboost", "lightgbm", "scipy", "mlflow")
    assert not [m for m in mods if m.split(".")[0] in banned]
    assert not [m for m in mods if "model" in m or "classif" in m or "eval" in m]
    assert "triage_iq.data.preprocess" in mods  # only the label-pattern list is borrowed
