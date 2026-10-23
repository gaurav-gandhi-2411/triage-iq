"""CLI test: file mode (eval set + plans JSONL) end to end on synthetic data, no replay."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "qp5cli", ROOT / "scripts" / "qp5_objective_metrics.py"
)
assert _SPEC and _SPEC.loader
qp5 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qp5)


def _write(tmp_path: Path) -> tuple[Path, Path]:
    rows = [
        {
            "id": "k-1",
            "repo": "kubernetes/kubernetes",
            "gold_component": "sig/node",
            "gold_priority": "high",
            "actual_resolution_days": 10.0,
            "similar_issues": [{"number": 5}],
            "meta": {"gold_priority_source": "label", "indexable_targets": [5], "weight": 2.0},
        },
        {"id": "k-2", "repo": "kubernetes/kubernetes", "gold_component": "sig/api-machinery"},
    ]
    plans = {
        "id": "k-1",
        "plan": {
            "predicted_component": "sig/node",
            "priority_guess": "high",
            "suggested_next_steps": ["see #5"],
            "similar_issues": [{"number": 5, "similarity": 0.9, "relevance_note": ""}],
            "expected_resolution_lower_days": 1.0,
            "expected_resolution_upper_days": 30.0,
        },
        "metadata": {"resolution_point_days": 12.0},
    }
    es, pl = tmp_path / "es.jsonl", tmp_path / "plans.jsonl"
    es.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    pl.write_text(json.dumps(plans) + "\n", encoding="utf-8")
    return es, pl


def test_file_mode_writes_report_with_provenance(tmp_path: Path) -> None:
    es, pl = _write(tmp_path)
    ev = tmp_path / "ev.parquet"
    import pandas as pd

    pd.DataFrame({"number": [5], "has_evidence": [True]}).to_parquet(ev)
    out = tmp_path / "r.json"
    qp5.main(
        [
            "--eval-set",
            str(es),
            "--plans",
            str(pl),
            "--evidence",
            f"kubernetes/kubernetes={ev}",
            "--label",
            "t",
            "--out",
            str(out),
        ]
    )
    rep = json.loads(out.read_text(encoding="utf-8"))
    p = rep["pooled"]
    assert (p["n_rows"], p["n_with_plan"]) == (2, 1)
    assert p["component"]["strict"]["unweighted"]["k"] == 1
    assert p["priority"]["n"] == 1 and p["similar_issue_relevance"]["n_scorable"] == 1
    assert p["next_step_grounding"]["step_grounded_share"]["point"] == 1.0
    assert p["resolution"]["plan"]["mean_ae_days"] == 2.0
    assert rep["sources"]["plans"]["sha256"] and rep["naive_days_input"]["kubernetes/kubernetes"]
    assert rep["label"] == "t"


def test_rejects_mixed_or_missing_inputs(tmp_path: Path) -> None:
    es, pl = _write(tmp_path)
    with pytest.raises(SystemExit):
        qp5.main(["--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):
        qp5.main(["--eval-set", str(es), "--out", str(tmp_path / "x.json")])
