from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "eval"))  # the replay harness imports its siblings top-level
_SPEC = importlib.util.spec_from_file_location("qp3b", ROOT / "scripts" / "qp3_build_eval_set.py")
q = importlib.util.module_from_spec(_SPEC)
sys.modules["qp3b"] = q
_SPEC.loader.exec_module(q)

COMMITTED_KEYS = set(
    json.loads((ROOT / "eval" / "eval_set.jsonl").read_text("utf-8").splitlines()[0])
)


def _row(**kw: object) -> pd.Series:
    base = {
        "number": 5, "title": "t", "body": "b", "cls": "sig/node", "stratum": "s", "primary": "p",
        "parent": "p", "in_family": True, "state": "CLOSED", "labels_final": ["kind/bug"],
        "dup_targets": [], "fix_peers": [], "fix_prs": [], "prio_label": None, "days": 2.0,
        "created_at": pd.Timestamp("2026-02-01T00:00:00Z"),
        "scraped_at_max": pd.Timestamp("2026-10-01T00:00:00Z"),
    }  # fmt: skip
    return pd.Series({**base, **kw})


def test_to_row_schema_and_priority_sources() -> None:
    r = q.to_row(_row(), "kubernetes/kubernetes", 2.0)
    assert COMMITTED_KEYS - {"similar_issues"} <= set(r)  # same keys as the committed set
    assert r["id"] == "k8s-5" and r["gold_priority"] == "medium"
    assert r["meta"]["gold_priority_source"] == "resolution_inferred"
    lab = q.to_row(_row(prio_label="high"), "kubernetes/kubernetes", 1.0)
    assert lab["gold_priority"] == "high" and lab["meta"]["gold_priority_source"] == "label"
    op = q.to_row(_row(state="OPEN", days=float("nan")), "microsoft/vscode", 1.0)
    assert op["actual_resolution_days"] is None and op["gold_priority"] is None
    assert op["meta"]["censored"] is True and op["meta"]["gold_priority_source"] == "none"


def test_replay_loader_consumes_the_schema(tmp_path: Path) -> None:
    """The harness's own loaders accept rows in this schema (with frozen similar_issues)."""
    import run_eval
    from frozen_retriever import build_frozen_retrievers

    rows = []
    for i, repo in enumerate(("kubernetes/kubernetes", "microsoft/vscode")):
        r = q.to_row(_row(number=100 + i), repo, 1.0)
        r["similar_issues"] = [{"number": 1, "score": 0.5, "text": "x"}]
        rows.append(r)
    path = tmp_path / "eval_set.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    assert [i["id"] for i in run_eval._load_eval_set(path)] == ["k8s-100", "vscode-101"]
    fr = build_frozen_retrievers(path)
    assert fr["kubernetes/kubernetes"].retrieve("", k=5, exclude_number=100)[0]["number"] == 1
