from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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


def make_pop(n: int = 400) -> pd.DataFrame:
    """Synthetic population: 4 strata of very different sizes, 6 classes with a rare one."""
    rng = np.random.default_rng(0)
    strata = rng.choice(["a", "b", "c", "d"], size=n, p=[0.6, 0.25, 0.1, 0.05])
    cls = rng.choice(["x", "y", "z", "w", "v", "rare"], size=n, p=[0.3, 0.3, 0.2, 0.1, 0.07, 0.03])
    return pd.DataFrame({"number": np.arange(1000, 1000 + n), "stratum": strata, "cls": cls})


def test_allocate_sums_and_respects_minimum_and_capacity() -> None:
    pop = {"a": 600, "b": 250, "c": 100, "d": 2}
    out = q.allocate(pop, 75, 5)
    assert sum(out.values()) == 75
    assert out["d"] == 2  # stratum smaller than the minimum is taken whole
    assert all(5 <= out[k] <= pop[k] for k in ("a", "b", "c"))
    assert out["a"] > out["b"] > out["c"]  # remainder is proportional


def test_allocate_is_proportional_when_minimum_is_inactive() -> None:
    out = q.allocate({"a": 800, "b": 200}, 50, 0)
    assert out == {"a": 40, "b": 10}


def test_allocate_rejects_impossible_minimums() -> None:
    with pytest.raises(ValueError):
        q.allocate({"a": 50, "b": 50, "c": 50}, 10, 5)


def test_draw_is_deterministic_and_seed_sensitive() -> None:
    pop = make_pop()
    a = q.draw(pop, 40, 3, seed=42)
    b = q.draw(pop.sample(frac=1, random_state=1), 40, 3, seed=42)  # row order must not matter
    assert a["number"].tolist() == b["number"].tolist()
    assert a["number"].tolist() != q.draw(pop, 40, 3, seed=7)["number"].tolist()


def test_draw_respects_strata_quotas_and_membership() -> None:
    pop = make_pop()
    s = q.draw(pop, 40, 3)
    quota = q.allocate(pop.groupby("stratum").size().to_dict(), 40, 3)
    assert s.groupby("stratum").size().to_dict() == quota
    assert s["number"].is_unique and set(s["number"]) <= set(pop["number"])
    assert len(s) == 40


def test_draw_spreads_over_classes_including_rare() -> None:
    pop = make_pop()
    s = q.draw(pop, 40, 3)
    assert set(s["cls"]) == set(pop["cls"])  # greedy least-drawn-class picks reach every class


def test_strata_tables_and_weights_recover_population() -> None:
    pop = make_pop()
    s = q.draw(pop, 40, 3)
    t = q.strata_tables(pop, s)
    assert sum(v["population"] for v in t["strata"].values()) == len(pop)
    assert sum(v["sample"] for v in t["strata"].values()) == 40
    covered = sum(n for n, k in t["stratum_x_class"].values() if k)
    assert covered + t["undrawn_population"] == len(pop)
    total = sum(q.weight(t["stratum_x_class"], r.stratum, r.cls) for r in s.itertuples())
    assert total == pytest.approx(covered)  # weights re-expand the sample to the covered cells


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
