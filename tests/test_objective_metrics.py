"""Unit tests for triage_iq.evaluation.objective_metrics (synthetic plans, no I/O beyond tmp_path)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from triage_iq.evaluation import objective_metrics as om
from triage_iq.evaluation.objective_metrics import Pair

K8S, VSC = "kubernetes/kubernetes", "microsoft/vscode"


def row(i: str, repo: str = K8S, gold: str = "sig/node", **kw: Any) -> dict[str, Any]:
    meta = kw.pop("meta", {})
    return {
        "id": i,
        "repo": repo,
        "gold_component": gold,
        "gold_priority": kw.pop("gold_priority", None),
        "actual_resolution_days": kw.pop("actual", None),
        "similar_issues": kw.pop("similar", []),
        "meta": meta,
    }


def rec(i: str, pred: str = "sig/node", **plan: Any) -> dict[str, Any]:
    metadata = plan.pop("metadata", {})
    return {"id": i, "plan": {"predicted_component": pred, **plan}, "metadata": metadata}


def pairs(*items: tuple[dict[str, Any], dict[str, Any] | None]) -> list[Pair]:
    return [Pair(r, c) for r, c in items]


# --- statistics -----------------------------------------------------------------------------


def test_wilson_known_value_and_empty() -> None:
    lo, hi = om.wilson(8, 10)  # type: ignore[misc]
    assert (round(lo, 4), round(hi, 4)) == (0.4902, 0.9433)
    assert om.wilson(0, 0) is None
    assert om.wilson(0, 5)[0] == 0.0  # type: ignore[index]


def test_bootstrap_is_deterministic_and_handles_empty() -> None:
    import numpy as np

    v = np.arange(10, dtype=float)
    a = om.bootstrap_ci(10, lambda i: v[i].mean(axis=1))
    assert a == om.bootstrap_ci(10, lambda i: v[i].mean(axis=1))
    assert a is not None and a[0] < 4.5 < a[1]
    assert om.bootstrap_ci(0, lambda i: i.sum(axis=1)) is None


# --- component ------------------------------------------------------------------------------


def test_component_strict_vs_k8s_convention_and_in_family() -> None:
    ps = pairs(
        (row("a", VSC, "editor", meta={"in_family": True, "weight": 3}), rec("a", "Editor")),
        (row("b", VSC, "editor", meta={"in_family": True, "weight": 1}), rec("b", "git")),
        (row("c", VSC, "other", meta={"in_family": False, "weight": 1}), rec("c", "other")),
        (row("d", VSC, "other", meta={"in_family": False, "weight": 1}), rec("d", "git")),
        (row("e", VSC, "editor", meta={"in_family": True}), None),
    )
    out = om.component_metrics(ps)
    assert out["n_scored"] == 4 and out["excluded_reasons"] == {"no_plan": 1}
    # strict: only 'a' hits (gold other is always a miss; predicted other never a hit)
    assert out["strict"]["unweighted"]["k"] == 1 and out["strict"]["unweighted"]["n"] == 4
    # k8s convention: 'a' and 'c' (other == other)
    assert out["k8s_convention"]["unweighted"]["k"] == 2
    # in-family strict: a, b -> 1/2; weighted 3/4 (weights 3 and 1)
    assert out["in_family_strict"]["unweighted"]["n"] == 2
    assert out["in_family_strict"]["weighted"]["point"] == 0.75
    assert out["strict"]["weighted"]["point"] == round(3 / 6, 4)


def test_component_legacy_rows_derive_family_and_default_weight() -> None:
    out = om.component_metrics(pairs((row("a", K8S, "test"), rec("a", "test"))))
    assert out["in_family_source"] == "derived_gold_ne_other"
    assert out["rows_with_default_weight_1"] == 1
    assert "rows_with_invalid_weight_defaulted" not in out
    assert out["strict"]["unweighted"]["point"] == 1.0


def test_component_missing_gold_excluded_and_empty_population() -> None:
    out = om.component_metrics(pairs((row("a", gold=""), rec("a"))))
    assert out["n_scored"] == 0 and out["excluded_reasons"] == {"no_gold_component": 1}
    assert out["strict"]["unweighted"]["point"] is None
    assert out["strict"]["weighted"]["point"] is None


# --- priority -------------------------------------------------------------------------------


def test_priority_scores_label_rows_only_and_reports_exclusions() -> None:
    def r(i: str, src: str | None, gold: str | None) -> dict[str, Any]:
        m = {} if src is None else {"gold_priority_source": src}
        return row(i, gold_priority=gold, meta=m)

    ps = pairs(
        (r("a", "label", "high"), rec("a", priority_guess="high")),
        (r("b", "label", "high"), rec("b", priority_guess="low")),
        (r("c", "label", "low"), rec("c", priority_guess="low")),
        (r("d", "resolution_inferred", "high"), rec("d", priority_guess="high")),
        (r("e", "none", None), rec("e", priority_guess="low")),
        (r("f", None, "low"), rec("f", priority_guess="low")),
        (r("g", "label", "high"), None),
    )
    out = om.priority_metrics(ps)
    assert (out["n"], out["k"]) == (3, 2)
    assert out["excluded_reasons"] == {
        "gold_source_resolution_inferred": 1,
        "gold_source_none": 1,
        "gold_source_unknown_legacy": 1,
        "no_plan": 1,
    }
    assert out["majority_reference"]["label"] == "high"
    assert out["majority_reference"]["accuracy"] == round(2 / 3, 4)


def test_priority_no_label_rows_is_unscorable_not_zero() -> None:
    out = om.priority_metrics(pairs((row("a", gold_priority="low"), rec("a"))))
    assert out["n"] == 0 and out["point"] is None and out["ci95"] is None


# --- assembly and loaders -------------------------------------------------------------------


def test_score_all_splits_by_repo_counts_degraded_and_joins_by_id() -> None:
    rows = [row("a"), row("b", VSC, "editor"), row("c")]
    records = {
        "a": rec("a"),
        "b": {**rec("b", "editor"), "metadata": {"llm_status": "degraded_provider_error"}},
    }
    out = om.score_all(om.join_records(rows, records))
    assert set(out["by_repo"]) == {K8S, VSC}
    assert out["pooled"]["n_rows"] == 3 and out["pooled"]["n_with_plan"] == 2
    assert out["pooled"]["n_degraded_plans"] == 1
    assert out["by_repo"][VSC]["component"]["strict"]["unweighted"]["point"] == 1.0
    assert out["pooled"]["component"]["excluded_reasons"] == {"no_plan": 1}
    json.dumps(out)  # report must be JSON-serialisable


def test_loaders_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "plans.jsonl"
    p.write_text(json.dumps(rec("x")) + "\n\n" + json.dumps(rec("y")) + "\n", encoding="utf-8")
    assert set(om.load_records(p)) == {"x", "y"}
    import pandas as pd

    pq = tmp_path / "ev.parquet"
    pd.DataFrame({"number": [1, 2], "has_evidence": [True, False]}).to_parquet(pq)
    assert om.load_evidence_lookup(pq) == {1: True, 2: False}


@pytest.mark.parametrize("bad", [None, "", "  "])
def test_norm_label_blank(bad: object) -> None:
    assert om.norm_label(bad) == ""


def test_moved_metrics_stay_importable_from_the_original_module() -> None:
    from triage_iq.evaluation import objective_metrics_retrieval as moved

    for name in ("relevance_metrics", "grounding_metrics", "resolution_metrics", "cited_numbers"):
        assert getattr(om, name) is getattr(moved, name)
    assert om.NAIVE_DAYS_DEFAULT is moved.NAIVE_DAYS_DEFAULT
    with pytest.raises(AttributeError):
        om.no_such_metric  # noqa: B018 - the access itself is the assertion


def test_invalid_explicit_weights_default_to_one_and_are_counted() -> None:
    def r(i: str, **meta: Any) -> dict[str, Any]:
        return row(i, meta=meta)

    ps = pairs(
        (r("a", weight=float("inf")), rec("a")),
        (r("b", weight=float("nan")), rec("b")),
        (r("c", weight=-2.0), rec("c")),
        (r("d", weight="3"), rec("d", "other")),
        (r("e", weight=4.0), rec("e")),
        (r("f"), rec("f")),  # absent: the existing default counter, not the invalid one
    )
    out = om.component_metrics(ps)
    assert out["rows_with_invalid_weight_defaulted"] == 4
    assert out["rows_with_default_weight_1"] == 1
    # hits a,b,c,e,f weigh 1,1,1,4,1; d (miss) weighs 1 -> 8/9, never inf/NaN
    assert out["strict"]["weighted"]["point"] == round(8 / 9, 4)
