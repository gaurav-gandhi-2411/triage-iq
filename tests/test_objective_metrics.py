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


# --- similar-issue relevance ----------------------------------------------------------------


def test_relevance_hit_recall_and_cited() -> None:
    sim = [{"number": 1}, {"number": 2}, {"number": 3}]
    ps = pairs(
        (
            row("a", similar=sim, meta={"indexable_targets": [2, 9]}),
            rec("a", similar_issues=[{"number": 9, "similarity": 0.5, "relevance_note": ""}]),
        ),
        (row("b", similar=sim, meta={"indexable_targets": [7]}), rec("b", similar_issues=[])),
        (row("c", similar=sim, meta={"indexable_targets": []}), rec("c")),
    )
    out = om.relevance_metrics(ps)
    assert out["n_scorable"] == 2 and out["status"] == "ok"
    assert out["retrieved_hit_rate"]["k"] == 1
    assert out["retrieved_recall_at_k"]["point"] == 0.25  # (1/2 + 0/1) / 2
    assert out["cited_hit_rate"]["k"] == 1
    assert out["retrieved_hit_rate"]["excluded_reasons"] == {"no_indexable_target": 1}
    # k truncates the retrieved list: target 2 is at rank 2
    assert om.relevance_metrics(ps, k=1)["retrieved_hit_rate"]["k"] == 0


def test_relevance_unscorable_when_no_indexable_targets() -> None:
    out = om.relevance_metrics(pairs((row("a", meta={"indexable_targets": []}), rec("a"))))
    assert out["n_scorable"] == 0 and out["status"].startswith("UNSCORABLE")
    assert out["retrieved_hit_rate"]["point"] is None


# --- next-step grounding --------------------------------------------------------------------


def test_cited_numbers_is_mechanical() -> None:
    assert om.cited_numbers("See #123 and #45, not 678 or issue-9, #1234abc") == {123, 45, 1234}


def test_grounding_counts_steps_citing_retrieved_neighbours_with_evidence() -> None:
    ev = {K8S: {10: True, 11: False, 12: True}}
    sim = [{"number": 10}, {"number": 11}]
    steps = [
        "Follow the fix in #10",  # grounded
        "Check #11 for context",  # cites retrieved, but no evidence
        "Try #12",  # has evidence but was not retrieved
        "Add a test",  # cites nothing
    ]
    ps = pairs(
        (row("a", similar=sim), rec("a", suggested_next_steps=steps)),
        (row("b", similar=[{"number": 11}]), rec("b", suggested_next_steps=["#11"])),  # no evid
        (row("c", VSC, similar=sim), rec("c", suggested_next_steps=["#10"])),  # no artifact
        (row("d", similar=sim), None),
    )
    out = om.grounding_metrics(ps, ev)
    assert (out["n_rows_scorable"], out["n_steps"]) == (1, 4)
    assert out["step_grounded_share"]["point"] == 0.25
    assert out["steps_citing_any_retrieved_share"] == 0.5
    assert out["rows_with_grounded_step"]["point"] == 1.0
    assert out["excluded_reasons"] == {
        "no_retrieved_neighbour_with_evidence": 1,
        "no_evidence_artifact_for_repo": 1,
        "no_plan": 1,
    }


def test_grounding_without_any_artifact_is_all_unscorable() -> None:
    out = om.grounding_metrics(pairs((row("a"), rec("a", suggested_next_steps=["x"]))), {})
    assert out["n_rows_scorable"] == 0 and out["step_grounded_share"]["point"] is None


# --- resolution -----------------------------------------------------------------------------


def test_resolution_ae_censoring_coverage_and_naive_reference() -> None:
    def r(i: str, actual: float | None, censored: bool = False) -> dict[str, Any]:
        return row(i, actual=actual, meta={"censored": censored})

    ps = pairs(
        (
            r("a", 10.0),
            rec(
                "a",
                expected_resolution_lower_days=5,
                expected_resolution_upper_days=20,
                metadata={"resolution_point_days": 12.0},
            ),
        ),
        (
            r("b", 50.0),
            rec("b", expected_resolution_lower_days=10, expected_resolution_upper_days=30),
        ),  # midpoint 20 -> AE 30; outside interval
        (
            r("c", None, True),
            rec("c", expected_resolution_lower_days=0, expected_resolution_upper_days=1),
        ),
        (r("d", 3.0), None),
    )
    out = om.resolution_metrics(ps, {K8S: 35.1})
    assert out["n"] == 2 and out["excluded_reasons"] == {"censored": 1, "no_plan": 1}
    assert out["point_source_counts"] == {
        "metadata.resolution_point_days": 1,
        "interval_midpoint": 1,
    }
    assert out["plan"]["mean_ae_days"] == 16.0 and out["plan"]["median_ae_days"] == 16.0
    assert out["interval_coverage"]["k"] == 1 and out["interval_coverage"]["n"] == 2
    assert out["mean_interval_width_days"] == 17.5
    # naive AE: |35.1-10| = 25.1, |35.1-50| = 14.9 -> mean 20.0
    assert out["naive_reference"]["mean_ae_days"] == 20.0
    assert out["naive_minus_plan_ae"]["mean_days"] == 4.0


def test_resolution_no_naive_constant_or_no_closed_rows() -> None:
    ps = pairs(
        (
            row("a", actual=4.0),
            rec("a", expected_resolution_lower_days=1, expected_resolution_upper_days=9),
        )
    )
    assert om.resolution_metrics(ps, {})["naive_reference"] is None
    out = om.resolution_metrics(pairs((row("b", actual=None), rec("b"))))
    assert out["n"] == 0 and out["plan"]["median_ae_days"] is None


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
