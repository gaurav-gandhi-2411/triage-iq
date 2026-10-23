"""Unit tests for triage_iq.evaluation.objective_metrics_retrieval (synthetic plans)."""

from __future__ import annotations

import json
from typing import Any

from triage_iq.evaluation import objective_metrics_retrieval as om
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


def test_resolution_non_finite_interval_is_no_interval_and_never_reaches_stats() -> None:
    nan, inf = float("nan"), float("inf")

    def r(i: str) -> dict[str, Any]:
        return row(i, actual=10.0)

    ps = pairs(
        # NaN upper bound, explicit point: AE kept, excluded from coverage/width
        (
            r("a"),
            rec(
                "a",
                expected_resolution_lower_days=1,
                expected_resolution_upper_days=nan,
                metadata={"resolution_point_days": 12.0},
            ),
        ),
        # inf bound and no point: the midpoint would be inf -> row leaves AE entirely
        (r("b"), rec("b", expected_resolution_lower_days=1, expected_resolution_upper_days=inf)),
        # -inf lower bound, explicit point
        (
            r("c"),
            rec(
                "c",
                expected_resolution_lower_days=-inf,
                expected_resolution_upper_days=5,
                metadata={"resolution_point_days": 11.0},
            ),
        ),
        (
            r("d"),
            rec(
                "d",
                expected_resolution_lower_days=5,
                expected_resolution_upper_days=20,
                metadata={"resolution_point_days": 10.0},
            ),
        ),
    )
    out = om.resolution_metrics(ps, {K8S: 35.1})
    assert out["n"] == 3 and out["excluded_reasons"] == {"no_point_estimate": 1}
    assert out["plan"]["mean_ae_days"] == 1.0 and out["plan"]["median_ae_days"] == 1.0
    cov = out["interval_coverage"]
    assert (cov["n"], cov["k"]) == (1, 1)
    assert cov["excluded_reasons"] == {"invalid_interval": 2}
    assert out["mean_interval_width_days"] == 15.0
    json.dumps(out, allow_nan=False)  # no NaN/inf anywhere in the report
