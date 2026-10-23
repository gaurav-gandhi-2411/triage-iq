"""Retrieval-side and resolution objective metrics (definitions 3-5 in objective_metrics.py).

Split out of objective_metrics.py so each module stays reviewable; behaviour is unchanged and the
names are still importable from `triage_iq.evaluation.objective_metrics`.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from triage_iq.evaluation.objective_metrics import (
    Pair,
    _ci,
    _mean_boot,
    _proportion,
    _r,
    _result,
    bootstrap_ci,
)

# Constant naive references for the resolution metric (inputs, echoed in the report): k8s is the
# 2023-2025 closed-issue median (docs/QUEUE_FOR_GG.md, 35.1 d); vscode is the training-set median
# in days quoted by triage_iq.models.resolution.naive_median_days (3.8374 d).
NAIVE_DAYS_DEFAULT: dict[str, float] = {"kubernetes/kubernetes": 35.1, "microsoft/vscode": 3.8374}
_CITE_RE = re.compile(r"#(\d+)")


def _numbers(items: object, key: str = "number") -> list[int]:
    out: list[int] = []
    for it in items if isinstance(items, list) else []:
        n = it.get(key) if isinstance(it, Mapping) else it
        if isinstance(n, int) and not isinstance(n, bool):
            out.append(n)
    return out


def relevance_metrics(pairs: Sequence[Pair], k: int | None = None) -> dict[str, Any]:
    """Similar-issue relevance against indexable dup/fix targets (retrieved and cited)."""
    excl: Counter[str] = Counter()
    ret_hit: list[bool] = []
    cit_hit: list[bool] = []
    recall: list[float] = []
    for p in pairs:
        if p.record is None:
            excl["no_plan"] += 1
            continue
        targets = set(_numbers(p.meta.get("indexable_targets")))
        if not targets:
            excl["no_indexable_target"] += 1
            continue
        retrieved = set(_numbers(p.row.get("similar_issues"))[:k])
        cited = set(_numbers(p.plan.get("similar_issues")))
        ret_hit.append(bool(retrieved & targets))
        cit_hit.append(bool(cited & targets))
        recall.append(len(retrieved & targets) / len(targets))
    return {
        "k": k,
        "n_scorable": len(ret_hit),
        "status": "ok" if ret_hit else "UNSCORABLE: no row has an indexable target",
        "retrieved_hit_rate": _proportion(ret_hit, excl),
        "retrieved_recall_at_k": _mean_boot(recall, excl),
        "cited_hit_rate": _proportion(cit_hit, excl),
    }


def cited_numbers(step: str) -> set[int]:
    """Issue numbers a next step references as `#N` (mechanical definition, see module doc)."""
    return {int(m) for m in _CITE_RE.findall(step)}


def grounding_metrics(
    pairs: Sequence[Pair], evidence: Mapping[str, Mapping[int, bool]]
) -> dict[str, Any]:
    """Share of next steps citing a retrieved neighbour that has resolution evidence."""
    excl: Counter[str] = Counter()
    grounded: list[int] = []
    steps: list[int] = []
    citing: list[int] = []
    for p in pairs:
        if p.record is None:
            excl["no_plan"] += 1
            continue
        lookup = evidence.get(str(p.row.get("repo")))
        if lookup is None:
            excl["no_evidence_artifact_for_repo"] += 1
            continue
        retrieved = set(_numbers(p.row.get("similar_issues")))
        with_ev = {n for n in retrieved if lookup.get(n)}
        if not with_ev:
            excl["no_retrieved_neighbour_with_evidence"] += 1
            continue
        texts = [s for s in (p.plan.get("suggested_next_steps") or []) if isinstance(s, str)]
        cites = [cited_numbers(s) & retrieved for s in texts]
        steps.append(len(texts))
        grounded.append(sum(bool(c & with_ev) for c in cites))
        citing.append(sum(bool(c) for c in cites))
    g, s, c = (np.array(x, dtype=float) for x in (grounded, steps, citing))
    total = float(s.sum())
    point = float(g.sum() / total) if total else None
    ci = bootstrap_ci(len(s), lambda i: g[i].sum(axis=1) / s[i].sum(axis=1))
    return {
        "n_rows_scorable": len(s),
        "n_steps": int(total),
        "n_excluded": int(sum(excl.values())),
        "excluded_reasons": dict(sorted(excl.items())),
        "step_grounded_share": _result(int(total), {}, point, ci, "bootstrap_by_row"),
        "rows_with_grounded_step": _proportion([bool(x) for x in g > 0], {}),
        "steps_citing_any_retrieved_share": _r(float(c.sum() / total)) if total else None,
    }


def _valid_interval(lo: object, hi: object) -> bool:
    """Both bounds are finite real numbers (NaN, +/-inf, bools and non-numbers are not)."""
    return all(
        isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
        for x in (lo, hi)
    )


def _point_days(p: Pair) -> tuple[float | None, str]:
    point = p.metadata.get("resolution_point_days")
    if isinstance(point, (int, float)) and not isinstance(point, bool) and math.isfinite(point):
        return float(point), "metadata.resolution_point_days"
    lo = p.plan.get("expected_resolution_lower_days")
    hi = p.plan.get("expected_resolution_upper_days")
    if _valid_interval(lo, hi):
        return (float(lo) + float(hi)) / 2.0, "interval_midpoint"
    return None, "none"


def _ae_stats(v: np.ndarray, label: str) -> dict[str, Any]:
    med = bootstrap_ci(len(v), lambda i: np.median(v[i], axis=1))
    mean = bootstrap_ci(len(v), lambda i: v[i].mean(axis=1))
    return {
        f"median_{label}_days": _r(float(np.median(v))) if len(v) else None,
        f"median_{label}_ci95": _ci(med),
        f"mean_{label}_days": _r(float(v.mean())) if len(v) else None,
        f"mean_{label}_ci95": _ci(mean),
    }


def resolution_metrics(
    pairs: Sequence[Pair], naive_days: Mapping[str, float] | None = None
) -> dict[str, Any]:
    """Absolute error of the plan's point estimate vs actual days on closed rows, plus coverage."""
    naive = NAIVE_DAYS_DEFAULT if naive_days is None else naive_days
    excl: Counter[str] = Counter()
    sources: Counter[str] = Counter()
    ae: list[float] = []
    nae: list[float | None] = []
    cover: list[bool] = []
    iv_excl: Counter[str] = Counter()
    width: list[float] = []
    for p in pairs:
        if p.record is None:
            excl["no_plan"] += 1
            continue
        actual = p.row.get("actual_resolution_days")
        if actual is None or p.meta.get("censored") is True:
            excl["censored"] += 1
            continue
        if not _valid_interval(actual, 0.0):
            excl["invalid_actual"] += 1
            continue
        point, source = _point_days(p)
        if point is None:
            excl["no_point_estimate"] += 1
            continue
        sources[source] += 1
        ae.append(abs(point - float(actual)))
        c = naive.get(str(p.row.get("repo")))
        nae.append(None if c is None else abs(c - float(actual)))
        lo = p.plan.get("expected_resolution_lower_days")
        hi = p.plan.get("expected_resolution_upper_days")
        if not _valid_interval(lo, hi):
            iv_excl["invalid_interval"] += 1
        else:
            cover.append(float(lo) <= float(actual) <= float(hi))
            width.append(float(hi) - float(lo))
    a = np.asarray(ae, dtype=float)
    out: dict[str, Any] = {
        "n": len(a),
        "n_excluded": int(sum(excl.values())),
        "excluded_reasons": dict(sorted(excl.items())),
        "point_source_counts": dict(sources),
        "ci_method": "bootstrap",
        "plan": _ae_stats(a, "ae"),
        "interval_coverage": _proportion(cover, iv_excl),
        "mean_interval_width_days": _r(float(np.mean(width))) if width else None,
    }
    if nae and all(x is not None for x in nae):
        n_arr = np.asarray(nae, dtype=float)
        diff = n_arr - a  # positive = the plan is closer to the actual than the constant
        out["naive_reference"] = {"naive_days": dict(naive), **_ae_stats(n_arr, "ae")}
        out["naive_minus_plan_ae"] = {
            "median_days": _r(float(np.median(diff))),
            "median_ci95": _ci(bootstrap_ci(len(diff), lambda i: np.median(diff[i], axis=1))),
            "mean_days": _r(float(diff.mean())),
            "mean_ci95": _ci(bootstrap_ci(len(diff), lambda i: diff[i].mean(axis=1))),
        }
    else:
        out["naive_reference"] = None
    return out
