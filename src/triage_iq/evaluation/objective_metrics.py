"""Objective plan metrics (spec-quality-push.md section 10, runbook Q8 first step).

Scores RECORDED triage plans against ground truth carried by the eval set. Zero tokens: nothing in
this module calls an LLM, the network or the cassette. The judge (`triage_eval.py`) stays only for
prose quality; every dimension below has ground truth and is scored mechanically.

PRE-REGISTERED DEFINITIONS (written before this module was run on any recorded plan; a change to
any definition after the first run needs a new version tag and a note in the report).

Inputs.  An eval-set row (eval/sets/quality_push_2026-10/eval_set.jsonl shape: id, repo,
gold_component, gold_priority, actual_resolution_days, similar_issues[{number, ...}], and `meta`
with weight, in_family, gold_priority_source, censored, indexable_targets) joined by `id` to a
recorded record `{"id", "plan": <TriagePlan.model_dump()>, "metadata": {...}}`. Rows with no record
are excluded from every metric with reason `no_plan`. Degraded/fallback plans are scored as served
(they are what a user would get); their count is reported, not removed.

Every metric reports n (rows or steps scored), n_excluded (a reason -> count map), a point estimate
and a 95% CI. CI methods: Wilson score for a proportion over independent rows; percentile bootstrap
(2000 resamples, numpy default_rng seed 42, resampling rows) for everything else, including
weighted, ratio-of-sums and median statistics. Each metric is reported pooled and per repo.

1. Component.  norm(x) = lower-cased, whitespace-collapsed. pred = plan.predicted_component,
   gold = row.gold_component, in_family = meta.in_family (absent -> `norm(gold) != "other"`; the
   report states which). Three variants, all over rows with a gold component:
   - strict (ADR-0070 headline): hit iff norm(pred) == norm(gold), gold != "other", pred != "other".
     Denominator: all scored rows (gold outside the family is a miss; predicted `other` is never a
     hit).
   - k8s_convention (ADR-0067/0070 comparability variant): hit iff norm(pred) == norm(gold), where
     `other` == `other` counts. Denominator: all scored rows.
   - in_family_strict: strict hit, denominator restricted to rows with in_family True.
   Each is reported unweighted (Wilson) and weighted by meta.weight (absent -> 1.0): the weighted
   point is sum(w * hit) / sum(w), CI by bootstrap over rows. Exact match only: no label aliasing.

2. Priority.  Scored ONLY on rows with meta.gold_priority_source == "label" and gold_priority in
   {low, medium, high}. Rows whose source is "resolution_inferred" (contaminated: inferred from the
   resolution time that other metrics predict), "none", or absent (legacy sets, source unknown) are
   never scored; they are counted in n_excluded by reason. Point: accuracy of plan.priority_guess
   (3 levels), Wilson CI. Reference row: the majority-class accuracy of the scored gold labels.

3. Similar-issue relevance.  Scorable rows: those with >= 1 number in meta.indexable_targets (dup
   targets and audited fix links that exist in the retriever's index); other rows are excluded
   (`no_indexable_target`). retrieved = numbers of the row's similar_issues (the frozen retriever's
   top-k, first k if `k` is given); cited = numbers in plan.similar_issues. Reported: hit rate of
   retrieved (share of scorable rows with retrieved intersect targets non-empty; Wilson), recall@k of
   retrieved (mean of |retrieved intersect targets| / |targets|; bootstrap), and hit rate of cited
   (Wilson). A served index holding none of the targets leaves n = 0 (UNSCORABLE), by design.

4. Next-step grounding.  Evidence lookup: number -> has_evidence (a fix-PR title or a non-bot closing
   comment; the candidate artifact resolution_evidence_<repo>_v1.parquet), one lookup per repo. A
   next step CITES neighbour N iff its text contains the token `#N` (regex `#(\\d+)`, digits
   maximal) and N is in the row's retrieved numbers (the eval row's similar_issues, not the plan's
   own list, which the LLM could invent). A step is GROUNDED iff it cites at least one retrieved
   neighbour whose lookup entry is True. A row is scorable iff the repo has a lookup and >= 1 of its
   retrieved neighbours has evidence; otherwise it is unscorable (`no_evidence_artifact_for_repo`,
   `no_retrieved_neighbour_with_evidence`). Point: grounded steps / steps over scorable rows
   (steps pooled; CI by bootstrap over rows, clustering steps by row). Secondary: share of scorable
   rows with >= 1 grounded step (Wilson); diagnostic: share of steps citing any retrieved neighbour.

5. Resolution vs actual.  Closed rows: actual_resolution_days is not None and meta.censored is not
   True; censored/open rows are excluded and counted (`censored`). Point estimate p = metadata
   .resolution_point_days when present, else the midpoint of the plan's lower/upper days (the
   report counts both sources). AE = |p - actual|. Reported: median AE and mean AE (bootstrap CIs),
   and the same for a constant naive reference (per-repo `naive_days`, k8s 35.1 d, vscode the train
   median 3.8374 d; constants are inputs, stated in the report), with the paired difference
   naive_AE - plan_AE (positive = the plan is closer; bootstrap CI). Interval coverage: share of
   closed rows with lower <= actual <= upper (Wilson) and mean width in days. A repo without a naive
   constant gets no reference row.

Nothing here writes or proposes a baseline or a gate: new baselines are proposed to GG, never
written (spec-quality-push.md section 10).
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

METRICS_VERSION = "qp5-objective-metrics-v1"
SEED = 42  # project-wide seed; every bootstrap below re-creates its generator from it
N_BOOT = 2000
PRIORITY_LEVELS = ("low", "medium", "high")


@dataclass(frozen=True)
class Pair:
    """One eval-set row joined to its recorded record (None when the row has no recorded plan)."""

    row: Mapping[str, Any]
    record: Mapping[str, Any] | None

    @property
    def plan(self) -> Mapping[str, Any]:
        return (self.record or {}).get("plan") or {}

    @property
    def metadata(self) -> Mapping[str, Any]:
        return (self.record or {}).get("metadata") or {}

    @property
    def meta(self) -> Mapping[str, Any]:
        return self.row.get("meta") or {}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def norm_label(label: object) -> str:
    """Lower-cased, whitespace-collapsed label (the comparison key of every exact match)."""
    return " ".join(str(label or "").split()).lower()


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """Wilson score interval for k successes in n trials; None when n == 0."""
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap_ci(
    n: int, stat: Callable[[np.ndarray], np.ndarray], n_boot: int = N_BOOT, seed: int = SEED
) -> tuple[float, float] | None:
    """95% percentile CI of `stat`, resampling n rows with replacement.

    `stat` receives the (n_boot, n) integer index matrix and returns n_boot statistics. Resamples
    that make the statistic undefined (NaN, e.g. an all-zero weight sum) are dropped.
    """
    if n <= 0:
        return None
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    with np.errstate(invalid="ignore", divide="ignore"):
        draws = np.asarray(stat(idx), dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0:
        return None
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _r(x: float | None) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), 4)


def _ci(ci: tuple[float, float] | None) -> list[float | None] | None:
    return None if ci is None else [_r(ci[0]), _r(ci[1])]


def _result(
    n: int,
    excluded: Mapping[str, int],
    point: float | None,
    ci: tuple[float, float] | None,
    method: str,
    **extra: Any,
) -> dict[str, Any]:
    """Uniform metric record: n, n_excluded (+ reasons), point, ci95, ci_method, extras."""
    out: dict[str, Any] = {
        "n": n,
        "n_excluded": int(sum(excluded.values())),
        "excluded_reasons": dict(sorted(excluded.items())),
        "point": _r(point),
        "ci95": _ci(ci),
        "ci_method": method,
    }
    out.update(extra)
    return out


def _proportion(hits: Sequence[bool], excluded: Mapping[str, int], **extra: Any) -> dict[str, Any]:
    n, k = len(hits), int(sum(hits))
    return _result(n, excluded, k / n if n else None, wilson(k, n), "wilson", k=k, **extra)


def _mean_boot(values: Sequence[float], excluded: Mapping[str, int], **extra: Any) -> dict:
    v = np.asarray(values, dtype=float)
    ci = bootstrap_ci(len(v), lambda idx: v[idx].mean(axis=1))
    return _result(len(v), excluded, float(v.mean()) if len(v) else None, ci, "bootstrap", **extra)


# ---------------------------------------------------------------------------
# Metrics (each takes the pairs of one population: pooled or one repo)
# ---------------------------------------------------------------------------


def _weight(meta: Mapping[str, Any]) -> float:
    w = meta.get("weight")
    return float(w) if isinstance(w, (int, float)) and not isinstance(w, bool) and w > 0 else 1.0


def component_metrics(pairs: Sequence[Pair]) -> dict[str, Any]:
    """Component exact match: strict (ADR-0070), k8s convention, in-family strict; see module doc."""
    excl: Counter[str] = Counter()
    rows: list[tuple[bool, bool, bool, float]] = []  # strict, lenient, in_family, weight
    derived_family = default_weight = 0
    for p in pairs:
        if p.record is None:
            excl["no_plan"] += 1
            continue
        gold = norm_label(p.row.get("gold_component"))
        if not gold:
            excl["no_gold_component"] += 1
            continue
        pred = norm_label(p.plan.get("predicted_component"))
        in_family = p.meta.get("in_family")
        if not isinstance(in_family, bool):
            in_family, derived_family = gold != "other", derived_family + 1
        default_weight += "weight" not in p.meta
        match = pred == gold
        rows.append((match and gold != "other", match, in_family, _weight(p.meta)))

    def variant(sel: Sequence[tuple[bool, bool, bool, float]], col: int) -> dict[str, Any]:
        hit = np.array([r[col] for r in sel], dtype=float)
        w = np.array([r[3] for r in sel], dtype=float)
        wp = float((w * hit).sum() / w.sum()) if len(sel) else None
        wci = bootstrap_ci(len(sel), lambda i: (w[i] * hit[i]).sum(axis=1) / w[i].sum(axis=1))
        unweighted = _proportion([bool(h) for h in hit], {})
        return {
            "unweighted": unweighted,
            "weighted": {"point": _r(wp), "ci95": _ci(wci), "ci_method": "bootstrap"},
        }

    return {
        "n_scored": len(rows),
        "n_excluded": int(sum(excl.values())),
        "excluded_reasons": dict(sorted(excl.items())),
        "in_family_source": "derived_gold_ne_other" if derived_family else "meta.in_family",
        "rows_with_derived_in_family": derived_family,
        "rows_with_default_weight_1": default_weight,
        "strict": variant(rows, 0),
        "k8s_convention": variant(rows, 1),
        "in_family_strict": variant([r for r in rows if r[2]], 0),
    }


def priority_metrics(pairs: Sequence[Pair]) -> dict[str, Any]:
    """Priority accuracy on label-sourced gold only; the majority class is the reference row."""
    excl: Counter[str] = Counter()
    gold_l: list[str] = []
    hits: list[bool] = []
    for p in pairs:
        if p.record is None:
            excl["no_plan"] += 1
            continue
        source = p.meta.get("gold_priority_source")
        gold = norm_label(p.row.get("gold_priority"))
        if source != "label":
            excl[f"gold_source_{source}" if source else "gold_source_unknown_legacy"] += 1
        elif gold not in PRIORITY_LEVELS:
            excl["gold_not_3_level"] += 1
        else:
            gold_l.append(gold)
            hits.append(norm_label(p.plan.get("priority_guess")) == gold)
    out = _proportion(hits, excl)
    if gold_l:
        counts = Counter(gold_l)
        label, count = counts.most_common(1)[0]
        out["gold_counts"] = dict(sorted(counts.items()))
        out["majority_reference"] = {
            "label": label,
            "accuracy": _r(count / len(gold_l)),
            "ci95": _ci(wilson(count, len(gold_l))),
        }
    return out


# ---------------------------------------------------------------------------
# Assembly and loaders
# ---------------------------------------------------------------------------


def join_records(
    rows: Sequence[Mapping[str, Any]], records: Mapping[str, Mapping[str, Any]]
) -> list[Pair]:
    """Join eval rows to recorded records by id (a row with no record gets None)."""
    return [Pair(r, records.get(str(r.get("id")))) for r in rows]


def _is_degraded(p: Pair) -> bool:
    status = str(p.metadata.get("llm_status") or "")
    return status == "parse_failure" or status.startswith("degraded_")


def score_population(
    pairs: Sequence[Pair],
    evidence: Mapping[str, Mapping[int, bool]],
    naive_days: Mapping[str, float] | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    """All five metrics for one population."""
    from triage_iq.evaluation.objective_metrics_retrieval import (
        grounding_metrics,
        relevance_metrics,
        resolution_metrics,
    )

    return {
        "n_rows": len(pairs),
        "n_with_plan": sum(p.record is not None for p in pairs),
        "n_degraded_plans": sum(p.record is not None and _is_degraded(p) for p in pairs),
        "component": component_metrics(pairs),
        "priority": priority_metrics(pairs),
        "similar_issue_relevance": relevance_metrics(pairs, k),
        "next_step_grounding": grounding_metrics(pairs, evidence),
        "resolution": resolution_metrics(pairs, naive_days),
    }


def score_all(
    pairs: Sequence[Pair],
    evidence: Mapping[str, Mapping[int, bool]] | None = None,
    naive_days: Mapping[str, float] | None = None,
    k: int | None = None,
) -> dict[str, Any]:
    """Pooled and per-repo scores. Pure: no I/O, no randomness beyond the seeded bootstrap."""
    ev = evidence or {}
    repos = sorted({str(p.row.get("repo")) for p in pairs})
    return {
        "metrics_version": METRICS_VERSION,
        "bootstrap": {"resamples": N_BOOT, "seed": SEED},
        "pooled": score_population(pairs, ev, naive_days, k),
        "by_repo": {
            repo: score_population(
                [p for p in pairs if str(p.row.get("repo")) == repo], ev, naive_days, k
            )
            for repo in repos
        },
    }


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    """Read a recorded-plans JSONL: one {"id", "plan", "metadata"} object per line."""
    out: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            out[str(rec["id"])] = rec
    return out


def load_evidence_lookup(path: Path) -> dict[int, bool]:
    """number -> has_evidence from a resolution-evidence parquet (candidate artifact)."""
    import pandas as pd

    df = pd.read_parquet(path, columns=["number", "has_evidence"])
    return {int(n): bool(h) for n, h in zip(df["number"], df["has_evidence"], strict=True)}


_MOVED = frozenset(
    {
        "NAIVE_DAYS_DEFAULT",
        "cited_numbers",
        "grounding_metrics",
        "relevance_metrics",
        "resolution_metrics",
    }
)


def __getattr__(name: str) -> Any:
    """Keep the public surface: the retrieval/resolution metrics moved to a sibling module.

    Lazy (PEP 562) because the sibling imports this module's helpers; an eager re-export would be
    a circular import whichever module is imported first.
    """
    if name in _MOVED:
        from triage_iq.evaluation import objective_metrics_retrieval

        return getattr(objective_metrics_retrieval, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
