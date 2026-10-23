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
