# ADR-0021 — Selective Prediction / Abstention

**Status:** Proposed — measured, gate-1 human review pending (tradeoff curves + default
operating point)
**Date:** 2026-07-09
**Decider:** Gaurav Gandhi

## Context

Roadmap item #2: extend the shipped conformal/calibration work so the system can honestly say
"low confidence" or "abstaining" on a per-issue basis instead of always answering. Chosen over
other candidates because it does not hit the vscode data ceiling (ADR-0017) the same way a new
model would, builds directly on already-shipped conformal (ADR-0010) and calibration (ADR-0004)
work rather than requiring one, and is demoable: a weak-signal issue produces an honest
"abstained" flag instead of a guess presented with false confidence.

**Investigation finding (reported before building, per the escalation gate):** every signal
selective prediction needs already exists and is already exposed on `/triage` —
`component_confidence` (calibrated, ADR-0004), `resolution_confidence_pct` (bucket classifier),
`resolution_interval_conformal` (CQR, ADR-0010), `grounding_status` (ADR-0015). Nothing here
required a new model. `priority_guess` was the one exception: no calibrated confidence signal
exists for it anywhere in the pipeline.

## Decision

Ship a deterministic, additive **selective-prediction gate** over these existing signals — not
a new model, not a synthesis-prompt change. Three stage decisions, approved before building:

1. **Component stage:** abstain when `component_confidence` is below a per-repo threshold, OR
   `grounding_status.component_grounded is False` (hard trigger, not swept — a component the
   classifier's own top-3 doesn't support abstains regardless of confidence).
2. **Resolution stage:** abstain when the CQR-adjusted interval's width exceeds a per-repo
   threshold — "too wide to be useful" is the system honestly admitting it can't predict, which
   is exactly what the CQR work was built to enable.
3. **Priority stage:** explicitly **out of scope for v1**. Flagging the gap (no calibrated
   confidence signal exists for `priority_guess`) rather than inventing a proxy threshold with
   no measured basis — same discipline this project applies everywhere else (no fabricated
   numbers).

**Both thresholds are derived from data, not picked.** `scripts/measure_abstention_tradeoff.py`
replays the clean cassette (`eval/cassettes/eval_cassette.json`, flag off — see ADR-0020) over
all n=65 issues, computes the per-issue signals, and sweeps every observed threshold value,
producing the full coverage-vs-abstention curve per stage per repo. The **curve is the result**,
not a single number — see Results below for the full tables and the proposed default operating
point picked from them.

**Scoring-only, no re-record.** This gate doesn't touch the synthesis prompt or judge
scoring — it reads already-recorded plan fields (`component_confidence`) and already-computed
deterministic checks (`grounding_status`, the conformal-interval formula already used by
`/triage`). `eval/cassettes/eval_cassette.json` and `reports/eval_baseline.json` are both
unchanged.

**Additive schema:** `AbstentionStatus` (`component: StageAbstention`, `resolution:
StageAbstention`, each `{abstained: bool, reason: str}`) lands as `TriagePlan.abstention_status`,
`None`-safe, computed in `src/triage_iq/api/app.py` right after the conformal-interval block
(fail-open — an exception there never blocks the response, same policy as conformal). The eval
harness (`eval/run_eval.py`) never populates it (it calls `triage_with_metadata()` directly,
bypassing `app.py`'s post-processing) but the field still serializes on every `TriagePlan`
(`None` by default) — excluded from the judge's `plan_json`, same pattern already used for
`declared_attribution` (ADR-0020), to keep the judge cache key matching the unchanged baseline.

## Results

n=65 (vscode 11, kubernetes/kubernetes 54), clean pre-attribution cassette, zero live calls.
Full curves: `reports/abstention_tradeoff.json`.

### Component stage

**kubernetes/kubernetes (n=54)** — baseline accuracy with no abstention: **50.0%**.

| confidence threshold | abstention rate | n answered | accuracy on answered |
|---|---|---|---|
| 0.00 (baseline) | 0% | 54 | 50.0% |
| 0.22 | 27.8% | 39 | 58.97% (+8.97pp) |
| 0.30 | 35.2% | 35 | 57.14% (+7.14pp) |
| **0.45 (proposed default)** | **48.1%** | **28** | **60.71% (+10.71pp)** |
| 0.82 | 79.6% | 11 | 72.73% (+22.73pp) |

**microsoft/vscode (n=11, INDICATIVE ONLY)** — baseline accuracy: **18.18%** (2/11 correct).
Proposed default (threshold 0.29): 27.3% abstention, n=8 answered, 25.0% accuracy (2/8) — a
single issue moving is a ±12.5pp swing on the answered subset. This curve is reported for
completeness, not as a measured result on par with k8s — see ADR-0017.

### Resolution stage

**kubernetes/kubernetes (n=54)** — baseline coverage with no abstention: **66.67%** (measured
on this 54-issue subset; the CQR calibration file's 76.6% empirical coverage was measured on a
separate, much larger 1049-issue held-out set — the two numbers are not expected to match and
the gap is ordinary sampling variation on a smaller subset, not a discrepancy).

| width threshold (days) | abstention rate | n answered | coverage on answered |
|---|---|---|---|
| max (baseline) | 0% | 54 | 66.67% |
| 108.32 | 35.2% | 35 | 74.29% (+7.62pp) |
| 94.62 | 44.4% | 30 | 76.67% (+10.0pp) |
| **91.42 (proposed default)** | **50.0%** | **27** | **77.78% (+11.11pp)** |
| 73.62 | 74.1% | 14 | 71.43% (+4.76pp, noisy tail) |

**microsoft/vscode (n=11, INDICATIVE ONLY)** — baseline coverage: **63.64%**. Proposed default
(width 195.20 days): 36.4% abstention, n=7 answered, 71.43% coverage (+7.79pp, one issue = ±14pp
on the answered subset). Indicative only, same caveat as component stage.

### How the proposed default was picked, and why it isn't the only reasonable choice

Rule: maximize accuracy/coverage-on-answered subject to abstention rate ≤ 50% (a ceiling — not
derived from the data — chosen because abstaining on a majority of issues arguably stops being
"selective" prediction), ties broken toward the lower abstention rate.

**Honest caveat:** on k8s, this rule landed at the edge of that ceiling for *both* stages (48.1%
and 50.0% abstention). Both are real, data-derived improvements (+10.71pp / +11.11pp) — but the
tables above show materially cheaper points already on the curve: 0.22 confidence threshold
(27.8% abstention, +8.97pp — nearly as much accuracy gain for about half the abstention cost)
and 108.32-day width threshold (35.2% abstention, +7.62pp). The full curve is in
`reports/abstention_tradeoff.json`; the "proposed default" above is one candidate from a simple
rule, not the only sensible operating point. **This is the escalation** — confirm which point to
ship before it is treated as final.

### Priority stage

Out of scope for v1 — no calibrated confidence signal exists for `priority_guess` anywhere in
the pipeline (unlike `component_confidence` or the CQR interval). Flagged as a gap, not gated
with a fabricated proxy.

## Consequences

- `TriagePlan.abstention_status` lands additively on `/triage` responses — `None`-safe, no
  existing field changes shape or meaning.
- `reports/eval_baseline.json` and `eval/cassettes/eval_cassette.json` are unaffected — this is
  a scoring-only, post-hoc gate, not a synthesis or judge-scoring change.
- The thresholds in `src/triage_iq/models/abstention.py`
  (`COMPONENT_CONFIDENCE_THRESHOLD`, `RESOLUTION_WIDTH_THRESHOLD_DAYS`) are **provisional** —
  sourced from the proposed-default picks above, explicitly pending human confirmation before
  being treated as the shipped default. Nothing deploys until that confirmation and a separate
  deploy decision.
- Generalization caveat: these numbers describe this exact prompt/model pair (Groq
  `llama-3.1-8b-instant`, no attribution — TRIAGE_PROMPT_INCLUDE_ATTRIBUTION off) and this exact
  gold set. A future prompt or model change requires re-measurement.
- vscode's n=11 curves (both stages) are reported as indicative-only per ADR-0017's data-ceiling
  finding — not weighted equally with k8s's n=54 in any default-picking decision.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Train a new uncertainty/confidence model | Unnecessary — every needed signal already exists in the shipped pipeline; a new model adds cost and a new failure surface for no measured benefit over thresholding what's already calibrated. |
| Gate priority with a proxy confidence (e.g. component-confidence agreement) | Would be a fabricated signal with no measured basis — inconsistent with this project's "no invented numbers" discipline (mirrors ADR-0006/ADR-0016's rejection of unmeasured hypotheses). Flagged as a gap instead. |
| Pick threshold defaults without a sweep | Would repeat the exact mistake ADR-0019's mean-band gate was built to avoid — a threshold "that feels right" isn't a data-derived one. The full curve was computed and is reported so the human picks the actual number. |
| Single overall threshold across repos | k8s and vscode have different baseline accuracy (50.0% vs 18.18%) and different curve shapes — a shared threshold would be tuned to whichever repo dominates by volume (k8s, 54/65), silently under- or over-abstaining on the other. Same per-repo discipline as the eval quality gate (ADR-0019). |
