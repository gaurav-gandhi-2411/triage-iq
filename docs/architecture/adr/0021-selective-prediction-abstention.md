# ADR-0021 — Selective Prediction / Abstention

**Status:** Rejected — documented negative result for v1. Measured, escalated, and not shipped:
component confidence is a real but marginal, noisy signal (flag-gated off, deferred as a
product-value call); resolution interval width does not predict coverage failure and is
rejected outright; priority was correctly never attempted.
**Date:** 2026-07-09 (measured); decided 2026-07-10
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

**Build and measure, then reject shipping any live default for v1.** A deterministic, additive
**selective-prediction gate** over existing signals — not a new model, not a synthesis-prompt
change — was built, measured against the full coverage-vs-abstention curve, and the result did
not clear a bar worth shipping as a live default. This follows the same honest-negative pattern
as ADR-0006 (cross-encoder reranker) and the W3 fine-tuned-retriever rejection: measured,
reported precisely, not shipped, kept in the codebase (flag-gated off) for a future revisit
rather than deleted. Three stage decisions, approved before building:

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

### Why the curve alone wasn't enough to decide, and what settled it

The initial pick rule (maximize accuracy/coverage-on-answered subject to abstention rate ≤ 50%)
landed at the edge of that ceiling on **both** k8s curves (48.1% and 50.0% abstention) — a red
flag on its own: a "best" point sitting at the boundary of an arbitrary cap, on a curve that is
visibly non-monotonic in the 30–50% range (e.g. component accuracy-on-answered *drops* from
58.97% at 27.8% abstention to 57.14% at 35.2%, then rises again), is a symptom of a simple
max-over-a-noisy-curve rule overfitting to sampling noise on n=54 binary outcomes, not
necessarily a real effect concentrated at high abstention. That suspicion was checked directly
against the per-issue data (not assumed) — see below.

**Component stage — checked directly, signal confirmed real:** mean `component_confidence` for
correctly-predicted issues is **0.548** vs **0.430** for incorrectly-predicted issues (k8s,
n=27/27) — a genuine +0.118 gap, consistent with the classifier actually being calibrated
(ADR-0004). The lower-abstention region of the curve (9–28% abstention, n_answered 39–49) shows
real, modest gains (+5.1 to +8.97pp) before the curve gets noisy. **Verdict: real but marginal
signal.** Shippable only as a deliberate product-value tradeoff (is a wrong component prediction
costly enough to justify trading ~10–28% coverage for a 5–9pp accuracy lift?) — that call is
deferred, not decided here. Kept flag-gated off; default is no abstention.

**Resolution stage — checked directly, initial hypothesis rejected, correct diagnosis found:**
the working hypothesis going in was that the conformal interval might be near-constant across
issues (the CQR scalar `Q` is tiny — 0.0118 days for k8s, per `reports/eval_summary.json` — so
if the base quantile-regression interval were also roughly fixed, there'd be nothing per-issue to
threshold on). **This was tested directly and is false:** conformal interval width varies
substantially per issue on k8s — min 6.91 days, max 153.72 days, mean 90.63, stdev 41.98
(coefficient of variation 0.463). The base LightGBM quantile model **is** producing
issue-adaptive intervals; `Q`'s near-zero contribution doesn't make the total width constant.

**The actual reason resolution abstention doesn't work: width doesn't predict coverage
failure.** Mean width for issues where the interval *did* cover the true value is **90.45
days** (n=36); mean width for issues where it *missed* is **91.01 days** (n=18) — a 0.56-day gap
against a ~42-day standard deviation, i.e. statistically indistinguishable. An interval's width
carries no information about whether *that specific interval* is right. This is precisely why
the sweep curve's apparent +7–11pp coverage gains at 35–50% abstention are not trustworthy: if
width doesn't correlate with coverage failure, then abstaining on high-width issues is
close to abstaining on a **random** subset, and the curve's rises and dips are the sampling
noise of shrinking an already-small n=54, not a real effect. **Verdict: rejected outright** —
not "the intervals aren't adaptive enough," but "the interval's own width is not a useful
per-issue uncertainty signal for *this specific failure mode* (coverage)." A genuinely useful
resolution-abstention signal needs a measure that actually correlates with getting the interval
wrong — this one doesn't, and that is the finding, not an implementation gap to patch.

**This is the interesting thread for any future resolution-abstention work:** the prerequisite
isn't "make the interval per-issue-adaptive" (it already is, width-wise) — it's finding or
building an uncertainty signal that is *discriminative of actual coverage failure*, which width
is not. That is a separate, larger investigation (likely requires examining what covariates
*do* predict coverage misses, or a fundamentally different uncertainty quantification approach),
not a tweak to this build.

### Priority stage

Out of scope for v1 — no calibrated confidence signal exists for `priority_guess` anywhere in
the pipeline (unlike `component_confidence` or the CQR interval). Flagged as a gap, not gated
with a fabricated proxy. Correctly not attempted.

### Verdict

**Rejected for v1.** Selective prediction via simple confidence/width thresholds does not clear
a bar worth shipping: component is real-but-marginal (a deferred product-value call, not a
technical win); resolution is not usable (width is non-discriminative of coverage failure,
confirmed directly against the data); priority was never attempted for lack of any signal. No
live default is shipped. The gate is implemented and tested but **flag-gated off**
(`TRIAGE_ENABLE_ABSTENTION_GATE`, unset/off by default) in
`src/triage_iq/api/app.py` — `abstention_status` stays `None` on every `/triage` response unless
explicitly turned on.

## Consequences

- **Nothing ships live.** `TRIAGE_ENABLE_ABSTENTION_GATE` defaults off — `plan.abstention_status`
  is `None` on every `/triage` response in every environment, including if this branch is merged
  to `main`, until someone explicitly sets that env var. This is a deliberate consequence of the
  rejection, not an oversight.
- `TriagePlan.abstention_status` and `AbstentionStatus`/`StageAbstention` remain in the schema
  additively (`None`-safe) — kept, not deleted, so `scripts/measure_abstention_tradeoff.py` and
  `tests/test_abstention.py` continue to exercise real code, and any future revisit (per the
  resolution-stage prerequisite above) has a working starting point instead of a rewrite.
- `reports/eval_baseline.json` and `eval/cassettes/eval_cassette.json` are unaffected — this was
  always a scoring-only, post-hoc gate, never a synthesis or judge-scoring change.
- The thresholds in `src/triage_iq/models/abstention.py`
  (`COMPONENT_CONFIDENCE_THRESHOLD`, `RESOLUTION_WIDTH_THRESHOLD_DAYS`) are retained as a record
  of what was measured, not as a shipped default — component's threshold is a candidate for a
  future product-value decision; resolution's threshold should not be reused without first
  finding a width-independent, coverage-discriminative signal (see "the interesting thread"
  above).
- **`COMPONENT_CONFIDENCE_THRESHOLD` is now STALE, effectively dead (ADR-0036, 2026-07-24).**
  These values (k8s=0.45, vscode=0.29) were tuned against the single-label classifier's
  calibrated confidence distribution. ADR-0036 replaced the shipped classifier with a
  multi-label one-vs-rest model with a different confidence stream (independent per-class
  sigmoids, recalibrated with its own temperature) — under the new distribution the SAME fixed
  thresholds fire at a wildly different rate on the held-out test set: **k8s 59.8% → 0.0%,
  vscode 13.9% → 0.5%** (see `reports/tfidf_multilabel_calibration_and_threshold_check.json`).
  This gate has stayed off the whole time this measured, so nothing in production changed — but
  **these constants MUST be re-derived from the new confidence distribution before
  `TRIAGE_ENABLE_ABSTENTION_GATE` is ever set to `1`.** Flipping that flag today would enable a
  gate calibrated to a confidence stream that no longer exists — silently dead (fires ~never on
  k8s) rather than doing what this ADR's own tradeoff analysis intended.
- Generalization caveat: these numbers describe this exact prompt/model pair (Groq
  `llama-3.1-8b-instant`, no attribution — `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION` off) and this
  exact gold set. A future prompt or model change requires re-measurement.
- vscode's n=11 curves (both stages) are reported as indicative-only per ADR-0017's data-ceiling
  finding — not weighted equally with k8s's n=54 in the verdict above.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Train a new uncertainty/confidence model | Unnecessary — every needed signal already exists in the shipped pipeline; a new model adds cost and a new failure surface for no measured benefit over thresholding what's already calibrated. |
| Gate priority with a proxy confidence (e.g. component-confidence agreement) | Would be a fabricated signal with no measured basis — inconsistent with this project's "no invented numbers" discipline (mirrors ADR-0006/ADR-0016's rejection of unmeasured hypotheses). Flagged as a gap instead. |
| Pick threshold defaults without a sweep | Would repeat the exact mistake ADR-0019's mean-band gate was built to avoid — a threshold "that feels right" isn't a data-derived one. The full curve was computed and is reported so the human picks the actual number. |
| Single overall threshold across repos | k8s and vscode have different baseline accuracy (50.0% vs 18.18%) and different curve shapes — a shared threshold would be tuned to whichever repo dominates by volume (k8s, 54/65), silently under- or over-abstaining on the other. Same per-repo discipline as the eval quality gate (ADR-0019). |
| Ship the curve-max default as-is (component 0.45 / resolution 91.42 days) | Rejected on inspection, not just on principle: both curve-max points sit at the edge of the abstention-rate ceiling on a visibly non-monotonic curve, and the resolution one is explained away by a direct check — mean interval width is statistically indistinguishable between covered (90.45d) and uncovered (91.01d) issues, so the apparent gain is consistent with noise from shrinking n, not a real effect. |
| "Intervals are near-constant, so nothing to threshold" (initial hypothesis) | Tested directly and rejected — k8s conformal width varies 6.91–153.72 days (CV=0.463), i.e. substantially per-issue. The real problem is that this variation doesn't predict coverage failure, a different and more specific diagnosis. |
