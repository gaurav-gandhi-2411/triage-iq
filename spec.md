# Project Spec: TriageIQ — Diagnostic Resolution Uncertainty (signal search)

## Goal

The resolution model produces per-issue conformal intervals whose WIDTH varies (6.91–153.72 days,
CV=0.463) but does NOT correlate with whether the interval actually covers the true value: mean
width on covered issues (90.45d) is statistically indistinguishable from mean width on missed
issues (91.01d) — ADR-0021. So the model expresses uncertainty that is NOT DIAGNOSTIC of its own
errors: a wide interval is no more likely to be wrong than a narrow one.

This iteration is an OPEN SEARCH for a signal — any available signal — that actually CORRELATES
with resolution coverage failure, so the system could know when its resolution estimate is likely
wrong (which would, in turn, make resolution abstention meaningful, reversing the ADR-0021
rejection). This is a research build, not a feature build. **The honest outcome may be "no available
signal correlates with resolution error" — that is a valid, documented result, not a failure.** The
deliverable is the systematic search + the correlation findings (positive or negative), not a
shipped feature.

## Current state (existing project)

- Resolution: LightGBM quantile + bucket + CQR conformal (de-leaked, ADR-0009/0010). Produces
  `expected_resolution_lower/upper_days`, `resolution_interval_conformal` (with empirical_coverage),
  `resolution_confidence_pct`, `resolution_bucket`.
- Eval: clean n=65 (54 k8s / 11 vscode), local qwen3:8b judge, live rev 00061-4xk (+ pending
  structured-verification merge). Baseline cassette_hash c9966414.
- The resolution ground truth (actual close time) is known for the eval issues (that's how coverage
  is measured). So "did the interval cover" is a computable label per issue — the target of the search.
- ADR-0021 established: interval WIDTH is not diagnostic. This build searches OTHER signals.

## Scope

### In scope

**Define the target label (the thing signals must predict):**
- Per issue, the binary "coverage failure" label: did the conformal interval FAIL to cover the true
  resolution time? (miss = 1, cover = 0). This is computable from existing data. Also consider a
  continuous target: the signed/absolute error magnitude (how far off the point estimate was).

**Candidate signals to test (systematic — test each, report each):**
Search across signals ALREADY AVAILABLE per issue (no new models unless an ensemble is cheap):
1. Classifier confidence / component_confidence on the issue.
2. Retrieval quality: similarity scores of the nearest neighbors (mean/max/top-k spread) — does a
   "no good neighbors" issue resolve less predictably?
3. Issue text features: length, code-block presence, title/body length, token count.
4. Component identity: does resolution error correlate with WHICH component (some components
   resolve more predictably)?
5. Raw quantile spread BEFORE conformal adjustment (the base LightGBM lower/upper spread).
6. Resolution bucket itself (do "months"/"long" bucket issues miss more than "hours"?).
7. Grounding status / similar-issue count.
8. (If cheap) ensemble disagreement: if multiple resolution models or quantile levels disagree,
   is that diagnostic? Only if it doesn't require retraining a heavy model.

**The search (honest statistics — this is the whole point):**
- For each candidate signal, measure its correlation with the coverage-failure label AND with error
  magnitude. Use appropriate stats: point-biserial / logistic-regression coefficient for the binary
  label, Spearman/Pearson for continuous error, each with a confidence interval or p-value.
- The bar (same discipline as ADR-0006's reranker rejection): a signal is DIAGNOSTIC only if its
  correlation is real (CI excludes zero / p below a stated threshold) AND meaningful (effect size,
  not just significance on a lucky sample). Report effect size, not just significance.
- n matters: k8s n=54 is where the search is powered; vscode n=11 is INDICATIVE-ONLY (report but
  don't claim). State per-repo, don't pool.
- Multiple-comparisons honesty: testing ~8 signals means ~8 chances for a spurious hit. Apply a
  correction (Bonferroni or FDR) or at minimum state that k signals were tested so a single p<0.05
  isn't overread. This matters — with 8 signals, one crossing p<0.05 by chance is likely.

**Outcome (either is valid, documented in the ADR):**
- POSITIVE: one or more signals correlate with coverage failure (real + meaningful, survives
  multiple-comparison correction). Then: propose (don't build yet) how it would enable resolution
  abstention, and escalate whether to build v2 abstention on it.
- NEGATIVE: no available signal correlates. Documented finding: "resolution error is not predictable
  from available per-issue features on this data" — a genuine result that closes the ADR-0021 thread
  honestly and tells you resolution abstention isn't achievable without new features/data.

### Out of scope

- No retraining the resolution model, no new heavy models (an ensemble is fine ONLY if it's cheap
  quantile-level disagreement, not a new fit).
- No building resolution abstention in THIS iteration — this SEARCHES for the signal; building v2
  on a found signal is a separate escalated decision.
- No change to the live pipeline or schema (this is analysis, scoring-only, no re-record).
- No reopening closed eval-integrity work.
- No claiming a vscode result at n=11 (indicative-only).

## Tech stack

- Existing Python + scipy/statsmodels for the correlation stats (numpy/scipy likely already present;
  statsmodels if needed for logistic regression + CIs — escalate if a new dep). Analysis over the
  existing n=65 cassette + resolution ground truth. No LLM, no re-record.

## Architecture

```
triage-iq/
  scripts/analyze_resolution_diagnosticity.py   # NEW — the signal search + stats
  reports/resolution_diagnosticity.json          # NEW — per-signal correlation results
  docs/architecture/adr/ADR-0023-*.md            # the search, per-signal findings, verdict
```
(No src/ changes, no schema changes — this is analysis. If a signal is found and v2 abstention is
later approved, THAT build touches src/.)

## Autonomy & escalation (CC runs autonomously — escalate ONLY these)

CC decides + executes: the exact candidate-signal list (may add signals beyond the 8 if sensible),
the statistical methods, the multiple-comparison correction, the analysis, the ADR.
Escalate ONLY:
1. **The search results** — the per-signal correlation table + the positive/negative verdict.
   Report before writing the final verdict, so the ship/build-v2/close-as-negative decision is
   human-made.
2. A new dependency (statsmodels etc.) if needed.
(No prod deploy in this build — it's analysis. If a signal is found, building v2 abstention is a
separate future spec with its own escalations.)

## Hard rules

- Honest statistics: report effect size + CI/p, apply multiple-comparison correction for ~8 signals,
  state n per repo, vscode n=11 is indicative-only. A single uncorrected p<0.05 out of 8 tests is
  NOT a finding — say so.
- The NEGATIVE outcome (no signal correlates) is a VALID, publishable result — frame it as a finding,
  not a failure. Do not fish for a spurious positive to avoid a negative.
- No retraining, no heavy new models, no schema/pipeline change (analysis only).
- Zero-cost (no LLM, no re-record — pure analysis on existing data). Branch only, I merge.
  Claude Max — never ANTHROPIC_API_KEY. Don't touch aetherart-497918.

## Verification commands

```yaml
- name: api-tests
  cmd: pytest -v
  required: true
- name: eval-gate
  cmd: pytest eval/ -v
  required: true
```
(The analysis script has its own reproducibility check: re-running produces byte-identical
resolution_diagnosticity.json.)

## Success criteria (CC verifies before reporting)

- Coverage-failure label + error-magnitude target computed per issue from existing data.
- Every candidate signal tested; per-signal correlation (effect size + CI/p) reported, per repo.
- Multiple-comparison correction applied and stated; vscode marked indicative-only (n=11).
- Verdict: which signal(s) if any are diagnostic (real + meaningful + survives correction), or the
  honest negative (none are).
- `resolution_diagnosticity.json` reproduces byte-identically on re-run.
- ADR-0023 written with the full per-signal table and the verdict framed as a finding either way.
- Staged on branch; nothing prod-facing (analysis only).

## Build order (CC autonomous)

1. Compute the coverage-failure label + error-magnitude target per issue (k8s n=54, vscode n=11).
2. Extract each candidate signal per issue from the existing cassette / pipeline outputs.
3. Run the correlation analysis per signal per repo: effect size + CI/p, multiple-comparison
   correction across the signal set.
4. ESCALATE: report the per-signal correlation table + the positive/negative verdict before
   finalizing.
5. ADR-0023 with the findings. If POSITIVE, note (don't build) how it would enable resolution
   abstention v2 as a separate future decision. If NEGATIVE, close the ADR-0021 thread: resolution
   error not predictable from available features on this data.
```

