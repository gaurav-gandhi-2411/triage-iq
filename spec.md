# Project Spec: TriageIQ — Resolution Model Improvement (Phase 1: model performance)

## Goal

The resolution stage is the pipeline's weakest: it LOSES to a naive prior on vscode (~-70%),
barely beats it on k8s (~+2%), and ADR-0021/0023 proved its uncertainty is non-diagnostic and its
errors are unpredictable from currently-extracted features. That last finding is the KEY INSIGHT:
resolution error being unpredictable from current features strongly suggests the current feature
set UNDER-DETERMINES resolution time — the model doesn't have the inputs it needs. This iteration
attacks that directly: engineer better resolution features and/or reframe the resolution task to
what's actually predictable, to genuinely reduce error (not to abstain on it — that's proven
impossible).

**This is real modeling work with a real chance of an honest negative** (maybe resolution time on
software issues genuinely isn't more predictable from available data — which would itself be a
strong finding, proven rigorously). The bar is ADR-0006's: an improvement ships only if it beats
the current model with a CI that excludes zero on the powered repo (k8s). No p-hacking, no
shipping a marginal gain as a win.

## Current state (existing project)

- Resolution model: LightGBM quantile + bucket classifier + CQR conformal, de-leaked (ADR-0009:
  created_at temporal split, has_priority and other post-creation features dropped). Current
  performance: k8s ~+2% over naive (MAE), vscode ~-70% (loses to naive), conformal coverage
  k8s 76.6% / vscode 74.1%.
- Feature set: currently creation-time features only (post-leak-fix). The de-leaking correctly
  removed post-creation signal — but may have left the model with too few creation-time features
  to predict well. That's the tension to explore: MORE creation-time features (not post-creation).
- Ground truth: actual resolution time (created_at → closed_at) known for gold + training issues.
- Eval: clean n=65, local qwen3:8b judge, mean-band gate. Resolution metrics via
  resolution_results.json (separate held-out pathway from the judge eval — the honest "beats naive"
  number).
- vscode is data-starved (n=11 eval, 411 pairs) — its resolution result is indicative; k8s (n=54)
  is where improvement is measured with power.

## Scope

### In scope

**Diagnose WHY resolution is hard (before building — informs the approach):**
- Given ADR-0023 (no feature predicts error), first characterize: is the error high-variance
  (irreducible noise — issues genuinely resolve unpredictably) or high-bias (the model is
  systematically wrong in a fixable way)? Plot predicted vs actual, residual distribution, per
  bucket. This decides whether MORE FEATURES help (bias) or whether resolution is fundamentally
  noisy (variance — in which case the honest move is task reframing, not feature engineering).

**Feature engineering (creation-time only — do NOT reintroduce leakage):**
- Engineer additional CREATION-TIME features that could carry resolution signal without leaking:
  issue text embeddings (semantic content → resolution signal), title/body length + complexity,
  code-block/stacktrace presence, reporter history IF available at creation time (careful: must be
  known at creation, not after), label-at-creation, component, cross-reference count at creation.
- HARD LEAKAGE GUARD: every new feature must be verifiable as known at issue CREATION time. Assert
  disjointness/no-leakage the same way ADR-0009 established. A feature that sneaks in post-creation
  signal would inflate the result — the exact bug this project has caught repeatedly. Each new
  feature gets an explicit "known at creation?" justification.

**Task reframing (if point-estimation is fundamentally hard):**
- If diagnosis shows resolution TIME is high-variance/unpredictable, test whether a COARSER target
  is predictable: bucket classification (hours/days/months/long) may be achievable where
  point-days aren't. Or ordinal (faster/slower than median). A well-calibrated bucket classifier
  that beats naive is more useful than a point estimate that loses to naive. Measure this as an
  alternative framing.

**Honest evaluation (the whole point):**
- Retrain the resolution model with the new features / reframed target. Measure vs the CURRENT
  model on the held-out pathway (resolution_results.json), per repo. Improvement ships ONLY if the
  CI on the improvement excludes zero on k8s (ADR-0006 bar). Report effect size + CI.
- vscode: report but indicative-only (n=11); don't claim, don't pool.
- If retrained model changes /triage outputs → it's a model artifact change → GCS upload + manifest
  update + re-record + re-baseline (the full deliberate cutover with drift guard — this is a REAL
  model change, treat it like the bucket-retrain cutover).

### Out of scope

- No reintroducing post-creation features (ADR-0009 leakage — the hard line).
- No resolution abstention (ADR-0021/0023 proved it unbuildable — this reduces error instead).
- No change to classifier/retriever/synthesis (resolution stage only).
- No shipping a marginal gain that doesn't clear the ADR-0006 CI-excludes-zero bar.
- No reopening closed eval-integrity work.

## Tech stack

- Existing LightGBM + Python. Embeddings via the existing BGE model (already loaded) if text
  features are used. scipy/statsmodels for the significance test. No new heavy deps without escalation.

## Architecture

```
triage-iq/
  scripts/w6_diagnose_resolution.py       # NEW — bias/variance characterization
  scripts/w6_resolution_features.py       # NEW — creation-time feature engineering + leakage guard
  scripts/w6_train_resolution.py          # retrain w/ new features or reframed target
  reports/w6_resolution_results.json      # NEW — new vs current, CI, per repo
  data/models/resolution_predictor_*.pkl  # RETRAINED artifacts (cutover if it ships)
  data/models/MANIFEST.sha256             # updated if artifacts change
  docs/architecture/adr/ADR-0024-*.md     # diagnosis, approach, result, ship/reject decision
```

## Autonomy & escalation

CC runs diagnosis + feature engineering + retraining + evaluation autonomously. Escalate ONLY:
1. **The diagnosis result** (bias vs variance) — because it decides the approach (feature eng vs
   task reframe). Report before committing to one.
2. **The improvement result** (new vs current, CI per repo) — before any ship/cutover decision. The
   ship decision (does it clear the ADR-0006 bar) is human-confirmed.
3. **The model cutover** (GCS upload + manifest + re-record + re-baseline + deploy) — the full
   deliberate cutover, escalated, if the improvement ships.
4. Any new leakage-risk feature where "known at creation?" is ambiguous — escalate rather than assume.

## Hard rules

- LEAKAGE: every new feature verifiably known at issue CREATION time; assert it; escalate ambiguity.
  This is the ADR-0009 line — do not cross it. A post-creation feature inflating the result is THE
  failure mode this project exists to catch.
- Ship only on ADR-0006 bar (CI excludes zero on k8s). A negative (no feature/reframe beats current)
  is a VALID finding — document it, don't force a positive.
- vscode n=11 indicative-only; k8s n=54 is the powered measurement; don't pool.
- A model artifact change = full deliberate cutover (backup, GCS upload, manifest, re-record,
  re-baseline, drift guard, rollback anchor, live verify) — treat like the bucket-retrain.
- Branch only (`feat/w6-resolution-improvement`); I merge. Zero-cost (local for any eval judge
  re-record). Claude Max — never ANTHROPIC_API_KEY. Don't touch aetherart-497918.

## Verification commands

```yaml
- name: api-tests
  cmd: pytest -v
  required: true
- name: eval-gate
  cmd: pytest eval/ -v
  required: true
```

## Success criteria

- Diagnosis (bias vs variance) reported; approach chosen from it.
- New features are all creation-time (leakage-asserted) OR a reframed target is tested.
- Retrained model measured vs current on held-out pathway, per repo, with CI + effect size.
- Ship/reject decision on the ADR-0006 bar (CI excludes zero on k8s), human-confirmed.
- If shipped: full cutover (artifacts + manifest + re-record + re-baseline + deploy + live verify).
- If not: honest negative documented (resolution not improvable from available data — a real finding).
- ADR-0024 with diagnosis, approach, result, decision.

## Build order (CC autonomous)

1. Diagnose: bias vs variance, residual analysis, per-bucket. ESCALATE the diagnosis (it picks the path).
2. Per the diagnosis: creation-time feature engineering (leakage-guarded) AND/OR task reframe (bucket).
3. Retrain; measure vs current per repo with CI. ESCALATE the result before ship decision.
4. On approval: cutover if it ships (full deliberate artifact cutover) OR document the negative.
5. ADR-0024.
```

