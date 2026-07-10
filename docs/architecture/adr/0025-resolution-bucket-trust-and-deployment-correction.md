# ADR-0025 — Resolution Bucket-Classifier Trust Decision & Deployment Correction (W6 Phase 1)

**Status:** Accepted
**Date:** 2026-07-10
**Decider:** Gaurav Gandhi

> **Numbering note:** `spec.md` (this iteration's brief) referred to this as "ADR-0024." That
> number is already claimed by an unmerged hygiene-pass ADR on another branch. This is ADR-0025.

---

## Context

The resolution stage is the pipeline's weakest: it loses to naive on vscode (`resolution_results.json`,
`improvement_pct=-70.5`) and barely beats naive on k8s (`improvement_pct=+2.1`). ADR-0021/0023 already
established the model's conformal interval width is not diagnostic of its own errors. This iteration's
brief: diagnose bias vs. variance before choosing between better creation-time features or reframing the
target — and only then decide what, if anything, ships.

## Diagnosis (bias vs. variance)

`scripts/w6_diagnose_resolution.py` reloads the currently-shipped `resolution_predictor_{slug}.pkl`
(no retraining) and replays it over the held-out temporal test set, decomposing the residual per true
bucket (`reports/w6_resolution_diagnosis.json`).

| | k8s (n=1498) | vscode (n=616) |
|---|---|---|
| Model MAE | 104.05d | 6.02d |
| Naive MAE | 106.29d | 3.53d |
| Oracle bucket-median MAE | 43.91d | 0.18d |
| Oracle recovers | 58.7% of naive's error | 94.9% of naive's error |

Both repos are **bias-dominated**, from different causes:

- **k8s**: error is concentrated in the "long" bucket (n=201, mean_bias ≈ mae ≈ -688d — systematic
  underestimation, not scatter). Knowing the true bucket recovers most of the error. This is a
  fixable-bias signature, not irreducible variance — and a fix for it already exists (see below).
- **vscode**: 563 of 616 test issues fall in the single "hours" bucket, all with a large positive
  mean bias (+5.99d — the model systematically overestimates fast-closing issues). Per ADR-0009 T1.2/
  T1.5, vscode's train window (2015–2020) is distribution-shifted from its 2026 test window; this is a
  stale-training-data artifact, not a feature gap, and no creation-time feature closes a train/test
  shift without fresher data (out of scope for this iteration).

**The fix for k8s's bias already exists and was never properly evaluated.** ADR-0009 T2.4 built a
bucket classifier and shipped it for k8s on an arbitrary `obo ≥ 60%` threshold (65.9% obo, passed) —
with no confidence interval, and no statistical test of whether it actually beats guessing. This
iteration evaluates that same classifier against a naive majority-class baseline with a bootstrapped
95% CI (2000 resamples, seed=42, paired per-issue delta), the same ADR-0006 "CI excludes zero on k8s"
bar every other improvement in this project is held to:

| | k8s | vscode |
|---|---|---|
| Trained accuracy vs. naive | 37.6% vs 34.3% | 33.6% vs 55.7% |
| **Accuracy delta, bootstrapped 95% CI** | **+3.27pp [+1.80, +4.74]** — excludes zero, positive | **-22.08pp [-25.81, -18.02]** — excludes zero, wrong direction |
| Trained obo vs. naive | — | — |
| **Obo delta, bootstrapped 95% CI** | **+6.94pp [+5.01, +8.88]** — excludes zero, positive | — |

## Decision 1 — k8s: formalize the existing bucket classifier's ship decision on the rigorous CI basis

**The model does not change.** k8s already ships `resolution_predictor_kubernetes_kubernetes.pkl`'s
bucket classifier per ADR-0009 T2.5. What changes is the *basis* for trusting it: T2.4's arbitrary
`obo ≥ 60%` threshold is replaced by the bootstrapped CI-excludes-zero criterion (ADR-0006), which is
the rigorous version of the same question and happens to agree with the old threshold's verdict for
k8s. No retrain, no artifact change, no `/triage` output change for k8s. This is a decision/documentation
change only — confirmed by the diagnosis script reloading the existing `.pkl` with zero training.

## Decision 2 — vscode: honest negative, scope held

`-22.08pp [-25.81, -18.02]` is a train/test distribution shift (ADR-0009 T1.5), not a feature gap. No
creation-time feature or target reframe fixes a stale-training-window problem; that requires fresher
data, which is out of scope here. vscode's bias-dominated error and its bucket-classifier failure are
documented as a correctly-scoped negative finding — not forced into a positive.

## Decision 3 — Deployment Correction: vscode's naive fallback was documented but never wired

**What was claimed** (ADR-0009 T2.5): "Vscode uses naive prior (majority class 'hours', 33% training
frequency) flagged as low-confidence" — i.e. `predict_bucket()` should serve the naive
`bucket_train_distribution` majority class for vscode, not the trained classifier's output.

**What was actually deployed:** `predict_bucket()` (`src/triage_iq/models/resolution.py`) had no
repo-conditional logic at all. It served `self.model_bucket`'s prediction whenever `model_bucket is
not None`, for every repo — including vscode. Since vscode's saved predictor has a real, trained
`model_bucket` (confirmed directly: `ResolutionTimePredictor.load(...).model_bucket is not None`),
production has been serving the **trained classifier's output for vscode since T2.4/T2.5 shipped**,
never the documented naive fallback. This iteration's diagnosis (`accuracy_delta_bootstrap` above)
proves that classifier is significantly worse than guessing — prod was knowingly worse than the
documented behavior for as long as this has been live.

**Discovery:** Found while confirming whether formalizing the k8s CI decision changed `/triage` output.
Checking whether vscode's fallback was actually wired (it should have been a no-op check) surfaced that
`model_bucket` was non-`None` and unconditionally served.

**Fix** (mirrors the ADR-0009 Deployment Correction disclosure pattern): `BUCKET_CLASSIFIER_TRUSTED`
(`src/triage_iq/models/resolution.py`) makes the trust decision explicit and testable — keyed on the
same accuracy-delta-CI criterion, not a hardcoded repo string check:

```python
BUCKET_CLASSIFIER_TRUSTED: dict[str, bool] = {
    "kubernetes_kubernetes": True,   # +3.27pp [+1.80, +4.74] -- excludes zero, positive
    "microsoft_vscode": False,       # -22.08pp [-25.81, -18.02] -- excludes zero, WRONG direction
}
```

`predict_bucket()` now falls back to the naive `bucket_train_distribution` prior when the repo's
classifier is untrusted, in addition to the existing `model_bucket is None` fallback. Covered by
`tests/test_resolution.py` (5 tests: trusted-repo uses classifier, untrusted-repo falls back despite
having a trained model, unmeasured-repo defaults trusted, the trust dict itself is pinned as a
regression guard, and `model_bucket is None` always falls back regardless of trust).

**Net effect on `/triage` output:** vscode's `resolution_bucket`/`resolution_confidence_pct` change
from the trained classifier's per-issue prediction to the fixed naive-prior label/probability. k8s is
unaffected (already trusted, already correct). This IS a prod-facing behavior change for vscode
(unlike Decision 1's k8s formalization, which changes no output) — deployed and live-verified as a
targeted code fix, not a model cutover: no retrain, no GCS artifact change, no manifest change.

**Judge quality impact:** re-measured directly (not assumed) via a targeted partial cassette re-record
of the 11 affected vscode judge entries (live local `qwen3:8b`, zero-cost) — 0 changed/removed entries,
11 new. vscode's judge mean is unchanged at 8.3636/15, per-dimension breakdown identical to 4 decimals.
The fix changes only a supplemental categorical field (`resolution_bucket`); the numeric point/interval
estimate the judge scores against (`model_point`/`model_q10`/`model_q90`) is untouched. All 12 eval
tests (9 invariants + 2 quality-regression gates + `test_cassette_hash_matches_baseline`) pass post-fix.

## Deferred — further creation-time features for k8s

k8s's monotonic underestimation on the "long" bucket suggests the model may lack a feature that signals
"this will be slow." This is a **separate, not-yet-approved follow-on** to the sure win already banked
above — requires its own escalation before investment, per this iteration's scope.

## Consequences

- k8s bucket classifier ships on a statistically rigorous basis; no output change.
- vscode's documented naive fallback is now actually true in production — closing a documented-vs-
  deployed gap of the same class ADR-0009's own Deployment Correction and ADR-0013's calibration-gap
  disclosure already caught in this project.
- vscode remains an honest negative: nothing in this iteration improves its resolution-time modeling
  itself; the fix only stops serving a strictly-worse-than-naive classifier.
- Trust is now a per-repo, CI-derived, testable flag — not an implicit "if trained, serve it" default —
  so any future repo's classifier must clear the same bar before being trusted.

## What was NOT shipped

- New creation-time features or target reframing (deferred, requires separate escalation).
- Any change to k8s's served `/triage` output (formalization only).
- A model retrain, GCS artifact republish, or manifest change for either repo.
