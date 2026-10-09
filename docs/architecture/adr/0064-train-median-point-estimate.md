# ADR-0064 — Serve the training-window median as the point estimate (both repos)

Status: Accepted (integration branch; not merged, not deployed)
Date: 2026-10-08
Decider: Gaurav Gandhi. D2 (vscode: "a model that loses to naive under every variant is not served")
extended to k8s by D7 (2026-10-08). Numbered 0064 because 0063 is
`0063-degrade-on-llm-provider-errors.md`; this file was 0063 on the PR #155 branch.

## D7 scope change (read this first)

The sections below were written for vscode only (D2). D7 applies the same decision to
`kubernetes/kubernetes`; the vscode evidence and design stand unchanged. What D7 adds:

- `POINT_ESTIMATE_TRUSTED` is `False` for both repos. `resolution_point_source` is `"model"` or
  `"train_median"` (renamed from the draft's `"naive_median"` so the label says which median:
  the training-window median the published naive baseline used, `median(train.resolution_hours)/24`;
  k8s 3.0365 d, vscode 3.8374 d).
- Evidence for k8s, served path through `_collect_signals` over all 2,992 test rows
  (`reports/served_k8s_metrics.json`, produced at 30e6fd8): the learned point has MAE 102.09 d
  [93.89, 110.63] vs naive 104.23 d, a +2.06% gain (paired bootstrap gain CI [1.84, 2.45] d).
  Premise check on D7's wording: the *unpaired* CIs overlap, but the *paired* gain CI excludes zero,
  so the mean gain is small and real. D7 stands on the other measure: **median absolute error is
  7.28 d for the model vs 3.52 d for naive**, i.e. the model helps on the mean (a few huge
  long-tail issues) and is worse on the typical issue. The served point is what a reader takes as
  "how long will this one take".
- k8s keeps its learned parts: the bucket classifier (+6.95 pp [5.55, 8.36] over always-"hours") and
  the CQR interval around the model's Q10/Q90 (CQR v2, coverage 79.7% [77.9, 81.4] on the held-out
  part). The k8s interval is NOT re-centred (`INTERVAL_RECENTRED[k8s] = False`): lo/hi, bucket,
  confidence, top-3 and retrieval are bit-identical to main 30e6fd8
  (`tests/test_vscode_naive_point.py::test_k8s_bucket_interval_and_retrieval_identical_to_pre_change_main`,
  and the full-test-set comparison in `reports/d7_resolution_before_after.json`). vscode keeps the
  re-centred interval below (`INTERVAL_RECENTRED[vscode] = True`) because its model point is far from
  the median and a model-centred interval would not contain the served point.
- Consequence for the prompt: the point and (vscode) interval printed in the synthesis prompt change
  for every issue, so all 64 cassette entries are re-recorded (validator output: 64 of 64
  `stale_synthesis`, 53 k8s + 11 vscode). The conformal store is not fingerprinted (ADR-0059
  addendum), so CQR v2 does not add a re-record by itself.

## Context

The vscode resolution regressor loses to the naive baseline (training-set median, 3.8374 d) on
every measurement we have (docs/DECISION_LOG_2026-10.md D17-D18; evidence for this ADR in
`reports/vscode_naive_serving_eval.json`, produced by `scripts/eval_vscode_naive_serving.py`):

| vscode, reconstructed 616-row window (2026-04-21..27) | MAE (d) | vs naive |
|---|---|---|
| Naive median (3.8374 d) | 3.533 | 0% |
| Served today (emb_*=0, dsrs=0, no author) | 5.448 | -54.19%, paired gain CI [-2.089, -1.717] d |
| With serving-time embeddings (study S1 / B) | 6.442 / 6.590 | -82.34% / -86.51% |
| Retrain without emb_*, 10 runs | 3.56-6.54 | none beats naive |
| Zd: served as-is but dsrs = days since train start | 3.359 | +4.93%, gain 0.174 d [0.082, 0.275] |

The system already stops serving the *bucket* classifier where it loses to guessing
(`BUCKET_CLASSIFIER_TRUSTED`, ADR-0025: vscode falls back to the naive majority-class prior) and
the API already carries `resolution_model_beats_naive` so the UI can badge it. The *point
estimate* given to the synthesis prompt (and so the LLM's `expected_resolution_*` fields and the
conformal interval built from them) had no equivalent gate: vscode kept serving a model point
that is 54% worse than a constant.

Data caveats that bound every number here: the original vscode test parquet is unrecoverable, so
the window is a reconstruction (last 616 closed issues, 7 days, 2026-04-21..27); its median
resolution is 0.049 d and 91.4% of issues are in the "hours" bucket, while the training window is
2015-16 with median 3.84 d. Train and test are 10 years apart, which violates exchangeability.

## Decision

1. `POINT_ESTIMATE_TRUSTED` (in `models/resolution.py`, next to `BUCKET_CLASSIFIER_TRUSTED`):
   `{kubernetes_kubernetes: False, microsoft_vscode: False}` (D7), unlisted repos default to trusted.
   For an untrusted repo `TriageAssistant._collect_signals` replaces the model point with
   `naive_median_days(train_df)` = `median(resolution_hours of the training set) / 24` (vscode
   3.8374 d, identical to the study's `median(train)/24`). The raw model outputs stay available as
   `signals["model_point_days" | "model_lo_days" | "model_hi_days"]`.
2. Response, expand-only: `resolution_point_days`, `resolution_point_source` (`"model"` or
   `"train_median"`), `resolution_interval_basis` (`"model"` or `"naive_scaled"`). Nothing is
   removed or renamed; `resolution_model_beats_naive` keeps its meaning (a measured property of
   the trained model, still `false` for vscode). Bucket and `resolution_confidence_pct` are
   unchanged (vscode already serves the naive majority-class prior).
3. Interval: candidates were scored on the same chronological split `scripts/10_calibrate_cqr.py`
   uses (first 30% / 40% by `created_at` as calibration, eval-set rows excluded from calibration,
   the rest held out). Held-out n=370 (40/60 split), target 80%:

   | Served combination | coverage (Wilson 95%) | median width (d) | mean interval score (d) |
   |---|---|---|---|
   | A. naive point + model Q10/Q90 + stored Q (what a point-only swap would serve) | 74.3% [69.6, 78.5] | 148.0 | 135.4 |
   | B0. naive point + empirical train q10/q90 [0.067, 456.8] d | 38.9% [34.1, 44.0] | 456.7 | 457.0 |
   | B1. B0 + split-conformal Q (0.052 d) | 74.3% [69.6, 78.5] | 456.8 | 456.8 |
   | C0. naive point + naive-scaled model width (`naive*lo/p`, `naive*hi/p`), no Q | 46.0% [40.9, 51.0] | 93.6 | 81.7 |
   | C1. C0 + freshly calibrated Q (0.049 d) | 80.0% [75.6, 83.8] | 93.7 | 81.6 |
   | **C2. C0 + stored production Q (0.0527 d), chosen** | **82.7% [78.5, 86.2]** | **93.7** | **81.6** |

   (30/70 split, n=432: A 74.5%, B1 73.2%, C1 77.8%, C2 81.7% [77.8, 85.1].) C2 is served.
   Why: it is the only option whose interval contains the point it is paired with, it has the
   narrowest width and best interval score, its coverage is within the 80% target's CI at both
   splits, and it needs no new stored artifact (it reuses the Q already in
   `cqr_conformal_adjustments.json`; C1's fresh Q would have needed a GCS overwrite, which is on the
   never-do list). Rejected: B (an interval of 0-457 days is information-free and, with a
   model-free fixed width, does no better on coverage than A), A (74% coverage and 148 d wide,
   and it would put a point from one distribution inside an interval centred on another).
   `ConformalIntervalResult.empirical_coverage` and its CI describe the *served* interval, so for
   `resolution_interval_basis == "naive_scaled"` they come from the constant
   `NAIVE_INTERVAL_COVERAGE` (0.827 [0.7852, 0.8622], copied from the evidence JSON), not from the
   artifact's model-quantile statistics.
4. Never crash a request: if the naive median cannot be computed (no `resolution_hours`, all NaN,
   non-positive) the model output is served with a WARNING (`reason: no_train_median`) and
   `resolution_point_source == "model"`; if the predictor itself throws, the naive median is still
   served with the existing fixed `[1, 30]` d interval (`resolution_interval_basis == "model"`).
5. (Superseded by D7: k8s now serves the median too, with its interval, bucket and retrieval
   bit-identical; see the D7 section. The original trusted-repo proof, kept for the record:)
   k8s was bit-identical while trusted. Proven two ways:
   67 real k8s validation rows (every 45th of the 2991-row frame) hash identically (sha256 of
   prompt, point, interval, bucket, confidence, top-3, retrieved issues) between
   `origin/fix/k8s-resolution-embeddings-reuse` (e79cc70) and this branch, aggregate
   `c280834036fe56c15909daa719ca9ea507eed04fb2a9853e2a7527eefa03d1a0`; and
   `tests/test_vscode_naive_point.py::test_k8s_signals_hash_identical_to_pre_change_branch` pins a
   golden hash computed on #150's code.

Measured result on the 616-row window through the real `_collect_signals` (zero LLM calls):
point MAE 5.448 d (-54.19%) -> 3.533 d (0.00%); bucket accuracy unchanged 91.40% (the naive
majority prior was already served; the study's 75.97% was the unserved classifier); interval
coverage over the whole window 82.5% [79.3, 85.3] at median width 91.7 d (not held out: includes
the calibration rows).

### Interval coherence (owner request 2026-10-08; `reports/interval_coherence_d7.json`)

Rows: the D7 served outputs of the real `_collect_signals` (zero LLM calls), script
`scripts/interval_coherence_d7.py`. Hold-out = the chronological last 60 pct of the vscode window
(n=370), the part the stored Q (fitted on the first 40 pct of this window, on the MODEL-centred
construction) never saw.

**vscode: the CQR guarantee does not transfer to the re-centred interval by construction, so it was
measured. It does not miss; no recalibration was made.** Served interval (re-centred, +/- stored Q,
as `api/app.py` attaches it), nominal 80 pct:

| Interval | hold-out coverage (Wilson 95) | median width | full window n=616 |
|---|---|---|---|
| before: model-centred +/- Q | 74.1% [69.4, 78.3] | 146.2 d | 77.0% [73.5, 80.1], 140.0 d |
| served: re-centred +/- Q | 82.2% [77.9, 85.7] | 93.4 d | 82.5% [79.3, 85.3], 91.7 d |
| re-centred, no Q (what the prompt shows) | 45.4% [40.4, 50.5] | 93.3 d | 48.9% [44.9, 52.8], 91.6 d |

Width 146 -> 93 d (-36 pct) with coverage up 8 pp. The point estimate meets the target and the CI
contains 80 pct; its lower bound (77.9) does not clear it, which is what one 7-day window supports
and no more. A recalibration for the served construction was also computed
(`reports/vscode_naive_serving_eval.json`, C1 = fresh Q on the first 40 pct): 80.0% [75.6, 83.8]
hold-out (77.8% on the 30/70 split), no better than the stored Q (C2, 82.7% [78.5, 86.2]), so the
stored Q stays and `NAIVE_INTERVAL_COVERAGE` is unchanged. Two cautions: (1) the coverage comes
largely from Q lifting a lower bound of a few hours over issues that resolve in about 1 h; the
interval that the PROMPT shows (no Q) covers 45 pct, so the prompt's range is not an 80 pct
interval, the API's conformal interval is; (2) a single window, train median 3.84 d vs test
median 0.049 d: marginal coverage elsewhere is not established.

**k8s: 5 of 2,992 served points (0.17 pct, all below the lower bound; shifts 0.09-2.23 d) lay
outside their own unchanged model interval.** The point is now clamped onto the interval
(`min(max(median, lo), hi)`), the response carries `resolution_point_clamped: true` for exactly
those requests (false otherwise, and always false for vscode and the model path), and the prompt
shows the clamped point. Effect on the served-path metrics: MAE 104.229 -> 104.227 d, median AE
3.520 -> 3.520 d (unchanged at four decimals); interval, bucket and retrieval are untouched.

**The prompt now shows the served interval (2026-10-09, ADR-0059 addendum).** Caution (1) above is
resolved: the CQR adjustment is applied before synthesis, so the prompt, the plan's interval fields
and `resolution_interval_conformal` carry one interval, and the prompt line reads "80% prediction
interval (coverage-calibrated)". Coverage of the interval now in the prompt (served path, Wilson 95,
`reports/interval_coherence_d7.json`): vscode hold-out 82.2% [77.9, 85.7] (n=370, median width
93.4 d; the API reports the created_at-split figure 82.7% [78.5, 86.2], same interval); k8s hold-out
(last 70 pct, the CQR v2 split) 79.7% [77.9, 81.4] (n=2,095, 237.1 d), full window 79.9% [78.4, 81.2]
(n=2,992). The clamp is applied against this served interval; for k8s the count stays 5 of 2,992.
Measured before the change on the recorded cassette (zero LLM calls,
`reports/prose_vs_conformal_interval_precheck.json`): 0 of 64 plans quote a prompt bound in prose and
0 contradict either interval, so the old mismatch was mostly the false "80%" label and the lower
bound (vscode 11/11 prompt lower bounds differ from the conformal ones at display precision, 0.1 d vs
0.0 d; k8s 53/53 identical because Q is about -1 h against bounds of days). The effect on judged
quality is unmeasured until the re-record, and confounded with the median point and the System 3
label.

## Consequences

- vscode point error equals naive by construction. The product claim is now honest ("typical
  closed issue takes ~3.8 d in this repo's history") instead of a model that is worse than that.
  It is also uninformative: the *current* vscode traffic resolves in hours (median 0.049 d), so
  3.84 d overstates, and the interval's lower bound collapses to ~0 after the Q widening, making it
  effectively an upper bound (~94 d median). We are reporting, not fixing, the staleness of the
  2015-16 training window.
- Single window caveat: calibration and held-out rows come from the same 7 days (reconstructed,
  not the original test set), train median 3.84 d vs test median 0.049 d. Coverage is marginal and
  not guaranteed under this shift; 82.7% is a measurement on one window, not a guarantee. All
  misses are lower-bound misses (upper miss 0.0%).
- Zd finding, not acted on: one variant (dsrs = days since train start) beats naive by 4.93%
  (gain 0.174 d, CI [0.082, 0.275]) but relies on a feature extrapolated ten years past the
  training window, on one 616-row window, and k8s coverage collapses under the same change. It is
  fragile, so it does not justify serving the model. If vscode is retrained on a recent window the
  gate in this ADR is the one switch to flip, after a new measurement.
- Why not retrain: 10 retrains without `emb_*` (seeds None/42/1/2/3, 15- and 11-feature variants)
  gave MAE 3.56-6.54 d, none beating naive; a retrain needs a new artifact, GCS publish and cutover
  for no demonstrated gain; the data problem is distribution shift, not model capacity.
- Cassette dependency (D7: now ALL 64 entries, see the D7 section; the paragraph below is the
  original vscode-only measurement): the vscode synthesis prompt contains the point estimate and
  interval, so the 11 vscode eval issues' cassette keys change (verified by strict replay: 11 of 64
  `CassetteMissError`, all vscode; 0 k8s; baseline with the gate forced to trusted: 0 misses):
  `vscode-239838, -278113, -286776, -311284, -311836, -311878, -312260, -312423, -4978, -4993,
  -4996`. They must be re-recorded (coordinator, after the checkpoint-validation tool lands);
  until then the CI cassette gate is expected to FAIL on those entries. Judge entries are keyed on
  the plan JSON and so follow the synthesis re-record. This PR does not touch the cassette.
- The prompt template is deliberately unchanged (it still labels System 3 "LightGBM"), because
  editing the template would also change every k8s key. The numbers differ, the label does not;
  fix the label together with a future k8s re-record if desired.
- LLM-emitted bounds: `expected_resolution_lower/upper_days` come from the LLM reading the
  prompt's interval; the offline coverage above assumes the LLM echoes the prompt interval (not
  run, zero spend). The live check is the post-re-record eval.

### UI follow-up (not done here; do not edit the UI repo from this PR)

`ConfidenceBadge` (triage-iq-ui `src/components/ConfidenceBadge.tsx`) shows "Model below naive
baseline / This repository's resolution model underperforms a naive median predictor in
evaluation." when `resolution_model_beats_naive === false`. Once vscode serves the median that
text describes a model the user is no longer shown. Drive it from `resolution_point_source`
(absent on old servers: treat as `"model"`):

- `resolution_point_source === "train_median"`: badge text **"Historical median estimate"**,
  tooltip **"The trained model for this repository did not beat a simple historical median in
  evaluation, so this estimate is the median resolution time of past closed issues (3.8 days),
  not a prediction for this issue. The range is wide and calibrated on a single week of data."**
  Show `resolution_point_days` as the headline estimate.
- `"model"` and `resolution_model_beats_naive === false` (old behaviour): keep the existing text.
- `"model"` and beats naive: no badge.

`UnderTheHood.tsx` (line ~272) and `Eval.tsx` (~299, "vscode: LightGBM underperforms a naive
prior (-67.4%)") describe the model comparison and stay true as evaluation facts, but should add
"the live API serves the median for this repository".

## Alternatives

- Keep serving the model and rely on the badge: the status quo; rejected by owner decision D2.
- Retrain / re-embed / change dsrs: see Consequences (Zd, retrain).
- Interval options A, B0, B1, C0, C1: scored above. C1 (fresh Q) is the runner-up; it would be
  the better-targeted number (80.0%) but needs a new stored calibration artifact.
- Put the new fields on `TriagePlan`: rejected; they would enter the Groq strict schema or need a
  post-hoc strip, risking k8s cassette keys. The result-dict convention
  (`resolution_model_beats_naive` precedent) is zero-risk for k8s.
