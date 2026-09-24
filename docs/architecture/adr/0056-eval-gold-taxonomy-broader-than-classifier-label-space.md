# ADR-0056 — The eval gold set's component labels were drawn from a broader taxonomy than the classifier was trained to emit

**Status:** Accepted (disclosure) — metric-fix decision deferred, not yet made
**Date:** 2026-09-04
**Decider:** Gaurav Gandhi

## Context

While diagnosing why `declared_attribution` is `None` on all 64 re-recorded entries (a separate,
already-reported schema/prompt investigation), the 4 vscode issues currently failing
`test_vscode_no_fabrication` were checked against gold labels as a sanity test of the hypothesis
that the grounding gate penalizes correct LLM overrides of a wrong classifier prediction. That
check surfaced something larger than the original question.

**Finding:** for 7 of vscode's 11 gold eval issues (63.6%) and 7 of kubernetes/kubernetes's 53
(13.2%), `gold_component` is a label the deployed component classifier was **never trained to
emit, under any configuration** — not merely outside its top-3, outside its entire ~28/35-class
output space. No top-N cutoff, confidence threshold, or model change fixes this; the label
literally does not exist in the classifier's `classes_`.

### Verification this is not sampling noise

vscode's classifier is independently reported at 89.8-90.4% top-3 accuracy on its own held-out
187-row test set (`reports/classifier_results.json`, ADR-0028/0036, CI [85.3, 93.8]). The 64-issue
eval set's actual gold-in-top-3 rate, measured directly by cassette replay
(`scripts/measure_grounding.py`, zero live calls), is **3/11 = 27.27%** for vscode (k8s: 37/53 =
69.81%, also below its own ~82.5% reported figure, same direction, smaller gap). At a true 90%
underlying rate, observing 3/11 successes has probability on the order of 1-in-a-million —
these are two different populations, not the same distribution sampled unluckily.

## Cross-check against prior ADRs (per explicit instruction — is this known or new?)

**Checked ADR-0018 (gold-set/train contamination) and ADR-0036 (classifier multi-label
supervision fix) directly. Neither characterizes this gap. This is a new finding, not a
known-but-unresolved one.**

- ADR-0018 is about issue-level *membership* leakage (gold issues present in `classifier_train`
  or `temporal_train`) — entirely about which *issues* are double-counted, never about which
  *labels* the classifier can emit.
- ADR-0036 is about *within-issue* multi-label collapse (`normalize_labels()` keeping only the
  first of several valid labels per issue, discarding the rest — 30.4% k8s / 8.0% vscode of test
  rows affected). That is a supervision-quality defect for issues the classifier *does* train on.
  It does not touch labels the classifier's output space excludes entirely.

## Root cause — traced to source, code-cited, and corrected mid-investigation

**The mechanism below (`stratified_classifier_split`'s `min_class_samples=10`) is real and does
gate the classifier's label space — but a direct corpus count shows it is NOT why these
particular 4 vscode / 5 k8s labels are missing. That first hypothesis is retracted below; the
actual mechanism is very likely training-data staleness. Recording both, per this project's own
honest-documentation standard (ADR-0018 precedent) — the wrong first explanation is as
informative as the right second one, and silently replacing it would hide how this was found.**

1. `src/triage_iq/data/preprocess.py:19-71` (`LABEL_FACET_PATTERNS`) is a hand-curated per-repo
   whitelist of "known" component labels — vscode's list has ~70 entries, including `scm`,
   `perf`, `terminal`, `editor-multicursor`, present since this whitelist's original commit (never
   added later — checked via `git log --follow -p`). `build_processed_df`
   (`preprocess.py:173-180`) writes every matching raw GitHub label into `issues_{repo}.parquet`'s
   `component` column, unfiltered by corpus-wide frequency.
2. `scripts/03_split.py:65` calls `stratified_classifier_split(df, label_col="component")`
   (`src/triage_iq/data/splits.py:72-111`) with its default `min_class_samples=10`. Any class with
   fewer than 10 corpus-wide examples is dropped from `classifier_train`/`classifier_val`/
   `classifier_test` (`splits.py:102-111`). **This is a real filter and does determine the
   classifier's label space** — `scripts/13_train_multilabel_classifier.py:69`
   (`classes = sorted(train["component"].unique().tolist())`) derives the currently-deployed
   multi-label classifier's class list directly from `{repo}_classifier_train.parquet`, the output
   of this exact split. Verified the live deployed `.pkl`'s `classes_()` exact-matches
   `reports/classifier_results.json`'s `per_class_f1` keys (28 vscode / 35 k8s).
3. **Retracted: genuine corpus rarity is not why `scm`/`perf`/`terminal`/`editor-multicursor` are
   excluded.** Reproduced `normalize_labels()` directly against all 13,315 raw vscode issue JSON
   files (`data/raw/microsoft_vscode/`, zero live calls, pure local computation). Raw label
   presence: `terminal` 385, `scm` 37, `perf` 22, `editor-multicursor` 16. Post-collapse (running
   the actual `normalize_labels()` first-match logic, accounting for ADR-0036's multi-label
   collapse defect): `terminal` 368, `scm` 33, `perf` 15, `editor-multicursor` 15 — **all four
   comfortably clear `min_class_samples=10` today**, most by a wide margin. Same check on k8s:
   `HA` 41, `cadvisor` 31, `provider/openstack` 27, `os/fedora` 13, `system-requirement` 10 — all
   five at or above threshold. None of these 9 labels should have been dropped by step 2's filter
   if it ran against today's raw corpus and today's whitelist.
4. **VERIFIED STALE (closed the loop — the recommended follow-up check was run directly).**
   Re-ran the real pipeline end to end: `scripts/02_preprocess.py` against current
   `data/raw/{microsoft_vscode,kubernetes_kubernetes}` (13,315 / 29,994 raw issue files) into a
   scratch directory, then `scripts/03_split.py`'s actual `stratified_classifier_split` against
   that fresh output (identical code — `preprocess.py`/`splits.py` diff byte-identical between
   this branch and `main`). Diffed the resulting `classifier_train.parquet`'s `component.unique()`
   against the **live deployed classifier's own `classes_()`** (not a report file):
   - **vscode: fresh label space is 47 classes vs. the deployed model's 28 — a strict superset,
     zero classes lost, 19 gained, including all 4 target labels** (`scm`, `perf`, `terminal`,
     `editor-multicursor` all present in the fresh `classifier_train` split).
   - **k8s: fresh label space is 47 classes vs. the deployed model's 35 — again a strict superset,
     zero lost, 12 gained, including 4 of 5 target labels** (`HA`, `cadvisor`, `os/fedora`,
     `provider/openstack`). **`system-requirement` remains excluded even on fresh data** — its
     post-collapse, `.notna()`-labeled count still falls under `min_class_samples=10` (the raw
     label count was exactly 10, but `normalize_labels()`'s collapse plus the labeled-only filter
     brings it under threshold) — this one is genuinely rare, not stale.
   - **Pool-size confirmation, both repos:** vscode's old training pool
     (`reports/classifier_results.json`: train 1,488 + val 187 + test 187 = 1,862) vs. today's
     fresh pool (train 3,380 + val 423 + test 423 = 4,226) — **2.27x growth**. k8s: old (2,284 +
     286 + 286 = 2,856) vs. fresh (5,368 + 671 + 671 = 6,710) — **2.35x growth**. Both repos show
     the same ~2.3x staleness ratio independently, which is strong corroboration this is a
     pipeline-refresh gap (the raw corpus growing over time, uncaptured by any retrain) rather
     than a per-repo anomaly.
   - **8 of the 9 originally-flagged out-of-taxonomy labels would be covered by a retrain on
     current data; 1 (`system-requirement`, k8s) remains genuinely too rare.**
5. The **temporal** split (date-ordered, built for the resolution predictor) applies no
   class-frequency filter at all — every whitelisted label survives it, rare or not, regardless of
   which snapshot fed it.
6. `scripts/10_curate_triage_gold.py::load_eval_splits()` (documented in ADR-0018) unions
   `temporal_val + temporal_test + classifier_val + classifier_test` as gold candidates. An issue
   whose component didn't survive into the classifier's (possibly stale) training split can still
   enter the gold set via the temporal split, carrying a label the classifier can never produce —
   this union mechanism is confirmed regardless of which upstream explanation (genuine rarity or
   snapshot staleness) turns out to be correct.

Each split behaves correctly for its own stated purpose. The gap is that `load_eval_splits()`
unions two splits with different class-inclusion policies without accounting for the difference —
so the gold set silently mixes classifier-representable and classifier-unrepresentable ground
truth, and nothing downstream was checking for that mix until this session. **What's newly
unresolved: whether "classifier-representable" is a genuine, permanent data-volume limit or a
fixable staleness gap — current evidence points to the latter.**

## Full list of out-of-taxonomy gold labels (VERIFIED, this 64-issue eval set)

| Repo | Out-of-taxonomy issues in this eval set | Raw/post-collapse corpus-wide frequency (today) |
|---|---|---|
| microsoft/vscode (7/11 = 63.6%) | `scm` ×2 (#311284, #286776), `terminal` ×3 (#311878, #312260, #239838), `perf` ×1 (#311836), `editor-multicursor` ×1 (#4996) | `terminal` 385/368, `scm` 37/33, `perf` 22/15, `editor-multicursor` 16/15 |
| kubernetes/kubernetes (7/53 = 13.2%) | `provider/openstack` ×3 (#12287, #12587, #12284), `os/fedora` ×1 (#13435), `system-requirement` ×1 (#14935), `HA` ×1 (#12665), `cadvisor` ×1 (#14762) | `HA` 41, `cadvisor` 31, `provider/openstack` 27, `os/fedora` 13, `system-requirement` 10 (raw counts; k8s post-collapse not separately re-derived) |

**Every one of these 9 labels clears `min_class_samples=10` today, most by a wide margin** — none
of them is rare in the current corpus. This is what falsifies the original "genuine rarity" root
cause and points to training-data staleness instead (see Root Cause step 4).

## What each affected metric actually measures, given the gap

| Metric | Reads gold? | Affected? |
|---|---|---|
| `grounding_status.all_grounded` / `verify_plan_grounding` | No — checks `predicted_component ∈ classifier_top3` only | Interpretation invalid on out-of-taxonomy issues: "ungrounded" there can mean the model was *right* and the classifier structurally couldn't agree, not that the model fabricated anything |
| `fabrication_rate` (`eval/run_eval.py:240`) | No (inherits grounding) | Same invalidity — this is the headline README number |
| `_GROUNDING_BASELINE` ratchet (`eval/test_invariants.py`) | No | Same invalidity — mechanism is fine, the ratcheted quantity is conflated |
| Component top-1/top-3 accuracy (`reports/classifier_results.json`) | Yes, but only within the classifier's own held-out test set, which by construction excludes every out-of-taxonomy label | Valid for what it measures; never was a valid ceiling estimate for a differently-sourced population that includes out-of-taxonomy labels |
| Judge quality mean | No — judge never sees `classifier_top3` or gold `component` (ADR-0028) | Unaffected (separate, already-documented blind spot) |
| Resolution-time / CQR / `priority_alignment` | No | Unaffected |

**Framing correction this ADR establishes:** `verify_plan_grounding` (ADR-0015) was designed as a
*consistency-with-the-pipeline* check ("did the LLM stay inside what System 1/2 actually gave
it"), not a *correctness-against-reality* check. That design was reasonable when classifier
top-3 recall was implicitly assumed to be a fair proxy for "the true answer is reachable." The
taxonomy gap breaks that assumption for a large minority of vscode's eval population. The metric
is doing exactly what it was built to do; what it was built to do stops being a fair proxy for
"is the model fabricating" once gold itself sits outside the classifier's reach.

## Concrete case evidence (4 currently-failing vscode issues, gold-verified)

| Issue | Model prediction | Classifier top-3 | Gold | Model correct? | Top-3 contains gold? |
|---|---|---|---|---|---|
| #239838 | terminal | typescript, install-update, javascript | terminal | Yes | No |
| #311284 | scm | api, tasks, typescript | scm | Yes | No |
| #311878 | terminal | api, javascript, debug | terminal | Yes | No |
| #311836 | webview | ux, debug, accessibility | perf | **No** | No |

3 of 4 are the LLM correctly overriding a classifier whose label space cannot reach gold — the
metric currently scores these identically to the 4th, which is genuine model error. The metric
cannot currently tell these apart.

## Published claims affected (listed, not edited — see companion session report)

`README.md:119-120` (fabrication rate 0.0%, both repos), `README.md:190` (hard zero-tolerance
gate framing), `docs/architecture/adr/0044-fabrication-rate-hard-gate-bound.md:21`
(`_GROUNDING_BASELINE` 0/53, 0/11), `eval/test_invariants.py`'s `_check_no_fabrication` docstring
("hard-fail correctness issue"). Component classifier top-3 accuracy claims (`README.md:93,95`)
are correctly scoped as stated but invite misreading as the LLM-grounding ceiling, which this ADR
shows they are not, on this population.

## Decision

**This ADR documents the disclosure only. The metric-fix decision is deferred — see the
companion session report's Priority 3 for three options (gold-based grounding for eval-only
reporting; excluding out-of-taxonomy issues from the denominator; retraining the classifier on
the full taxonomy) with trade-offs and a recommendation. The data-volume question for the
retraining option is answered by this ADR (see Consequences) — the corpus is not the blocker;
implementation is still deferred regardless. None of the three is implemented by this ADR.**

**Explicitly not done, and why:** `enable_validated_override_rescue` is not being enabled as a
"fix" for this gap. With the taxonomy gap identified, the override-rescue mechanism would rescue
the 3 correct cases *and* possibly the 1 genuinely wrong one too, for the wrong reason (it would
be validating overrides against a broken denominator's symptoms, not against the actual
taxonomy-coverage defect). Right answer for those 3 cases, wrong mechanism — deferred alongside
the metric-fix decision, not bundled into this disclosure.

**Also explicitly not done:** no re-record, no re-baseline. Re-recording against a metric known to
be measuring the wrong thing for a material fraction of vscode's eval population would produce
another untrustworthy baseline, compounding rather than resolving the problem ADR-0052 already
disclosed (no valid baseline currently exists).

## Consequences

- The published "0.0% fabrication rate" claims (README, ADR-0044) remain technically accurate
  under the metric's literal definition but should be read as "classifier-consistency rate," not
  "the model never fabricates" — a real, if narrow, honesty gap in how the number is framed.
- The vscode eval arm (n=11, dropping to n=4 if out-of-taxonomy issues are excluded per Priority
  3's option B) cannot support a hard pass/fail gate under any of the three metric-fix options
  considered — this is a sample-size problem underneath the metric-design problem. The
  eval-set-expansion work ADR-0052 already flagged as urgent is a precondition for a defensible
  vscode gate, not an optional enhancement.
- `declared_attribution`'s schema/prompt fix (queued from the prior investigation) is validated
  against a metric this ADR shows is partially broken — it stays queued, not implemented, until
  the metric question is settled (see companion report Priority 4c).
- **Resolved, correcting this ADR's own first hypothesis:** the raw corpus comfortably contains
  enough examples of all 9 affected labels (10-385 each, corpus-wide) to make classifier
  retraining (option C) viable from a data-volume standpoint — this is **not** the same
  data-volume ceiling ADR-0036 documented for genuinely rare tail classes. The evidence instead
  points to the classifier having been trained on a stale, smaller snapshot than what's currently
  in `data/raw/` — see Root Cause step 4 for the sizing evidence and the exact follow-up check
  (diff a fresh `03_split.py` run's label set against the deployed classifier's `classes_()`) that
  would confirm this precisely. This materially changes option C's cost estimate in the companion
  session report: if staleness is confirmed, retraining is a pipeline-refresh problem, not a
  data-collection problem.

## Phase 3 addendum (2026-09-04, same session): retrained, and it's a genuine trade-off

**Retrained the multi-label classifier on the fresh split (`scripts/13_train_multilabel_classifier.py`,
unmodified, zero spend, local CPU only), saved to the existing staging convention
(`component_classifier_{repo}_multilabel_staged.pkl` — NOT the deployed path). Leakage guard
passed both repos. `argmax_preserved=True` both repos (drop-in-replacement guarantee holds).**

**Label-space coverage: 8 of 9 originally-flagged out-of-taxonomy labels now covered.**
vscode: 47 classes (was 28), all 4 target labels present. k8s: 47 classes (was 35), 4 of 5
target labels present (`system-requirement` remains genuinely too rare even on fresh data).

**Accuracy: a real regression, not noise.**

| Repo | Metric | Old (deployed, smaller label space) | New (retrained, larger label space) | Δ |
|---|---|---|---|---|
| vscode | Top-3 | 89.84% [84.7, 93.4]* | 85.82% [82.17, 88.82] | **-4.02pp** |
| vscode | Top-1 | 76.47% | 69.98% [65.44, 74.15] | -6.49pp |
| k8s | Top-3 | 87.06% [82.7, 90.5]* | 85.99% [83.16, 88.41] | -1.07pp |
| k8s | Top-1 | 60.49% | 54.84% [51.06, 58.57] | -5.65pp |

*Old CIs from ADR-0036/README; **this is not a paired comparison** — the new test set (n=423
vscode, n=671 k8s) is a different, larger population than the old one (n=187/n=286), unlike
ADR-0036's own paired bootstrap. CIs overlap substantially on both repos' top-3 (not a
CI-excludes-zero regression), but the direction is consistent across both repos and both metrics
(top-1 and top-3), which is the same "consistent direction across independent measurements"
signal ADR-0036 itself used to trust a result. More classes to distinguish among is the
mechanical explanation — expanding 28→47 (vscode) / 35→47 (k8s) classes makes the classification
problem objectively harder, independent of any staleness question.

**Per the explicit instruction: this is reported and NOT recommended for shipping as-is.** Do not
promote the staged artifacts to the deployed path. This is a genuine trade-off, not a strict
improvement — more explicitly:

- **For grounding-metric validity, the new classifier is a clean, direct fix**: re-running the
  gold-in-top-3 check with the new classifier (zero LLM calls, classifier inference only) shows
  overall gold-in-top-3 jumping from 62.5% to **93.75%** (k8s 69.81%→94.34%, vscode
  27.27%→90.91%). Of the 4 known override cases, **3 of 4 now have gold correctly in the new
  classifier's top-3, AND the LLM's own (correct) prediction now also appears in that same
  top-3** — `#239838` (terminal ∈ [terminal, typescript, electron]), `#311284` (scm ∈ [scm, git,
  api]), `#311878` (terminal ∈ [terminal, api, tasks]). These 3 would no longer register as
  "ungrounded" under the metric's own existing, unmodified definition — no metric redesign
  needed for them. `#311836` (the genuine error) stays correctly ungrounded: neither gold
  (`perf`) nor the model's wrong guess (`webview`) appears in the new top-3
  (`[ux, extensions, api]`) — the metric continues to correctly flag the one real mistake.
- **For the classifier's own standalone accuracy, it's a step backward** — exactly the tension
  Phase 3e anticipated.

This changes Phase 4's metric-fix framing materially: if this classifier ever ships, most of the
"correct-override-flagged-as-fabrication" problem this ADR exists to document would resolve
itself as a side effect, without needing a redefinition of grounding. But it cannot ship as a
regression, so the metric-fix decision (Phase 4) still needs to stand on its own, independent of
whether/when a taxonomy-complete, accuracy-neutral-or-better classifier becomes available.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Treat 4/11 vscode fabrication rate as a real regression and investigate the LLM | Would have been the wrong diagnosis — 3 of the 4 "fabrications" are the LLM being correct where the classifier's label space cannot agree. Investigating the LLM would find nothing wrong with it. |
| Silently exclude out-of-taxonomy issues from the eval set going forward, no disclosure | Rejected — matches the project's honest-documentation standard (ADR-0018's own precedent: disclose contamination even when inconvenient, don't quietly patch the denominator). |
| Enable `enable_validated_override_rescue` immediately, since it would pass the 3 correct cases | Rejected per explicit instruction — right answer for those 3 cases, but validates against a denominator now known to be broken, and would also risk rescuing the 1 genuinely-wrong case for the same flawed reason. |
