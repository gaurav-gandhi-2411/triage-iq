# ADR-0018 — Gold Eval Set Train-Contamination Disclosure + Re-Baseline Plan

**Status:** Accepted
**Date:** 2026-07-06
**Decider:** Gaurav Gandhi

## Context

While verifying disjointness for an in-progress gold-set expansion (n=60→119, tracked separately,
not yet merged to `main` as of this ADR), a three-way disjointness guard was built for the first
time. No equivalent guard existed before that work. Running it retroactively against the
*existing* n=60 gold set (`data/gold_triage_plans.parquet`, curated 2026-04-29, predating this
guard) surfaced train-set overlap that had never been checked.

This ADR discloses that contamination, its two distinct root causes, its blast radius against the
n=60 gold set currently in production, what it does and does not affect on the live
`/eval/summary` endpoint, and the fix to prevent recurrence. **Scope note:** this ADR and its
accompanying changes are deliberately limited to the disclosure + the root-cause fix. The larger
gold-set expansion (n=60→119) and the resulting re-baseline are separate, not-yet-finished work on
another branch and are explicitly out of scope here — see "Not in scope" below.

## Root cause 1 — `component_match` / `classifier_train` (pre-existing, NOT caused by ADR-0009)

`scripts/10_curate_triage_gold.py::load_eval_splits()` sources gold candidates from
`temporal_val + temporal_test + classifier_val + classifier_test`, deduplicated by issue number.
The temporal split and the classifier split are two **independently computed** splits over the same
corpus (different logic, different label — `created_at` order vs. stratified by `component`). An
issue held out by the temporal split says nothing about its membership in the classifier split, and
`load_eval_splits()` never checked the latter against the *other* split's train set. This gap has
existed since the gold set was first curated, 2026-04-29.

Verified directly (not inferred): `data/models/component_classifier_{slug}.pkl` hashes match
`MANIFEST.sha256` (confirmed deployed) and were fit 2026-04-28 — one day *before* gold curation —
with only a calibration wrapper added 2026-05-19 (temperature scaling on existing logits, no
retrain: CHANGELOG confirms "accuracy unchanged +0.0pp"). `stratified_classifier_split()` itself
(`src/triage_iq/data/splits.py`) has been unmodified since creation (`c989980`, 2026-04-28), and the
ADR-0009 commit (`5560eb9`) diff touches only the temporal-split call site in `scripts/03_split.py`
— the classifier-split call site is untouched.

**This means `component_match` has been contaminated since the very first judge run this project
ever produced (W1.1, 2026-04-29/30) — it is not a new regression, and it predates ADR-0009 by
roughly a month.** Live on `/eval/summary` for approximately three months.

## Root cause 2 — `resolution_estimate_reasonableness` / `temporal_train` (genuinely caused by ADR-0009)

ADR-0009 (commit `5560eb9`, 2026-05-31) changed the resolution predictor's temporal split from
`closed_at` to `created_at` ordering, correctly fixing a distribution-shift bug. But this
regenerated `data/processed/*_temporal_train.parquet` (mtime 2026-05-30) with **different issue
membership** than existed when gold was curated (2026-04-29). Issues correctly held out under the
old sort were reassigned into `temporal_train` under the new one. `data/models/
resolution_predictor_{slug}.pkl` (mtime 2026-06-19, confirmed deployed) was fit on this new,
gold-overlapping `temporal_train`.

**This contamination is real and is a direct side effect of the ADR-0009 fix — first live the day
that model was deployed, 2026-06-19.** Live on `/eval/summary` for approximately two weeks.

These are two different bugs, with two different fixes, discovered together only because W5 was the
first time anyone checked either. Attributing both to ADR-0009 (an earlier framing of this
disclosure) would be wrong and is explicitly rejected below.

## Blast radius (verified directly against current split files, not estimated)

| Repo | classifier_train overlap | temporal_train overlap | contaminated (union) | clean |
|---|---|---|---|---|
| kubernetes/kubernetes (n=30) | 15 | 14 | 24 | 6 |
| microsoft/vscode (n=30) | 5 | 27 | 30 | 0 |

Of the current production gold set (n=60), **6 issues are clean (all kubernetes/kubernetes); 54 are
contaminated.**

## What is exposed vs. not

- **Exposed:** `component_match` and `resolution_estimate_reasonableness` dimension means in
  `reports/eval_summary.json` (served live, verbatim, by `GET /eval/summary`,
  `src/triage_iq/api/app.py:379`) and `reports/eval_baseline.json` (the CI quality-regression gate
  baseline, per-repo).
- **Not exposed — `similar_issues_relevance`:** the deployed retriever is the zero-shot BGE baseline
  (the W3 fine-tune was rejected per ADR-0016 and never shipped); nothing in that pathway trains on
  gold issues.
- **Not exposed — `priority_alignment`:** gold priority is derived from ground-truth resolution
  speed, not produced by any trained model; there is nothing to memorize.
- **Not exposed — the headline "beats naive" resolution number** (`reports/resolution_results.json`,
  cited in READMEs/data card): this is computed on the resolution predictor's own dedicated
  train/val/test split, a separate pathway from the judge gold set. **Unaffected.**

## Is the inflation magnitude recoverable?

**No — investigated and not recoverable without manufacturing a number.** `reports/eval_baseline.json`
was first committed 2026-06-20, three weeks after the ADR-0009 regen — there is no pre-contamination
snapshot of this baseline to diff against. The only pre/post-regen judge runs that exist
(`triage_results_w12_cohere.json` vs. `triage_results_w4_cohere.json`) were produced by the same
commit that *also* removed leaky resolution features and recalibrated the model — diffing them would
attribute model-quality changes to contamination, which is wrong. `reports/w4_ablation/*.json`
isolates split-only effects but for the resolution predictor's own regression metric, not the judge
dimension. We know contamination exists and its exact membership (table above); we do not have and
will not fabricate a "how many points did it inflate the score by" number.

## Fix applied — closing the recurrence path, not just documenting it

`scripts/10_curate_triage_gold.py::load_eval_splits()` now loads both `classifier_train` and
`temporal_train` issue-number sets (new `load_train_numbers()` helper) and excludes any candidate
present in either, in addition to the existing val/test union. Verified against the real
(non-mocked) split files: this drops 950
previously-eligible kubernetes/kubernetes candidates and 390 microsoft/vscode candidates that were
held out by one split while training the other. Covered by
`tests/test_10_curate_triage_gold.py` (5 tests: loader behavior, missing-file handling, and both
directions of the cross-split exclusion). **Any future re-curation from this script can no longer
reintroduce this specific leak.**

## Decision

1. Mark `judge.dimensions.component_match` and `judge.dimensions.resolution_estimate_reasonableness`
   in `reports/eval_summary.json` as under correction — stop serving them as valid via
   `/eval/summary` without qualification. (Applied in this same change.)
2. Fix `load_eval_splits()` now, in this change, so the underlying defect cannot recur regardless of
   which gold set is curated next. (Applied in this same change.)
3. Re-baseline `component_match` and `resolution_estimate_reasonableness` once a verified-clean gold
   subset is available. **Not part of this change** — tracked on the separate gold-set-expansion
   branch/PR, which is not yet merged and not yet re-recorded. This ADR does not assert a specific
   re-baseline n or per-repo split; that will be documented when the re-baseline actually ships.

## Not in scope (deliberately excluded from this change)

- The gold-set expansion from n=60 to a larger set — separate, unmerged branch/PR.
- Any specific "clean subset" size or per-repo composition for a future re-baseline — depends on
  the expansion above, which hasn't landed.
- Re-running the judge/cassette pipeline or updating `reports/eval_baseline.json`'s means — a
  distinct, human-approved step that happens after a clean gold set exists.
- Per-repo grounding-ratchet threshold changes — unrelated to this disclosure, decided separately.

This keeps the disclosure — which is urgent, since production currently serves contaminated numbers
as valid — decoupled from the re-baseline, which isn't ready yet.

## Consequences

- Honest: two independent, differently-timed bugs, not one. `component_match` has been silently
  wrong since project week 1 (~3 months live); `resolution_estimate_reasonableness` since the
  ADR-0009 deploy, 2026-06-19 (~2 weeks live). Neither was catchable before W5 built the first
  disjointness guard, and neither is catchable by that guard either — it only checks freshly
  *ingested* rows, not the original 60, which is why this ADR's direct fix to
  `load_eval_splits()` (not just the W5 guard) closes the actual recurrence path.
- The headline resolution "beats naive" metric is unaffected and remains citable as-is.
- No inflation-magnitude number is claimed; only membership-level contamination is disclosed.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Manufacture a "before/after" delta from W1.2 vs. W4-A | Confounds contamination with simultaneous de-leaking/recalibration in the same commit — would misattribute model-quality change as contamination effect. |
| Attribute both contaminations to ADR-0009 | False per the mtime/hash evidence — `classifier_train` predates ADR-0009 by a month; would misdirect any future reader trying to prevent recurrence, and would incorrectly imply the W5 disjointness guard (built for a different script) already covers this path. |
| Document the `load_eval_splits()` gap without fixing it | Leaves the exact leak reproducible on the next re-curation. Rejected — fixed directly, with tests. |
| Quietly re-baseline without disclosure | Live `/eval/summary` currently asserts numbers known to be wrong; violates the project's honest-documentation standard. |
