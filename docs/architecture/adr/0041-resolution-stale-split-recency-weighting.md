# ADR-0041 — Resolution predictor: stale-split fix (k8s ships, vscode doesn't) + recency weighting rejected

**Status:** Accepted — k8s re-split is a verified candidate awaiting cutover approval; vscode
re-split and recency weighting are both rejected, negative results recorded
**Date:** 2026-08-06
**Decider:** Gaurav Gandhi

## Context

GG's LEVER 3 ask: retrain the vscode resolution model on temporally-recent data (or with recency
weighting) to fix the train/test era mismatch ADR-0009 T1.5 diagnosed, and asked to fold in a
bigger finding first: `{repo}_temporal_{train,val,test}.parquet` were last generated
**2026-05-30**, but `data/processed/issues_{repo}.parquet` was regenerated **2026-07-11** (Phase
2b corpus growth) and grew substantially since. The currently shipped resolution models were
trained on stale, narrow splits relative to their own now-larger source corpus:

| Repo | OLD split (05-30) | Current corpus available (07-11) |
|---|---|---|
| vscode | n=6,154, train ends Apr 2016, test = a single 5-day window in Apr 2026 | n=12,242 closed issues, 99% more than the OLD split used |
| k8s | n=14,968, train/val/test span only Jun 2014-Oct 2015 (16 months) | n=29,911 closed issues, 100% more than the OLD split used, but genuinely capped at ~2016 (see below) |

## What was tried, in order (measure-first, same leakage discipline: temporal split by
`created_at`, no post-creation features, per ADR-0009)

**1. Re-split from the current corpus** (`scripts/lever3_resplit_resolution.py`, `time_based_split`
called directly — deliberately NOT `scripts/03_split.py`'s `main()`, which also regenerates the
component classifier's stratified splits in the same call; that's a separate, already-shipped,
ADR-0036-verified system and touching its training data was explicitly out of scope here).

**2. Retrain point + bucket models on the re-split** (`scripts/lever3_train_resolution.py`,
untuned `ResolutionTimePredictor.fit()` defaults — a first-pass "does re-splitting help at all"
measurement; Optuna tuning is a follow-up if a result looks promising enough to invest in).

**3. Recency-weighted retrain on top** (`scripts/lever3_train_resolution_recency.py`) — exponential
weight `exp(-ln(2) · age_days / half_life)`, age measured from each example's `created_at` to
train's own latest `created_at` (never to the val/test window — that would leak test-period
knowledge into the weighting scheme itself). Swept half-lives {90, 180, 365, 730} days, selected
by validation MAE, reported on test.

## Results

### k8s — re-split is a real, sizeable win. Recency weighting adds nothing on top (expected).

| | Naive | Re-split (untuned) | Re-split + recency (selected half-life=730d) | Currently shipped (05-30 split) |
|---|---|---|---|---|
| Point MAE | 104.23d | 101.98d (+2.2%) | 101.27d (+2.8%) | 104.05d vs 106.29d naive (+2.1%) |
| Bucket accuracy delta vs naive | — | **+6.35pp CI[+5.08,+7.55]**, excludes zero | +6.62pp CI[+5.31,+7.92], excludes zero | +3.27pp CI[+1.80,+4.74] (currently shipped) |

**The re-split roughly doubles k8s's bucket-classifier gain over naive** (+3.27pp → +6.35pp),
using genuinely more of the available data (23,928 vs ~12,000 train rows). Recency weighting
selects a 730-day (near-maximal) half-life — effectively negligible weighting — and changes
nothing materially. This is expected: k8s's closed-and-timestamped corpus is **still entirely
2014-2016** even after using 100% more rows — the corpus's issue-number range (#1-30,000) simply
doesn't reach further forward in calendar time (k8s created 49,266+ issues total by 2026-07-10 per
prior corpus-feasibility work; only the first 30,000 by number were ever scraped). There's no
later era for k8s to weight toward — "more data, same window" is the whole effect, not "newer
data."

### vscode — re-split alone makes things WORSE. Recency weighting doesn't rescue it.

| | Naive | Re-split (untuned) | Re-split + recency (selected half-life=90d) |
|---|---|---|---|
| Point MAE | 3.91d | 11.81d (**-202.3%, much worse**) | 6.82d (**-74.6%, still worse**) |
| Bucket accuracy delta vs naive | — | **-1.06pp CI[-1.63,-0.57]**, excludes zero (worse) | **-24.73pp CI[-27.10,-22.20]**, excludes zero (much worse) |

**Root cause, investigated, not assumed**: the re-split's test window (Feb 2025–Apr 2026) is
**83.8% "hours"-bucket issues** (median resolution ≈1.2 hours) — a far sharper skew than even the
OLD split's already-fast test window (ADR-0009 T1.5: median 0.1 days ≈2.4h). This makes the naive
majority-class baseline extremely strong by construction (83.76% bucket accuracy just by always
guessing "hours") and correspondingly hard to beat. Recency weighting improves the point-MAE
disaster somewhat (-202%→-75%, still a clear loss) but makes the bucket classifier dramatically
worse (-24.73pp) — down-weighting the bulk of historical training volume starves the classifier
of the signal it needs to discriminate the ~16% of issues that AREN'T near-instant closures,
without a corresponding gain from the recent examples it's left with.

**Spot-checked the fast-closing issues directly, not just the aggregate stat**: sampled the
sub-24h test-window issues and found a cluster (#240070-240145, tight number range, ~14
different low-history authors) of near-duplicate "Terminal not working" reports, closed within
hours — consistent with a specific regression triggering a wave of duplicate bug reports that
get triaged/closed fast, not a general "vscode now resolves everything instantly" behavior
change. `vs-code-engineering[bot]` appears 57 times among the fast-closing issues' authors —
confirms some bot-driven activity in the mix. **This is a disclosed observation, not chased
further here** (filtering duplicate/bot-driven issues from the eval, or using a wider/more
representative test window, is a legitimate follow-up but is new scope beyond this ADR).

## Decision

**Ship the k8s re-split as a candidate for cutover** (same escalated process as ADR-0040's
retrieval index: rebuild verified, `resolution_predictor_kubernetes_kubernetes_lever3.pkl`
saved, NOT yet promoted to the served path, NOT yet published to GCS, pending GG's explicit go).

**Do NOT ship anything for vscode.** Both the plain re-split and re-split+recency-weighting
underperform naive on vscode, sometimes badly. `BUCKET_CLASSIFIER_TRUSTED["microsoft_vscode"] =
False` (ADR-0025) stays correct and unchanged — vscode continues serving the naive-prior
fallback, now with additional evidence (not just the original -22.08pp finding) that the
resolution-time task is currently unlearnable for vscode with the data and features available,
not merely under-split.

**Recency weighting, as implemented, is rejected for both repos** — a null result for k8s (no
data span to exploit) and a negative result for vscode (actively harmful to the bucket
classifier). Not pursuing further half-life tuning or a different weighting function without a
new idea about why exponential-by-age weighting would help here that this measurement doesn't
already speak against.

## Consequences

- **k8s point regression + bucket classifier get materially better** if the cutover is approved
  and executed — same rebuild → verify → publish → deploy discipline as ADR-0040, not done here.
- **vscode's resolution predictor stays exactly as-is** — no regression, no change, the naive
  fallback was already the right call and stays the right call.
- **Open finding, not actioned**: the duplicate-report-wave pattern in vscode's recent fast-closing
  issues suggests the eval methodology itself (not just the model) may be measuring a
  non-representative window. A future session revisiting vscode resolution should look at
  filtering likely-duplicate/bot-triaged issues from the eval population, or widening the test
  window's calendar span, before concluding the task is unlearnable — that's a different question
  from "does this model beat naive on this specific window," which is what was measured here.
- **k8s's split freshness needs a maintenance answer, not just a one-time fix**: if the corpus
  gets forward-scraped again (past #30,000) in the future, this exact staleness bug will recur
  unless split regeneration becomes part of the corpus-growth checklist rather than a manual step.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Tune vscode's re-split model with Optuna before giving up | The gap (-202% MAE, -1.06pp bucket even before weighting) is too large to plausibly be a hyperparameter artifact — the untuned model isn't close to naive, it's dramatically worse. Hyperparameter search on a data problem this size doesn't fix a data problem. |
| A wider half-life sweep for vscode recency weighting | The trend across the swept range (90d best, monotonically worse as half-life grows) already shows the direction; there's no evidence a narrower half-life would reverse the bucket-classifier collapse rather than worsen it further by starving the model of even more training signal. |
| Filter the duplicate-issue wave from the vscode eval and re-measure now | Genuinely promising, but it's a different, new piece of work (needs a duplicate-detection heuristic decision, not just a config flip) — logged as an open finding above rather than attempted inside this ADR's scope. |
