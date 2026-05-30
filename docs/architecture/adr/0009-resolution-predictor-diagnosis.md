# ADR-0009 — Resolution-Time Predictor Diagnosis (W4 Phase 1)

**Status:** Accepted  
**Date:** 2026-05-30  
**Decider:** Gaurav Gandhi

> **This diagnosis invalidates all previously reported resolution-predictor metrics (k8s +3.3%, vscode +19.1%); both were produced by distribution-shifted `closed_at` splits and must not be cited.**

---

## Context

The audit flagged the k8s resolution predictor as effectively broken:

| Metric | k8s | vscode |
|---|---|---|
| MAE | 682 days | 3.4 days |
| vs naive | +3.3% | +19.1% |
| CI coverage (80%) | 0% | 42.5% |

The downstream judge score for `resolution_estimate_reasonableness` is 1.28/3 — the weakest
dimension. Phase 1 diagnosis determines which failure mode dominates before committing to Phase 2
work (re-scrape, reframe, or feature fix).

Diagnostic scripts: `scripts/w4_diagnostics/01_diagnose.py`  
Raw results: `reports/w4_diagnostics/diagnosis_results.json`  
Plots: `reports/w4_diagnostics/t1_1_replication.png`, `t1_2_temporal_dist.png`,
`t1_3_intrinsic_floor.png`, `t1_5_vscode_k8s_compare.png`

---

## T1.1 — Replication

Numbers replicate within 2% of reported values:

| | Reported | Replicated |
|---|---|---|
| LightGBM MAE | 682 days | 693.9 days |
| Naive MAE | 705.8 days | 705.8 days |
| CI coverage | 0% | 0% |

**Why 0% CI coverage — exactly.** The 20-sample CI inspection exposes the mechanism:

| Actual | Q10 | Q90 | Covered? |
|---|---|---|---|
| 850 days | 10.6 days | 37.8 days | No |
| 958 days | 5.1 days | 52.8 days | No |
| 267 days | 4.5 days | 40.8 days | No |
| 2213 days | 4.0 days | 61.9 days | No |
| 1169 days | 14.9 days | 129.8 days | No |

The model's Q10/Q90 predictions span [0.1d, 60d] and [4.5d, 202d] respectively — calibrated for
the train distribution. **The test set minimum resolution time is 65 days.** No test issue can
fall within an interval whose upper bound caps at 202 days while the median actual is 677 days.
The Q10/Q90 are not inverted or collapsed — they are well-calibrated for the training distribution
but the training distribution is completely disjoint from the test distribution.

---

## T1.2 — Staleness and Temporal Distribution

**The closed_at temporal split is structurally broken.**

The k8s data was created Jun 2014–Oct 2015 (16 months), but `closed_at` spans Jun 2014–May 2025
(11 years). 9.5% of issues closed after 2016. The temporal split by `closed_at`:

| Split | closed_at range | resolution_hours (median) | resolution_hours (max) |
|---|---|---|---|
| Train (80%) | Jun 2014 – Sep 2015 | **1.0 day** | 428 days |
| Test (10%) | Dec 2015 – **May 2025** | **677 days** | **3,846 days** |

**Train = issues that resolved quickly. Test = issues that languished for months to decades.**
This is guaranteed by construction: any issue still open at the Sep 2015 training cutoff date ends
up in the test set, and "still open after Sep 2015" implies long resolution time.

**Within-window temporal split (by created_at, 70/30):**

To isolate staleness from the split methodology, I re-split by `created_at` — all issues stay
in the 2014-15 era, but early-created issues train and late-created ones test:

| | Train | Test |
|---|---|---|
| n | 10,477 | 4,491 |
| Resolution median | 1.5 days | 3.1 days |
| Resolution p95 | 609 days | 705 days |
| LightGBM MAE | — | 86.5 days |
| Naive MAE | — | 90.9 days |
| Improvement | — | **+4.8%** |

The within-window test distribution is now comparable to training (both are 2014-15 era, similar
resolution shape). The MAE drops from 693 days to 86.5 days — **improvement is almost entirely
explained by fixing the split, not by the model**. The 2014-15 data itself is fine; the `closed_at`
split is what creates the broken evaluation.

---

## T1.3 — Intrinsic Difficulty Floor

Fitted four models on the within-window temporal split (by created_at) to establish the floor:

| Model | MAE (days) | vs Naive |
|---|---|---|
| Naive (predict train median) | 90.9 days | baseline |
| Text + temporal only (no leaky features) | 90.3 days | +0.7% |
| Full features, no label/assignee feats | 89.5 days | +1.5% |
| Full 93 features (with leaky label feats) | 86.7 days | +4.7% |

**Best model achieves only +4.7% improvement over naive.** This is on a within-window test whose
median is 3.1 days but p95 is 705 days — the distribution has massive right-skew regardless of era.

Text and temporal features alone are essentially useless (+0.7%). Adding non-leaky structural
features moves the needle by +1.5%. The small additional gain from leaky label features (+3.2pp
on top) is illusory at production time (see T1.4).

**Resolution time from issue-creation features is intrinsically near-unlearnable as a continuous
regression target.** The determinants of how long an issue takes to resolve are organizational
(who picks it up, team priorities, release cycles) and procedural (PR chain length, reviewer
availability) — none of these are captured by title/body/labels at creation time.

The honest upper bound from creation-time features is ~+5% over naive, which translates to
~86 days MAE on a test set with median 3 days and p95 700 days. The MAE number is dominated
by the extreme right tail; median absolute error would be much more informative.

---

## T1.4 — Leakage Audit

**`has_priority` is severely leaky — the top feature by gain (9,897) is near-unusable at
production time.**

| Feature | Risk | Evidence |
|---|---|---|
| `has_priority` | MEDIUM-HIGH | Priority fill: fast issues (<1d) 6.9% vs slow issues (>6mo) 93.1%. Correlation with log(res_hrs): **0.595**. k8s `priority/*` labels are applied during triage workflow, not at issue creation. |
| `has_component`, `comp_*` | MEDIUM | Component fill correlates 0.411 with log(res_hrs). Labels added during triage. |
| `num_assignees` | LOW | Scrape-time value; low correlation (0.045). Fast vs slow fill difference is small (28% vs 40%). |
| `num_comments` | NOT IN FEATURES | Correctly excluded. Would be severely leaky (comments accumulate throughout lifetime). |

**`has_priority` acts as a nearly direct proxy for long resolution time in training data** —
because issues that took months to close had their priority label set during that extended triage
period. At production time (issue just created), priority has typically not been set yet (only
6.9% of sub-day issues have it). The model learned a spurious correlation that will fail in
production.

The `author_prior_median_hrs` feature uses the resolution times of the author's past issues,
but those past issues may not be closed at the time the current issue is created. This is
technically leaky but has low practical impact for the k8s corpus (most authors have few issues
and the feature fills with −1 fallback).

---

## T1.5 — vscode vs k8s Comparison

vscode achieves +19.1% and 42.5% CI coverage using the same architecture. Why?

| | vscode | k8s |
|---|---|---|
| Train closed span | Nov 2015 – Nov 2020 | Jun 2014 – Sep 2015 |
| Test closed span | **Apr 22–27, 2026** | Dec 2015 – May 2025 |
| Train resolution median | ~3.9 days | 1.0 day |
| Test resolution median | **0.1 days** | 677 days |
| Train/test res. overlap | True | True (technically) |

**vscode has the OPPOSITE distribution shift**: train = slow historical issues (2015-2020
era, median 3.9d); test = very fast recent issues (Apr 2026, median 0.1d = ~2.4 hours).
All 616 vscode test issues closed in a single 5-day window (Apr 22–27, 2026) — the recent
2025-2026 scrape batch.

vscode's +19.1% happens because: (1) the test issues are fast-resolving recent issues, (2)
CI intervals calibrated on the slower train distribution happen to cover ~42% of the faster
test issues, and (3) vscode has more data and better label coverage than k8s.

Both repos suffer from broken `closed_at` temporal splits. vscode's split happens to produce
"easier" test issues (recent fast-resolvers) while k8s's split produces "harder" test issues
(decade-old slow-resolvers). Neither evaluation is a reliable indicator of production quality.

**The vscode +19.1% is partially an artifact, not a principled win.**

---

## Diagnosis: Ranked Failure Modes

| Rank | Mode | Evidence | Phase 2 fix |
|---|---|---|---|
| **1** | **Temporal split methodology failure** | `closed_at` split guarantees train/test target non-overlap. Fix the split → MAE drops from 693d to 87d (8× improvement). | Re-split by `created_at`. Both repos. |
| **2** | **Intrinsic task difficulty** | Within-window best model: +4.7% over naive. Extreme right skew (median 3d, p95 700d). Point regression is near-unlearnable from creation-time features. | Reframe to coarse ordinal buckets (hours / days / weeks / months / years+). |
| **3** | **Feature leakage** | `has_priority` (top feature, gain 9,897): 6.9% fill on fast issues vs 93.1% on slow. Correlation 0.595 with log(res_hrs). Provides false signal at production time. | Drop `has_priority` from production feature set; treat it as diagnostic only. |
| **4** | **Staleness (not primary)** | The 2014-15 k8s data itself produces learnable patterns when split correctly (+4.8% within-window). Staleness adds noise but is not the dominant factor. | Defer re-scrape until bucket reframe is validated on existing data. |

---

## Recommendation for Phase 2

The primary problem is the split methodology, not the data vintage. The actionable sequence:

1. **Re-split by `created_at`** for both repos. This is a code change in `scripts/03_split.py`,
   not a re-scrape. Fixes the broken evaluation immediately.

2. **Reframe output to 5 coarse ordinal buckets** rather than a point regression + interval:
   - "hours" (< 24h)
   - "days" (1–7 days)
   - "weeks" (1–4 weeks)
   - "months" (1–6 months)
   - "long" (> 6 months)

   These buckets cover the natural resolution-time modes and are predictable from creation-time
   features. The judge evaluates `resolution_estimate_reasonableness` qualitatively — a calibrated
   bucket ("likely takes weeks") is more useful than a point estimate that's 600 days off.

3. **Remove `has_priority`** from production feature set (keep for analysis). Add it back only
   with explicit leakage documentation for research use.

4. **Re-scrape k8s (Phase 3, if needed)**: If the bucket classifier trained on 2014-15 data
   underperforms after the methodology fix, then staleness may be limiting. Defer this decision
   to post-Phase 2 evaluation.

**Do NOT re-scrape before fixing the split.** The current broken evaluation makes the 2014-15
data look worse than it is. Fix methodology first, measure honestly, then decide if staleness
matters.

---

## Alternatives Considered

| Alternative | Reason deferred |
|---|---|
| Re-scrape k8s immediately (Phase 2) | Would re-run the same broken `closed_at` split on new data, producing the same failure. Fix split first. |
| Keep point regression, tune hyperparameters | Best achievable is ~+5% over naive; insufficient given extreme right skew. Diminishing returns. |
| Conformal prediction intervals on existing model | Addresses CI coverage symptom but not intrinsic unpredictability or leakage. |
| Drop resolution predictor entirely | The coarse-bucket reframe makes the problem tractable. Drop only if bucket classifier also fails. |
