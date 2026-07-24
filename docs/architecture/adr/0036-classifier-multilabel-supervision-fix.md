# ADR-0036 — Component Classifier: Multi-Label Supervision Fix (Ship, Cutover Staged)

**Status:** Accepted — ship the multi-label TF-IDF+LR classifier, pending the cutover plan below
(staged, **not yet deployed** — escalated for GG's go)
**Date:** 2026-07-24
**Decider:** Gaurav Gandhi (CPU-only experiment executed autonomously by CC while Phase B/DeBERTa
stays GPU-blocked; verification + ADR + cutover staging per GG's explicit instruction)

---

## Context

Phase A (classifier eval audit) found no analogous bug to the retrieval-harness corrections
(ADR-0035) — production/train/eval query construction, leakage, and label accounting were all
clean. But it re-confirmed a known, quantified defect: `preprocess.py::normalize_labels()` keeps
only the *first* matching component label per issue and discards the rest — **30.4% of k8s test
issues and 8.0% of vscode's have more than one valid component label**, collapsed to one at
preprocessing time. This is a **supervision** defect (the training signal itself is wrong for
30%/8% of examples), not an architecture question.

Phase B (DeBERTa-v3-base, two arms) was designed to test this against a transformer architecture,
but GPU access is blocked (AetherArt's continuous, non-cyclical GPU contention — confirmed via
10s-interval sampling; a contention-aware pace-check retry at a much smaller config still could
not complete a single optimizer step in 5 minutes with 97%+ VRAM in use, ruling out "just reduce
the config" as a workaround). Rather than idle-wait (new standing rule, CLAUDE.md 46a), GG
redirected to the highest-value CPU-only experiment available: test the supervision fix directly
on the **currently shipped** TF-IDF+LR classifier, independent of any transformer question.

## Method

**Multi-label retraining**: one-vs-rest logistic regression over ALL valid component labels per
issue (multi-hot targets built via `classifier_eval.py::all_matching_component_labels()` — the
same function that measured the 30.4%/8.0% collapse and that Phase B's DeBERTa ARM 2 uses),
instead of the single collapsed `component` column. Identical TF-IDF feature extraction to the
shipped baseline (`max_features=50000, ngram_range=(1,2), stop_words="english",
strip_accents="unicode", min_df=2, sublinear_tf=True`) — only the supervision changes.

**Bug caught and fixed before trusting any result**: the first run, using the baseline's
`solver="saga"`, produced a catastrophic collapse (k8s macro-F1 ≈0.0002, top-1 ≈0.35%, only
15/35 classes ever predicted). Diagnosed rather than accepted: confirmed even at `max_iter=5000`
the identical degenerate collapse persisted — `saga` does not converge on the 35 independent,
far-more-imbalanced binary sub-problems `OneVsRestClassifier` creates (it is tuned for the
original single 35-way multi-class softmax problem, a different optimization landscape).
Switched to `solver="liblinear"` (sklearn's standard recommendation for per-class L2-regularized
binary logistic regression) — converges cleanly, uses the full label space (27-30/35 classes
actually predicted vs. `saga`'s 15/35).

**Leakage/methodology re-assert (GG's verification #2)**: `scripts/classifier_assert_leakage_guard.py`
called as a hard pre-flight gate (`assert_leakage_guard_passed()`, `scripts/tfidf_multilabel.py`
line ~60) — re-confirmed 0 issue-number overlap across train/val/test, both repos, same as every
other check this session. Evaluation uses `classifier_eval.py::evaluate_classifier()` — the
identical function, byte-for-byte, used for the shipped baseline, the DistilBERT re-eval
(ADR-0035-adjacent correction), and Phase B's DeBERTa arms. No new metric definitions were
introduced; top-3/top-1/any-valid-label-credit/macro-F1 are computed exactly as established.

### Results — paired bootstrap vs. the shipped single-label baseline (loaded, not retrained — see Alternatives)

| Repo | Metric | Baseline | Multi-label | Paired Δ | 95% CI | Excludes zero? |
|---|---|---|---|---|---|---|
| vscode (n=187) | Top-1 | 68.98% | 76.47% | **+7.49pp** | [+2.67, +12.83] | **Yes** |
| vscode | **Top-3 (ship bar)** | **90.37%** | 89.84% | −0.53pp | [−4.28, +3.21] | No |
| vscode | Macro-F1 | 0.585 | 0.627 | — | — | — |
| vscode | Any-valid-label top-1 | 71.7% | 79.14% [72.8, 84.4] | — | — | — |
| k8s (n=286) | Top-1 | 51.40% | 60.49% | **+9.09pp** | [+3.85, +14.69] | **Yes** |
| k8s | **Top-3 (ship bar)** | **82.52%** | **87.06%** | **+4.55pp** | **[+0.35, +8.39]** | **Yes** |
| k8s | Macro-F1 | 0.466 | 0.462 | — | — | — |
| k8s | Any-valid-label top-1 | 59.4% | 72.73% [67.3, 77.6] | — | — | — |

**Tail-class recall (<15 train examples)**: mixed, small-n as expected — some newly recoverable
(k8s `rkt` 0→1.0, `provider/gcp`→1.0; vscode `editor-find` 0→1.0), several still 0 (data-limited
regardless of supervision — a data-volume ceiling, not something this fix or any architecture
change resolves).

## Verification 1 — Calibration (GG's explicit gate before shipping)

**Does row-sum≠1 break anything downstream?** Checked every consumer of classifier confidence:
- `grounding.py::verify_plan_grounding` — set-membership check on **labels only**
  (`top3_labels = {entry["label"] for entry in classifier_top3}`); never reads or sums confidence
  values. Grounding logic is unaffected by probability semantics.
- `triage.py:566` — `component_confidence=float(top.get("confidence", 0.0))` takes the single
  top-1 confidence value directly; no normalization or sum-to-1 assumption anywhere.
- `TriageResponse.component_confidence` (schemas) — Pydantic validates `ge=0.0, le=1.0` on one
  float; no cross-field or sum constraint.
- The LLM prompt passes `classifier_top3` (label+confidence pairs) as textual context; the LLM
  reads these qualitatively, not via any code path that assumes a normalized distribution.

**Verdict: nothing in the codebase computationally requires probabilities to sum to 1.** Confirmed
empirically that they do not: independent-sigmoid row sums average **3.95 (vscode) / 4.93 (k8s)**,
not ~1.0 like a softmax.

**Is the confidence itself still meaningful?** Computed ECE with the *exact* ADR-0004 methodology
(`calibration_analysis()`), on the new model's raw, uncalibrated output, against the shipped
baseline's already-calibrated ECE:

| Repo | Baseline ECE (calibrated, shipped) | Multi-label ECE (raw, uncalibrated) | Multi-label mean_conf vs mean_acc |
|---|---|---|---|
| vscode | 0.1381 | **0.0863** (better) | 0.722 vs 0.765 — still underconfident, less severely |
| k8s | 0.1558 | **0.1114** (better) | 0.706 vs 0.605 — **flips to overconfident** |

The raw multi-label ECE is *numerically better* than the baseline's already-calibrated ECE on
both repos — but k8s **flips direction** (from the original severe underconfidence pattern to a
mild ~10pp overconfidence), a genuine semantic change even though the magnitude looks fine.
Top-3 shape check: independent sigmoids still decay sensibly in rank order (mean top-3 values
0.72→0.38→0.28 vscode, 0.71→0.53→0.37 k8s) — not degenerate — but **11.5% of k8s test rows**
(1.6% vscode) show all three top labels simultaneously above 0.5 confidence, something a
single-distribution softmax structurally cannot produce. Arguably this is *more honest* behavior
for genuinely multi-labeled issues (independent presence-probabilities, not a forced ranking) —
but it is a real, user-facing semantic shift from "P(this is the one correct label)" to "P(this
label applies, independently)," not a silent equivalent swap.

**Proposed fix (not yet implemented — part of the cutover plan below, not this ADR's scope):**
re-run ADR-0004's temperature-scaling procedure against the multi-label model's own top-1
confidence stream (same technique — fit scalar T minimizing NLL/ECE on the val split,
argmax-preserving by construction) to correct k8s's newly-introduced overconfidence before
shipping to production. This is required in the cutover plan, not optional polish.

## Verification 2 — Leakage/methodology re-assert

Confirmed via direct code citation, not re-derivation: `assert_leakage_guard_passed()` (calling
`scripts/classifier_assert_leakage_guard.py`, the same script committed for Phase B) runs as the
first line of `tfidf_multilabel.py::main()` — hard-fails before any training if disjointness is
violated. `evaluate_classifier()` is imported and called identically to every other classifier
evaluation this session (baseline, DistilBERT re-eval, and Phase B's DeBERTa arms all share this
one function) — no metric-definition drift between the numbers being compared.

## Decision: SHIP (cutover staged, not deployed)

**Why this clears the bar where ADR-0031's weighted fusion didn't, despite a superficially
similar CI shape (GG's reasoning, recorded verbatim for the decision record):**

ADR-0031's weighted fusion had a marginal k8s CI **and** a vscode regression — inconsistent
direction across repos is a noise signature. This result has **consistent direction**: three
separate CIs excluding zero across both repos (k8s top-1 +9.09pp [+3.85,+14.69], vscode top-1
+7.49pp [+2.67,+12.83], k8s top-3 +4.55pp [+0.35,+8.39]). vscode's top-3 is flat for a
**mechanical, explainable reason**: 90.4% is near ceiling, and vscode's multi-label collapse rate
(8.0%) is small — the defect being fixed barely exists there, so there is little room for top-3
to move even though top-1 (a less saturated metric) shows the same underlying effect clearly.
That is a coherent effect with a diagnosed mechanism (the label-collapse rate predicts the effect
size), not one lucky threshold-scrape.

**Top-1 also matters practically**: it's what `component_confidence` reflects and what a human
reading the API response sees as *the* predicted component, even though `grounding.py`'s
correctness definition is top-3. A +9pp/+7.5pp top-1 gain at zero additional inference cost
(same TF-IDF features, same latency class) is real, immediately deployable product value
independent of the top-3 ship-bar question.

## Consequences for Phase B (DeBERTa)

- **ARM 2 (multi-label BCE) value is substantially raised.** This result is direct evidence the
  supervision defect is a real, fixable bottleneck, not merely a caveat — a transformer with the
  same supervision fix, more model capacity, and longer-range text understanding could plausibly
  extend this gain further, particularly on vscode where TF-IDF's top-3 hit its ceiling.
- **ARM 1 (single-label softmax) value is lowered further.** A single-label transformer
  (DistilBERT) already lost to TF-IDF on the real ship bar once (ADR-0035-adjacent correction:
  88.2% vs. 90.4% vscode, 74.5% vs. 82.5% k8s). Combined with this result showing the *supervision*,
  not the architecture, is where the real lever lives, ARM 1 is now the less promising of the two
  arms — still worth its one measure-first run per the original design, but expectations should be
  calibrated accordingly.

## Cutover plan (STAGED — escalating for GG's go, not deploying)

1. **Fix calibration first** (Verification 1's proposed fix): re-fit temperature scaling on the
   multi-label model's top-1 confidence stream, val split, same argmax-preserving method as
   ADR-0004. Re-check ECE/overconfidence direction before proceeding.
2. **Re-save model artifacts**: `TFIDFComponentClassifier`-equivalent save for the multi-label
   OvR model (vectorizer + OvR classifier + label encoder + new calibrator), to
   `data/models/component_classifier_{repo}.pkl` — **only after** step 1, and only as a staged
   local artifact, not uploaded.
3. **Update `MANIFEST.sha256`** (`scripts/publish_models.py`) — but do NOT run
   `scripts/publish_models.py`'s GCS upload step without GG's explicit go; this repeats the exact
   local/GCS mismatch class of incident ADR-0004's Deployment Correction documented (calibrated
   pkl committed locally 2026-05-19, GCS not updated until 2026-06-20 — production served stale
   uncalibrated weights for 6 weeks, undetected until the CI gate caught it).
4. **Re-record eval numbers**: `reports/classifier_results.json` (or a new
   `reports/classifier_results_multilabel.json`), README table update — same pattern as every
   other correction this session (old numbers archived, not deleted).
5. **Re-baseline any downstream gate** that references the shipped classifier's accuracy numbers
   (CI eval-regression tests, `test_calibration_ece_in_tolerance` per ADR-0011/ADR-0004) — these
   thresholds were tuned to the single-label model's calibrated ECE and will need updating to the
   new model's numbers, or they will fail (correctly) on the next CI run.
6. **Drift guard**: run `scripts/verify_model_manifest.py` after any GCS upload to confirm the
   live artifact matches the new manifest — this is the exact mechanism that caught the
   ADR-0004 deployment gap; use it, don't skip it.
7. **Rollback anchor**: record the current Cloud Run revision ID serving the single-label model
   *before* any deploy, so a regression can be rolled back by traffic-shifting to that revision
   without a redeploy.
8. **Live verify**: after deploy, hit `/triage` for a known issue and confirm `classifier_top3`
   and `component_confidence` reflect the new model (same verification pattern ADR-0004's
   Deployment Correction used: `vscode#2093` returned a specific, checkable `component_confidence`
   value post-fix).

**Nothing in steps 2-8 has been executed.** This is a staged plan for GG's review, not a completed
deployment — per the standing rule, cutover/deploy decisions escalate separately from the
measurement decision.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Retrain the single-label baseline fresh for the paired comparison | Rejected on a documented past incident: `LogisticRegression`'s `saga` solver has no fixed seed, and a prior session's bare retrain silently overwrote the shipped `.pkl` non-reproducibly. Loaded the existing shipped artifact instead — the comparison is against what's actually in production. |
| Keep `solver="saga"` for consistency with the baseline's exact config | Would ship a demonstrably broken model (macro-F1 ≈0.0002) in the name of superficial consistency. `saga`'s failure mode here is diagnosed (doesn't converge on OvR's imbalanced binary sub-problems), not a preference — `liblinear` is the correct tool for this specific optimization problem. |
| Treat k8s's CI shape as automatically disqualifying (ADR-0031 precedent) | GG's call, recorded above: the precedent doesn't transfer because the failure signature differs (inconsistent cross-repo direction vs. this result's consistent direction across three independent CIs) and this result has a diagnosed mechanism the ADR-0031 lever never had. |
| Ship without addressing k8s's overconfidence flip | Explicitly against GG's instruction — "don't ship an accuracy gain that silently breaks confidence semantics." Folded into the cutover plan as a required first step, not shipped silently. |
