# ADR-0071 - vscode conformal method: LAC at alpha 0.1, chosen on validation before the test split was read

Status: Accepted (unattended run; frozen before test)
Date: 2026-10-23
Decider: orchestrator, unattended run under spec-autonomous-run.md Q4; ADR-0069 decision 4 requires this ADR before the
vscode test split is evaluated. Open to review by Gaurav Gandhi.

Evidence: `reports/qp1_vscode_phase1_2026-10.json` (keys `conformal_selection`, `validation_grids`, `selected_model`),
produced by `scripts/qp1_vscode_phase1.py --stage validation` and `--stage conformal` at commit `9567774`. Validation
labels only; the test split had not been read when this was written.

## Context

The vscode classifier selected on validation is `tfidf_lr_family_only_v2` (TF-IDF + multinomial LR, C 64, balanced,
trained on in-family issues only; validation strict top-3 0.7797 vs 0.7621 for the k8s-style V1 with an `other` class,
0.1401 for the majority prior). It has no `other` column (probability 0), so a gold-`other` issue (11.3% of labelled
validation issues, ADR-0070) can never be inside a prediction set. Conformal sets are therefore calibrated and
evaluated on in-family issues only (n = 5,691 of 6,419 labelled eligible validation issues).

## Decision

1. **Method: LAC (score = 1 - p of the class), alpha = 0.1 (target coverage 0.90 w.r.t. the primary label),
   calibrated on the full in-family validation set, applied to test.** No randomisation is involved for LAC.
2. Selection protocol (declared in the script docstring before running): candidates APS non-randomised, APS randomised,
   RAPS (lambda 0.1, k_reg 1, the k8s setting), LAC; 50 seeded random 50/50 splits of the validation population
   (seed 42; calibrate on one half, evaluate on the other); choose mean coverage within 0.90 +/- 0.02, then the smallest
   mean set size. Results (mean over 50 splits; sds are in the report):

   | Score | Coverage | Mean set size | Within 0.90 +/- 0.02 |
   |---|---|---|---|
   | APS non-randomised | 0.9721 | 41.54 | no (over-covers, near-vacuous) |
   | APS randomised | 0.8985 | 13.54 | yes |
   | RAPS (0.1, 1) | 0.8987 | 3.76 (sd 0.08) | yes |
   | **LAC** | 0.8985 | **3.60** (sd 0.17) | yes |

   LAC beats RAPS by 0.17 labels, about one split-to-split standard deviation: close to a tie, decided by the
   pre-declared rule, not by judgement. The k8s choice (RAPS) is not carried over because ADR-0069 decision 4 says
   vscode is chosen on its own validation. (ADR-0069 used alpha 0.05 for k8s; this run uses 0.1 as the task brief
   specified, so the two are not directly comparable.)
3. **All vscode conformal numbers are EXPLORATORY** (ADR-0069 decision 2): calibration (2025-H2) and test (2026) are
   separated in time, so exchangeability is approximate. Test coverage is reported with Wilson intervals for in-family
   issues, and also over all labelled issues (where gold-`other` issues are uncoverable by construction, so that
   figure is bounded by the in-family share).
4. Confirmation set as ADR-0069 decision 4: issues created after the vscode test split's last `created_at`, at least 500
   labelled, at least 30 days old, run once with the frozen model and method.

## Consequences

- Sets are for in-family routing only; a gold-`other` (new product area) issue is not covered. The abstention flag
  (max probability < tau = 0.3054, chosen on validation for abstention F1) is the open-set signal, separate from the sets.
- Mean set size about 3.6 of 85 classes at 90% in-family coverage on validation.
- The test stage refuses to run unless this file names the chosen score.

## Alternatives

- RAPS (0.1, 1) as in k8s: 0.17 labels larger on validation; runner-up, not tried again.
- APS (either form): rejected on validation, set sizes 13.5 and 41.5.
- Conformal over all labelled issues including `other`: not possible for the selected model (its `other` probability is 0);
  would require the V1 model, which lost on the headline.
- Choosing on test: rejected, ADR-0069.
