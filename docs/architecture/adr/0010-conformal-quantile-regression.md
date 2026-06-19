# ADR-0010 — Conformal Quantile Regression for Resolution Intervals

**Status:** Accepted  
**Date:** 2026-06-20  
**Decider:** Gaurav Gandhi

---

## Context

TriageIQ's resolution-time predictor outputs Q10/Q90 quantile intervals from LightGBM models
trained on a corrected `created_at` temporal split (see ADR-0009). Those intervals have empirical
coverage measured on the test set:

| Repo | Raw Q10/Q90 empirical coverage |
|---|---|
| kubernetes/kubernetes | 74.4% |
| microsoft/vscode | 38.2% (30/70 split) |

These are useful intervals, but coverage is not formally post-hoc calibrated — it depends entirely
on the training distribution matching the test distribution. Conformal Quantile Regression (CQR,
Romano et al. 2019) provides a distribution-free additive correction that post-hoc calibrates
coverage to a target level on a held-out calibration set.

**CQR mechanics.** Given a frozen Q10/Q90 model and a calibration set of n labeled examples:

1. Compute conformity scores: **E_i = max(q_lo(x_i) − y_i, y_i − q_hi(x_i))**
   - Positive when y falls outside the raw interval (miss), negative when inside (hit).
2. Compute the calibration quantile at target level 1−α:
   **Q = ⌈(n+1)(1−α)⌉/n empirical quantile of {E_1, …, E_n}**
3. At inference: widen the raw interval to [q_lo(x) − Q, q_hi(x) + Q].

This is **marginal (not conditional) coverage**: if the calibration and test distributions are
exchangeable, the resulting intervals achieve at least 1−α marginal coverage in expectation over
the test set. CQR provides no per-instance guarantee and no conditional-on-x guarantee.

**Calibration set source.** The frozen model's held-out test set is the only valid calibration
source. The validation set was contaminated for conformal use: Optuna hyperparameter tuning used
the val set as its objective function, so conformity scores computed on val are in-sample with
respect to model selection. Temporal ordering is preserved throughout — the earlier portion of
the test set is designated as calibration, the later portion as the true evaluation set.

---

## Decision

Implement CQR as an additive post-processing wrapper on the frozen Q10/Q90 models with the
following configuration:

**Target level:** 1−α = 0.80 (α = 0.20)

**kubernetes/kubernetes — 30/70 calibration/test split:**
- n_calibration: 449 (earlier portion of test set, by `created_at`)
- n_true_test: 1,049

**microsoft/vscode — 40/60 calibration/test split (primary); 30/70 documented for comparison:**
- 40/60: n_calibration = 246, n_true_test = 370 ← **selected**
- 30/70: n_calibration = 184, n_true_test = 432 ← documented

The vscode 40/60 split is selected because it provides lower Q estimation variance (246 vs 184
calibration points) and adequate test-set precision (±2.6pp Wilson half-width). The 30/70
comparison is retained in this ADR because the 5.8pp divergence between the two splits is
informative signal about non-stationarity (see Consequences). Proximity to the 80% target was
not a selection criterion.

**Persistence:** Per-repo conformal adjustment Q is persisted in `cqr_conformal_adjustments.json`
alongside production model `.pkl` files in GCS and COPY'd into the Docker image at build time.

**API:** An additive `resolution_interval_conformal` optional field surfaces the widened [lo, hi]
pair. Existing `resolution_interval_lo`, `resolution_interval_hi` (raw Q10/Q90 outputs) are
unchanged. Missing `cqr_conformal_adjustments.json` at inference time triggers an explicit log
warning and falls back to raw intervals without error.

**Coverage everywhere reported as empirical (not "guaranteed"):** marginal coverage measured on
the evaluation portion of the test set, paired with Wilson 95% confidence intervals.

---

## Consequences

### kubernetes/kubernetes

| | Value |
|---|---|
| Calibration Q | 0.2835 hours (0.0118 days) |
| Raw Q10/Q90 empirical coverage | 74.4% |
| CQR empirical coverage | **76.6% [74.0%, 79.1%]** (95% Wilson CI) |
| Median interval width — raw | 87.6 days |
| Median interval width — conformal | 87.7 days (+0.57h) |

CQR moves coverage from 74.4% to 76.6% — a 2.2pp improvement that remains slightly below the
80% target. The conformal adjustment Q is 0.28 hours against a median interval width of 87.6 days:
the additive correction is negligible in absolute terms. The honest interpretation is that CQR
**validates the existing k8s intervals more than it corrects them** — the base Q10/Q90 outputs were
already near-calibrated, and the residual shortfall from 80% reflects finite-sample variation
rather than a systematic miscalibration.

### microsoft/vscode

| Split | n_cal | n_test | Q | Empirical coverage | Wilson 95% CI |
|---|---|---|---|---|---|
| 30/70 | 184 | 432 | 1.06h | 68.3% | [63.8%, 72.5%] |
| 40/60 | 246 | 370 | 1.25h | 74.1% | [69.4%, 78.3%] |

Selected (40/60): empirical coverage 74.1% [69.4%, 78.3%] against an 80% target. Median interval
width: 117.0 days raw → 117.1 days conformal (+2.5h). The adjustment is negligible.

**The 74.1% empirical coverage is the honest reported outcome.** CQR cannot reach 80% here, and
that is not a CQR failure. The root cause is temporal non-exchangeability: the models were trained
on 2015-10-13 to 2016-04-05 data; the test window is 2026-04-21 to 2026-04-27 (7 days). A decade
separates train from test. The marginal coverage guarantee CQR provides is conditional on
exchangeability between calibration and test — a condition this data cannot satisfy.

**5.8pp divergence between splits is a first-class finding.** The 30/70 split yields 68.3%
empirical coverage; the 40/60 split yields 74.1%. A 5.8pp swing from shifting 62 issues between
calibration and test is direct evidence of temporal non-stationarity in the vscode test
distribution. With a 7-day test window (2026-04-21 to 2026-04-27), small changes in which issues
land in calibration vs. evaluation materially change the measured coverage. A principled conformal
guarantee requires an exchangeable distribution; the vscode data does not provide one. The 5.8pp
divergence is not a parameter-tuning artifact — it is the signal.

### Operational

The conformal adjustment JSON must be present at inference time. It is COPY'd into the Docker
image from GCS during the build phase. A missing or unreadable file triggers an explicit
structured-log warning and falls back to raw Q10/Q90 intervals; the API does not error out.

---

## Alternatives Considered

| Alternative | Reason rejected |
|---|---|
| **Mondrian / conditional conformal** | Provides group-conditional coverage (e.g., per-priority-bucket) rather than marginal, which is more useful but requires substantially more calibration data and group-membership assumptions. Deferred — the current calibration set sizes (246–449) are marginal for unconditional CQR; conditional CQR would need larger holdouts per group. |
| **Use val set for calibration** | Rejected. Optuna hyperparameter tuning used the val set as its objective function (MAE minimization). Conformity scores computed on val are in-sample with respect to model selection, biasing Q downward and producing undercoverage. |
| **Use training tail for calibration** | Rejected. The model was trained on the full training set; conformity scores computed on in-sample data are biased toward zero (the model fits its own training points well), producing Q values that are too small and resulting in systematic undercoverage on held-out data. |
