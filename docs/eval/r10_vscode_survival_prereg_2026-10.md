# R10 pre-registration: microsoft/vscode resolution time as a survival problem, no embeddings

Status: PRE-REGISTERED 2026-10-29, committed before any validation number was computed.
Do not change after the validation commit. Nothing here touches production.

## Question and hypothesis

H1: a censoring-aware survival model on creation-time, non-embedding features predicts vscode
resolution time better than (a) the SERVED estimator and (b) the naive baseline, on the frozen
temporal test split.

- Served estimator: the constant `median(resolution_hours of data/processed/microsoft_vscode_
  temporal_train.parquet) / 24` = 3.8374 d (ADR-0064: `POINT_ESTIMATE_TRUSTED[vscode] = False`, the
  training-window median is served). The script asserts the value from the parquet.
- Naive baseline (M0a): median resolution of CLOSED issues in the new frozen train split (repo
  convention, `scripts/qp4_resolution.py`). Extra reference (not gated): train Kaplan-Meier median.
- Prior art for conventions: `scripts/qp4_resolution.py` (k8s survival study, D84(6)). It used
  BGE embedding features; this study uses none. Primitives (KM, IPCW, Harrell C, CQR, hazard bins)
  are imported from it, not re-implemented.

## Data and split (reused, not invented)

- Issues: `D:\triageiq-data\derived_v2\microsoft__vscode\issues.parquet`; splits: frozen
  `D:\triageiq-data\splits\microsoft__vscode\{train,validation,test}.parquet`
  (`reports/qp0_split_manifest_microsoft__vscode_2026-10.json`; train < 2025-07-01, validation
  2025-07-01..2025-12-31, test >= 2026-01-01, test strictly later).
- Dedupe: the frozen `classifier_eligible` flag (no text-hash / duplicate-pair leakage across splits;
  for vscode also >= 1 component label, ADR-0070). Primary population = eligible issues of each split.
  Secondary DESCRIPTIVE population (not in the criteria): all non-leaky issues
  (`~(leak_texthash | leak_dup_pair)`) of validation/test, scored with the model trained on
  eligible train only. Its purpose is to show how the result transfers to issues that never got a
  component label (they resolve much faster).
- Target: T = `resolution_hours` (floored at 0.01 h) for closed issues; open issues are
  right-censored at the scrape cutoff `scraped_at_max` (2026-10-22 21:21:46 UTC), T = hours from
  `created_at` to the cutoff. Event = closed. Same convention as qp4.
- Known limitation, stated up front: train outcomes use follow-up up to the cutoff (as qp4), so
  train labels for late-2025 issues are "seen" later than test creation; this is the repo's
  convention and affects all candidates and baselines equally.

## Features (creation time only; no embeddings, no TF-IDF, no classifier probabilities)

Every feature of an issue created at t_q uses only information existing at t_q.
1. Own text and calendar: author_association (NONE 0, CONTRIBUTOR 1, COLLABORATOR 2, MEMBER 3),
   title/body length, body line count, code fence, stack-trace regex, URL count, image count,
   "steps to reproduce" mention, VS Code version line, "insiders" mention, '?' in title, hour/day of
   week/month (sin and cos).
2. Labels at creation: number of labels applied within 60 s of `created_at` (`label_events`
   action `labeled`) and 0/1 flags for the 20 most frequent such labels among eligible TRAIN issues.
3. Author history: earlier issues by the same author, how many closed by t_q, median resolution of
   those (`qp4_features.author_history`; an earlier issue counts as resolved only if closed_at <= t_q).
4. Repo load and recent speed (all issues in the table, not only eligible): open issues at t_q,
   issues created / closed in the 7 days before t_q, number and median resolution hours of issues
   closed in the 90 days before t_q (a drift-aware feature that needs no absolute time).
   No absolute-time feature (the ADR-0064 Zd finding shows the danger of extrapolating one).
Missing values: median of train (plus a missing flag for the two history medians); counts, lengths
and hours enter as log1p; all columns standardised with train mean/std; zero-variance train columns
dropped. One matrix feeds every family.

## Candidates (4 families, small fixed grids, seed 42)

| id | family | grid |
|---|---|---|
| F1 | penalised Cox, elastic net (scikit-survival `CoxnetSurvivalAnalysis`) | alpha {0.001, 0.01, 0.1} x l1_ratio {0.1, 0.5} |
| F2 | parametric AFT (lifelines) | {log-normal, Weibull} x penalizer {0.01, 0.1} |
| F3 | gradient-boosted Cox (scikit-survival `GradientBoostingSurvivalAnalysis`, loss coxph) | n_estimators {100, 200} x max_depth {2, 3}; learning_rate 0.1, subsample 0.8, min_samples_leaf 50 |
| F4 | discrete-time hazard LightGBM (qp4 M1 bins and expansion) | num_leaves {15, 31} x n_estimators {200, 400}; learning_rate 0.05, min_child_samples 50 |

CPU only (`CUDA_VISIBLE_DEVICES=-1`). A candidate whose single fit would exceed 30 min is dropped
and reported. Packages (scikit-survival, lifelines) live in the throwaway venv
`D:\triageiq-data\venvs\r10_surv`; their versions are recorded in the report.

Survival and point: Cox families: Breslow baseline from the training linear predictor,
S(t|x) = exp(-H0(t) exp(lp)); AFT: the fitted distribution; F4: qp4 hazards with exponential-in-bin
interpolation. POINT = predicted median survival time (S = 0.5), capped at 730 d when S never reaches
0.5; no recalibration of the point. Ranking score for C-indexes: higher = faster resolution
(Cox: lp; AFT: -mu; F4: -sum of S over the 9 finite bin edges).

## Metrics (all on closed issues unless "all")

- Harrell C (all issues, qp4 definition) and Uno C (IPCW, G from train, truncated at tau = 365 d).
- Median absolute error of the point, days, closed issues (the repo serving rule's metric).
- IPCW mean absolute error, days: closed issues weighted 1/G(T_i), G = KM of the censoring
  distribution of the EVALUATED split (administrative censoring is a property of the split's
  calendar window; floor 0.01), Hajek-normalised: sum(w |err|) / sum(w). Closed-only error favours
  fast issues; this adjusts for it. Weights are computed once per split and held fixed across
  bootstrap resamples (stated approximation).
- IBS: mean over the 9 bin edges of the IPCW Brier score (qp4 definition, G from TRAIN; split-G as
  sensitivity), also for the KM baseline curve.
- Calibration (descriptive): mean predicted P(T <= t) vs the KM estimate on the evaluated split at
  1, 7, 30, 90 d, and by quintile of predicted P(T <= 7 d).
- Interval, EXPLORATORY: 10%/90% predicted quantiles, CQR-adjusted on VALIDATION closed issues in
  log1p-hours space (qp4 `cqr_q`, target 80%), coverage and width on test closed issues; the
  constant comparison is the M0a train-closed 10/90 quantiles with the same CQR. The served vscode
  interval (re-centred model width plus stored Q) is not re-created here.
- Bootstrap: 95% percentile, 2000 resamples, seed 42 (qp1_eval.bootstrap), paired differences on
  the same indices. Exception: Harrell/Uno C use 500 resamples (O(n^2) pair matrices; stated), 100 on the secondary population.
  Amendment 2026-10-29, still before any validation number: the script treats a candidate fit that
  raises as a reported failure (not a silent skip), and the C bootstrap count above was added.

## Selection (validation only)

Within a family, the grid point with the highest validation Harrell C (ties: lower validation median
AE on closed issues, qp4 rule). Across families, the family whose selected configuration has the
lowest validation IPCW mean absolute error (days) is the single CHAMPION. Only the champion and
the baselines are scored on test (no multiplicity across families). Validation results for every
family and baseline are written to the report and committed together with a freeze record
(`freeze`: champion, hyperparameters, `test_read: false`, sha256 of the pre-registration and split
manifest) BEFORE the test is read.

## Success criteria (test, champion vs served constant AND vs naive; all must hold)

- C1 ranking: Harrell C 95% CI lower bound > 0.5 (a constant predictor has C = 0.5 exactly).
- C2 typical error: paired bootstrap 95% CI of [median-AE(baseline) - median-AE(champion)] on test
  closed issues excludes zero in the champion's favour, against each of the two baselines.
- C3 censoring-adjusted error: same for the IPCW mean absolute error (paired over all test issues).
- Not gating, reported: Uno C, IBS vs KM, calibration, the interval table, the secondary population,
  mean-AE gain.
Result wording: "met" only if C1, C2 and C3 all hold against both baselines; otherwise the partial
pattern is reported as is. A negative result is a valid result.

## Test protocol

The test is read exactly once (authorised by GG for this run). The script refuses the final stage if
the report already contains a test entry or `freeze.test_read` is true, and flips it to true before
any test target is read. The test result is committed as its own commit. No code, feature, grid or
criterion changes after the validation commit.

## Reproduction

```
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src D:\triageiq-data\venvs\r10_surv\Scripts\python.exe \
    scripts/r10_vscode_survival.py            # validation + freeze
    scripts/r10_vscode_survival.py --final-test   # once
```
