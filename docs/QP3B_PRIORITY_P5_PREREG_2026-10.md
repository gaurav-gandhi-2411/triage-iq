# QP3b priority model P5: pre-registration (2026-10-29)

Written and committed BEFORE any P5 training. Do not change after the first validation result;
any later change is a new, dated amendment section, not an edit. Offline study only: no serving,
prompt or flag change (`TRIAGE_PRIORITY_MODEL` stays off; enabling anything is queued for GG).
Implementation: `scripts/qp3b_priority_p5.py` (reuses `qp3b_priority.py` / `qp1_*` helpers).

## Why P5 (decision G2, DECISION_LOG D88; return brief Q5)

P4 (`p4_tfidf_text_lr`, text only) is HELD because it over-predicts "high": test confusion
(rows gold low/medium/high) [[22,15,17],[19,40,33],[19,19,63]], validation [[9,14,7],[14,17,22],
[5,7,42]]; accuracy 0.506 vs majority 0.409 (test). Two causes are named in the brief: (a) in the
prioritised population "high" is the largest class and P4 inflates it (predicted 113 vs gold 101 on
test, 71 vs 54 on validation); (b) P4 is trained and evaluated only on issues a maintainer
prioritised (12.6% of the train split) but serving predicts for every issue.

## Design choice: (a) two-stage, with abstention

The API contract is `priority_guess: Literal["low","medium","high"]` (`models/triage.py`, the LLM
plan schema and the prompt); there is no "none" value and changing the response schema is out of
scope. A single model with a "none" class (option b) would therefore still need a rule for what to
serve when it says "none". Two stages make that rule explicit and measurable: when stage 1 says "no
priority will be assigned" P5 ABSTAINS and the existing behaviour (LLM-proposed priority) is kept;
when stage 1 says yes, stage 2 supplies the level. P5 thus never forces a level onto issues that
look unprioritisable. Stage 2 can be compared to P4 on exactly P4's rows.

## Inputs and features (serve-time only, identical to P4)

Text only: `title + "\n" + body[:4000]` (raw, as P4). No labels, no author association, no
metadata. Features = `qp1_tfidf_features.Featurizer(title_sep on, paths off)` fitted on the TRAIN
split of the model being fitted. Model family: LogisticRegression (class_weight balanced,
max_iter 300, random_state 42). P4's own design forbids nothing else but nothing richer is tried.

## Targets

* Stage 1 population = ALL issues of the split (no filter). Positive = the issue's FINAL labels
  carry at least one of the four priority levels (same rule as `qp3b_priority.gold4` not None;
  `priority/awaiting-more-evidence` alone is negative). This is "will ever be prioritised", not
  "has a label at creation+72h"; the 72h variant is reported only as a descriptive count.
  LIMITATION (BELIEVED): recent issues may not have been prioritised yet, which makes some
  negatives soft; the same bias applies to train, validation and test.
* Stage 2 population = the prioritised issues (exactly P4's rows), 3-level gold as P4
  (critical-urgent / important-soon -> high; important-longterm -> medium; backlog -> low).

## Selection (VALIDATION only; train split is fitted, test is not read)

* Stage 1: C in {0.25, 1, 4}; select by validation average precision (ties: smaller C). Gate
  threshold t = the probability maximising validation F0.5 of the positive class over the
  validation probabilities (F0.5, because a false "yes" forces a spurious level onto an issue; ties:
  higher t). Stage 1 is fitted on all train issues.
* Stage 2: C in {0.25, 1, 4}; class re-weighting of the predicted probabilities by a multiplicative
  vector (w_low, w_medium, w_high) = (1, m, h) with m in {1, 1.1, 1.2, 1.3, 1.5} and h in {1, 0.9,
  0.8, 0.7, 0.6, 0.5}; prediction = argmax(p * w). Select by validation macro-F1 (ties: accuracy;
  then smaller sum |ln w|; then smaller C). This is the "over-predicts high" correction; it is
  chosen on 137 validation rows, so overfitting to validation is a stated risk and the test
  result, not the validation one, decides.
* Freeze: after selection write `{stage-1 C, t, stage-2 C, (m, h), train majority}` and the sha256
  of that JSON (and of the P4 test prediction file used for pairing) to the report BEFORE the test
  split is read. `--final-test` refuses to run unless the freeze record exists, and refuses a
  second test evaluation in the same report.

## Metrics

On validation (reported at freeze): stage-1 average precision and ROC AUC vs prevalence, precision
and recall at t, gate pass rate; stage-2 accuracy, macro-F1, per-class recall, confusion matrix and
predicted-vs-actual distribution, next to P4 on the same rows.
On test, once: (1) stage 2 on P4's 247 rows: accuracy and macro-F1 vs P0 majority and vs P4 on the
same rows (paired bootstrap, 2000 draws, seed 42), per-class recall, confusion matrix, predicted vs
actual distribution; (2) stage 1 on all test issues: AP, AUC, precision/recall at t, gate pass rate;
(3) end-to-end on all test issues: of issues the gate passes, the share that are truly prioritised;
level served per gate-passed issue; served-high count as a share of ALL issues vs the gold high
share of all issues, for P5 and, for contrast, for P4 (which serves a level for every issue).

## Success criterion (all four must hold on TEST; otherwise P5 is reported as not met)

* C1: macro-F1(P5 stage 2) minus macro-F1(P4), paired bootstrap 95% CI lower bound > 0 on the same
  247 rows.
* C2: accuracy(P5 stage 2) minus majority accuracy, paired bootstrap 95% CI lower bound > 0 (P4's
  own bar is kept).
* C3: |predicted high share - gold high share| <= 3 percentage points on the 247 prioritised test
  rows (P4: 4.9 pp).
* C4: end-to-end served-high share of ALL test issues (gate pass AND stage-2 = high) is at most 5
  percentage points above the gold share of high-prioritised issues among all test issues, AND
  stage-1 precision at t exceeds test prevalence (lift > 1) with the AUC bootstrap 95% CI lower
  bound > 0.5.

## Test discipline

P5 is evaluated on test exactly once, after the freeze record is written. No test outcome is read
before then (P4's saved test predictions are read only inside `--final-test`). Models stay under
`D:\triageiq-data\models\kubernetes__kubernetes\priority\p5\`; nothing is uploaded. Packaging P5 and
any flag change are out of scope and queued for GG.
