# ADR-0069 — Conformal method frozen (RAPS), results exploratory until confirmed on a post-test confirmation set

Status: Accepted (frozen 2026-10-20)
Date: 2026-10-20
Decider: Gaurav Gandhi (decision of 2026-10-20: the RAPS choice was made after a test peek, so its result is
exploratory until confirmed)

## Context

The pre-registered conformal score (non-randomised APS) gave formally valid but vacuous prediction sets on the k8s
`sig/*` baseline: test coverage 0.995, mean set size 12.9 of 22 classes (`reports/qp1_tfidf_lr_kubernetes__kubernetes_2026-10.json`).
An independent verifier then showed on the test split that LAC and randomised APS give far smaller sets, so the
score family was widened and the score chosen on VALIDATION only (50 seeded 50/50 splits of the validation set;
smallest mean set size among scores with mean coverage >= 0.95; `scripts/qp1_conformal.py`). That selection picked
RAPS (lambda 0.1, k_reg 1) for the k8s baseline: test coverage 0.963 primary / 0.977 any-valid, mean set size 2.77.
Because the test split had already been seen when the family was widened, that result is **exploratory**. It is also one
randomisation draw (other draws moved coverage by about +-0.003 and set size by about +-0.1), and calibration
(2025-H2 validation) and evaluation (2026 test) are separated by a time gap, so exchangeability is only approximate.
On validation RAPS(0.1, 1) and LAC were a statistical tie (coverage 0.9501 vs 0.9493, sizes 2.89 vs 2.96).

## Decision

1. **The k8s conformal method is frozen:** RAPS, lambda = 0.1, k_reg = 1, randomised with a fixed uniform seed (42),
   alpha = 0.05, calibrated on the validation split only, applied to the final chosen k8s model's probabilities.
   No other score or parameter is tried again for k8s.
2. **Everything conformal is labelled EXPLORATORY** (reports, PRs, decision log, README drafts) until the
   confirmation run below has been reported: "exploratory: score chosen after the test split had been seen".
3. **k8s confirmation set:** every k8s issue created strictly after 2026-10-19T18:14:00Z (the last `created_at` of the
   frozen test split), scraped with the existing scraper as it accumulates, restricted to issues with at least one
   `sig/*` label at final state and at least 30 days old at evaluation time (so labels have settled). The frozen
   model (same artefact hash) and the frozen method are run **once** on it, when it holds at least 500 labelled
   issues (expected no earlier than about February 2027 at roughly 5 labelled issues a day; expected Wilson half-width
   at n = 500 and coverage 0.95 is about +-1.9 points). Reported: coverage w.r.t. the primary label and the any-valid
   label with Wilson intervals, mean/median set size with bootstrap CIs, empty-set rate, and the sensitivity of both
   to the randomisation seed (20 seeds). The result is reported whatever it is; a coverage lower bound below 0.95 is
   stated as such.
4. **vscode:** the conformal method is chosen on validation only with the same selection protocol, and is written
   into a new ADR BEFORE the vscode test split is evaluated for the final model. A vscode confirmation set is
   defined the same way (issues created after the vscode test split's last `created_at`, >= 500 labelled, >= 30 days
   old, run once).
5. **Wording rule for top-3:** the k8s top-3 accuracy is reported as 95.4% [94.3, 96.5]; ">= 95%" is never written
   without that qualifier (the point estimate meets the target, the lower bound does not). It is a new 22-class
   `sig/*` task and is never compared with the retired 47-class `area/*` number.

## Consequences

- No conformal claim reaches the README as established before the confirmation run; Phase 6 copy uses "exploratory".
- The confirmation set needs months of accumulation for k8s; vscode needs about a month. Incremental scraping is a
  small recurring task, not a model change.
- If the confirmation run fails (coverage below target), the finding is reported and the method is revisited in a
  new ADR; nothing silently re-tunes on the confirmation set.

## Alternatives

- Re-select on the confirmation set: rejected, it would make it a second tuning set.
- Use LAC (slightly smaller on test, 2.65): rejected as a post-hoc choice; on validation the two are a tie and the
  pre-registered selection rule picked RAPS.
- Treat 0.963 / 2.77 as established: rejected, selection followed a test peek.
