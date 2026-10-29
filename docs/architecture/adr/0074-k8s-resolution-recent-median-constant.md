# ADR-0074 — k8s resolution: serve the 2023-2025 closed-issue median with a fixed interval

Status: Accepted (GG decisions G3, `docs/DECISION_LOG_2026-10.md` D88, and D92.3, 2026-10-29)
Date: 2026-10-29
Decider: Gaurav Gandhi (G3, D92.3)

Supersedes, for kubernetes only, the served point of ADR-0064 (training-window median) and the model Q10/Q90 + CQR v2 interval. vscode is unchanged.
Code: `src/triage_iq/models/resolution.py` (constants pinned in code, no artifact). Numbers: `reports/k8s_constant_35d_calibration.json`,
`reports/eval_summary.json` (`resolution_served.repos["kubernetes/kubernetes"]`).

## Context

ADR-0064 serves the median of the training window as the point estimate. For k8s that window is the 2014-2016 corpus and the median is 3.04 d. The
2023-2025 qp4 table (4,806 closed eligible train issues) has a median of 35.0954 d, an order of magnitude longer: the old constant describes a project
that no longer exists. The interval came from the 2014-2016 model's Q10/Q90 with a CQR v2 adjustment, calibrated on the same old corpus.

## Decision

1. **Point.** k8s serves 35.0954 d, the median of the 4,806 closed eligible train issues of the qp4 table (2023-2025).
2. **Interval.** A fixed interval [0.2755, 372.5905] d from the pre-registered qp4 M0a CQR recipe, set before validation was read: q10/q90 on the
   first 60% (chronological) of train closed issues, CQR Q on the last 40% in log1p space (Q = -0.354), alpha 0.2. Test was never read. The interval
   already carries its CQR adjustment, so the v2 Q is not added again, and the constant is served even if the predictor fails to load.
3. **The interval is accepted on censoring-aware coverage, not on closed-only coverage (D92.3).** On the 885 eligible validation issues:
   - Closed-only coverage, 749 issues: 0.913 [0.892, 0.932], above the 0.80 target. The pre-set ±0.05 band is missed on the conservative side, and the
     recipe was not re-tuned.
   - Closed-only coverage is biased upward. It drops the 136 issues still open at the cutoff, and those are the long-resolving ones; 72 of them have
     already passed the upper bound. Counting the open issues both ways gives censoring-aware bounds of [0.773, 0.845], which contain the 0.80
     target. That is the basis for acceptance. Any published coverage figure for this interval cites the censoring-aware bounds, or names the
     closed-only number as biased.
4. Mean width is 372.3 d. The interval is honest, but too wide to be useful for planning, and the README keeps calling resolution time coarse
   guidance.

## Consequences

- Validation (749 closed): mean AE 69.84 d [64.68, 74.82], median AE 34.63 d [34.30, 34.89]. The 3.04 d point scores 104.23 d mean AE on the ADR-0041
  2014-2016 re-split test (n=2,992). That number comes from a different window and is not a head-to-head comparison.
- Prompts change, because the resolution estimate enters the prompt. The single re-record (D92.6) captures this configuration, and #320 merges only
  after that re-record.
- The constants are code, so a refresh is a code change with its own calibration report, not an artifact upload.

## Alternatives

- Keep 3.04 d: rejected as stale (G3).
- The hazard/survival models of Q6 and the kNN K1 estimator: neither beat the naive median on median AE (D84, D85).
- Re-tune the interval to hit 0.80 on closed-only issues: rejected. It would tune toward a biased target, and tuning after reading validation breaks
  the pre-registration.
