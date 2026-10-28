# QP3b priority P5: results (2026-10-29)

Pre-registration: `docs/QP3B_PRIORITY_P5_PREREG_2026-10.md` (committed before training). Artifact:
`reports/qp3b_priority_p5_2026-10.json` (validation + freeze committed before the test split was
read; test evaluated once). Script: `scripts/qp3b_priority_p5.py`. All numbers below are VERIFIED
from that JSON.

**Verdict: P5 does NOT meet the pre-registered criterion (C1 fails: 3 of 4 met). Do not enable.**
The flag `TRIAGE_PRIORITY_MODEL` and serving are unchanged. P4 stays held.

Frozen config (sha256 `35b8ab78...3179` in the report): stage 1 C=0.25, gate threshold 0.6726
(validation F0.5); stage 2 C=1, class weights (low, medium, high) = (1, 1.3, 1.0); seed 42.
Validation selected a medium up-weight and no high down-weight (weights are chosen on 137 rows).

## Test (n = 1585 issues, 247 prioritised; P4's rows = the 247)

| Criterion | Result | Met |
|---|---|---|
| C1 macro-F1 P5 - P4, paired 95% CI lower > 0 | +0.008 [-0.026, +0.042] (0.494 vs 0.486) | no |
| C2 accuracy - majority, lower > 0 | +0.113 [+0.032, +0.190] (0.522 vs majority 0.409) | yes |
| C3 \|predicted high share - gold\| <= 3 pp on the 247 | 97/247 vs 101/247: 1.6 pp (P4: 113, 4.9 pp) | yes |
| C4 served-high share of all issues <= gold + 5 pp, precision at t > prevalence, AUC CI lower > 0.5 | 2.9% vs gold 6.4% (-3.5 pp); precision 0.550 vs prevalence 0.156; AUC 0.803 [0.772, 0.833] | yes |

Stage 2 test confusion (rows gold low/medium/high): [[18,23,13],[19,50,23],[15,25,61]]; predicted
52 / 98 / 97 vs gold 54 / 92 / 101 (P4: 60 / 74 / 113). Per-class recall low/medium/high 0.33 /
0.54 / 0.60 (P4: 0.41 / 0.43 / 0.62). The high over-prediction is fixed in the distribution, but
accuracy (0.522 vs 0.506) and macro-F1 gains over P4 are within noise.

Stage 1 (all test issues): prevalence 15.6%, AP 0.458, AUC 0.803; the gate passes 129 issues
(8.1%), of which 71 are truly prioritised (precision 0.55, recall 0.29). End-to-end P5 would serve
a level for 8.1% of issues and abstain (keep the LLM priority) for the rest; P4 would serve "high"
for 52.9% of all test issues against 6.4% of issues that are actually high-prioritised.

## Reading

* What P5 fixes (VERIFIED): the unprioritised-issue problem. P4 labels every issue and calls over
  half of them "high"; P5 abstains on 92% and its served-high rate is below the gold rate.
* What it does not fix: level discrimination among prioritised issues is not demonstrably better
  than P4 (C1). Honest conclusion: the stage-1 gate is the useful part; stage 2 is essentially P4
  with a re-balanced output.
* Limits: single repo, 247 test rows (wide CIs); gate recall only 0.29 (most truly prioritised
  issues are never served a level); the stage-1 label is "ever prioritised" so recent issues are
  soft negatives (BELIEVED, not measured). Whether the 8.1% gated issues, served on the LLM-plan
  path, change reviewer-visible priority quality is NOT measured.
* Possible follow-up (needs GG decision; not done): ship only the abstaining gate with P4's levels
  as a new pre-registered candidate P6, or accept P5 on C2-C4 by changing the criterion (not done:
  criteria are not edited after the fact).
