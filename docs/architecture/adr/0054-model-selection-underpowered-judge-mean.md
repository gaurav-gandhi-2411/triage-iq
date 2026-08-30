# ADR-0054: Model selection rests on parse-success, not judge mean -- n=20 cannot resolve the quality gap

Status: Proposed (blocked on approval of the tie-breaker amendment described below)
Date: 2026-08-30

## Context

The 2026-08-29/30 bake-off screen (`docs/eval/bakeoff_prereg_2026-08-29.md`) narrowed to
two surviving arms: A (`openai/gpt-oss-20b`, few-shot) and C (`openai/gpt-oss-120b`,
few-shot). Both clear the pre-registered elimination floor (parse-success >= 90%,
grounding/fabrication clean). The pre-registration's design makes judge mean the primary
decision metric among survivors.

**Judge mean cannot do that job at this sample size.** Observed SD: Arm A 1.48 (n=10,
see below), Arm C 2.46 (n=20). Paired comparison on the 10 issues both arms have: mean
diff -0.2, SD of diffs 1.40. Minimum detectable difference at n=20, 80% power, alpha=0.05:
~0.88 (paired) to ~2.18 (unpaired) -- both far larger than the observed 0.2-0.35 point
gap. Resolving a true 0.2-point paired difference would need n=384, not 20. This is the
same underpowered-gate shape ADR-0046/the 11-issue grounding arm already documented once
in this project's history; not repeating it here.

A biased-comparison error was caught and corrected before this ADR was written: Arm A's
first-reported judge mean (11.80) was computed on only 10 of its 20 issues -- the ones
that happened to clear Groq's rate limit first across several retry rounds, not a random
subsample. It excluded #12254, a confirmed early-termination failure. On the fair,
paired 10-issue comparison, Arm C actually scores marginally higher (12.0 vs 11.8), not
lower. Neither number is resolvable per the power calculation above, but the direction
matters for not overstating Arm A's case.

Separately, a reliability gap IS resolved, cleanly, at the sample sizes already
collected: parse-success rate. Arm C: 44/44 (100%) across all attempts this session.
Arm A: 29/31 (93.5%), with 2 confirmed early-termination failures -- syntactically valid
JSON that stops before satisfying the schema, the same defect that eliminated Arm B (see
ADR-0053). Same prompt (mean prompt tokens: A 5,940, C 5,948 -- effectively identical),
same clamp exposure (A 84% of calls clamped, C 86% -- clamp is a shared condition
between the two arms, not a confound for this specific comparison, though it remains an
open question for production in general -- see Consequences). Arm C's completions are
also shorter (p50 1,099 vs 1,433; max 1,583 vs 2,038 -- Arm C never comes within 465
tokens of the 2,048 ceiling) and higher quality-per-completion-token (10.12 vs 8.46).

## Decision

**Propose an amendment to the pre-registration's decision rule, and select Arm C
(`openai/gpt-oss-120b`, few-shot) on that basis -- pending explicit approval, not applied
unilaterally:**

> Tie-breaker addendum: when the primary metric's (judge mean) minimum detectable
> difference at the pre-registered n exceeds the observed effect (computed and reported,
> not assumed), the decision reverts to whichever surviving arm has a categorically
> resolved advantage on a secondary metric -- specifically parse-success rate, since only
> that metric shows a gap backed by a confirmed, mechanistically-characterized defect
> rather than sampling noise.

This is stated as a proposed amendment, not a reinterpretation of the existing rule
applied after seeing data -- the pre-registration's original text does not specify this
fallback, and claiming it did would be exactly the kind of after-the-fact rule-bending
the pre-registration exists to prevent.

## Consequences

- **What changes:** `TRIAGE_MODEL` in `src/triage_iq/model_config.py` moves from the
  retired `llama-3.1-8b-instant` to `openai/gpt-oss-120b`, once this ADR is approved.
  Inverts a starting assumption of this screen -- the larger, "more expensive" model
  turns out to be cheaper per call (fewer completion tokens), more reliable, and at
  least as good on quality, not a premium option only justified by a clear quality win.
- **What becomes easier:** The 64-issue re-record and any eval-set expansion cost
  slightly less in actual tokens on Arm C than Arm A under the current guard (~22.7
  calls/day vs ~21.7); this gap would widen if a per-model-calibrated token-budget
  predictor (proposed, not implemented -- see the companion session report) ships later,
  since Arm C's genuinely shorter completions aren't yet reflected in a tighter
  reservation.
- **What stays open, deliberately:** whether the token-budget clamp itself contributes
  to early termination in general (not just as a non-confound for this specific A-vs-C
  comparison) is unresolved -- Arm B's unclamped status is perfectly collinear with its
  short prompt (not a controlled comparison), and Arm A's own clamped-vs-unclamped split
  (2/26 vs 0/5) is too small to rule out an effect. Worth investigating before or
  alongside production restoration, since fixing it would improve whichever model ships,
  but it does not block this selection.
- **What this ADR does NOT do:** merge anything, spend quota, or start the 64-issue
  re-record. Those follow only after this ADR is approved.

## Alternatives considered

- **Wait for Arm A's remaining 10 calls, then decide on judge mean as originally
  planned.** Rejected on the power calculation: no amount of finishing Arm A's remaining
  n within the pre-registered n=20 changes the outcome -- the MDD at n=20 (0.88 paired)
  cannot resolve a ~0.2-0.35 point gap regardless of which 20 issues make up the sample.
  Finishing the run answers other useful things (a complete, unbiased judge-mean number
  to report honestly as directional) but will not decide the model.
- **Silently reinterpret "primary decision metric" to mean parse-success without saying
  so.** Rejected -- exactly the failure mode this engagement has been correcting for
  several turns running (the elimination rule, the recovery-window claim, the biased
  subsample). The amendment is proposed explicitly so it can be approved, rejected, or
  modified before being acted on.
- **Expand n toward the ~384 needed to resolve judge mean.** Rejected as disproportionate
  -- that's roughly 19x the current sample, would take weeks at observed quota pacing,
  and the resolved parse-success gap already gives a defensible, mechanism-backed
  decision without it.
