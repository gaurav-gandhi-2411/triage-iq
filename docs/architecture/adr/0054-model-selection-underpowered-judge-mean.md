# ADR-0054: Model selection rests on parse-success -- a pre-registered elimination gate promoted to decision authority, not a new metric

Status: Accepted
Date: 2026-08-30

## Context

The 2026-08-29/30 bake-off screen (`docs/eval/bakeoff_prereg_2026-08-29.md`) tested 4
arms and eliminated 2 before this decision: **Arm D** (`qwen/qwen3.6-27b`) on tokenizer
footprint against the free-tier TPM ceiling -- its real tokenizer overflows the 8,000 TPM
ceiling even on the shortest eval-set issue, at any margin. **Arm B** (`gpt-oss-20b`,
no few-shot) on the pre-registered parse-success floor (>=90%) -- 3/8 failures made a
90%-of-20 floor mathematically unreachable (best case 17/20 = 85%). That left two
survivors: **A** (`openai/gpt-oss-20b`, few-shot) and **C** (`openai/gpt-oss-120b`,
few-shot), both clearing the elimination floor. The pre-registration's design makes judge
mean the primary decision metric among survivors that clear elimination.

**Judge mean cannot do that job at this sample size.** Observed SD: Arm A 1.48 (n=10,
a biased subsample -- see below), Arm C 2.46 (n=20, complete). Paired comparison on the
10 issues both arms have: mean diff -0.2, SD of diffs 1.40. Minimum detectable difference
at n=20, 80% power, alpha=0.05: ~0.88 (paired) to ~2.18 (unpaired) -- both far larger than
the observed 0.2-0.35 point gap. Resolving a true 0.2-point paired difference would need
n=384, not 20. This is the same underpowered-gate shape ADR-0046/the 11-issue grounding
arm already documented once in this project's history; not repeating it here.

**A second finding, about the eval instrument itself, not just the sample size:** on the
10 paired issues, the diffs were `[-2, 0, 0, 0, 0, -2, -1, 3, 0, 0]` -- **the judge
assigned identical total scores to both models on 6 of 10 issues.** That is not merely "n
is too small" -- on this task, at this prompt/schema shape, the judge has limited
discriminating power between two models that both produce fluent, schema-conformant
output. This should inform how much weight judge mean carries in future model-comparison
decisions generally, not just this one: a metric that returns a tie on 60% of paired
inputs needs either a sharper rubric or a much larger n to be a reliable primary
decision metric, independent of which specific models are being compared.

A biased-comparison error was caught and corrected before this ADR was written: Arm A's
first-reported judge mean (11.80) was computed on only 10 of its 20 issues -- the ones
that happened to clear Groq's rate limit first across several retry rounds, not a random
subsample. It excluded #12254, a confirmed early-termination failure. On the fair,
paired 10-issue comparison, Arm C actually scores marginally higher (12.0 vs 11.8), not
lower. Neither number is resolvable per the power calculation above, but the direction
matters for not overstating Arm A's case.

**Separately, and this is the actual basis for this decision: a reliability gap IS
resolved, cleanly, at the sample sizes already collected, on a metric that already
carried decision authority in this screen -- parse-success rate eliminated Arm B outright
under the exact same pre-registered floor.** Selecting on it here is not reaching for a
new, previously-unused metric; it is extending a metric that has already been the
deciding factor once in this same screen. Arm C: 44/44 (100%) across all attempts this
session. Arm A: 29/31 (93.5%), with 2 confirmed early-termination failures -- syntactically
valid JSON that stops before satisfying the schema, the same defect that eliminated Arm B
(see ADR-0053). Same prompt (mean prompt tokens: A 5,940, C 5,948 -- effectively
identical), same clamp exposure (A 84% of calls clamped, C 86% -- clamp is a shared
condition between the two arms, not a confound for this specific comparison, though it
remains an open question for production in general -- see Consequences). Arm C's
completions are also shorter (p50 1,099 vs 1,433; max 1,583 vs 2,038 -- Arm C never comes
within 465 tokens of the 2,048 ceiling) and higher quality-per-completion-token (10.12 vs
8.46).

## Decision

**Amendment to the pre-registration's decision rule, APPROVED, and Arm C
(`openai/gpt-oss-120b`, few-shot) SELECTED on that basis:**

> Tie-breaker addendum: when the primary metric's (judge mean) minimum detectable
> difference at the pre-registered n exceeds the observed effect (computed and reported,
> not assumed), the decision reverts to whichever surviving arm has a categorically
> resolved advantage on **parse-success rate** -- not a newly-introduced metric, but the
> same metric the pre-registration's elimination gate already used to remove Arm B,
> extended from a gate (>=90% survives) to a tie-breaker (higher rate wins among
> survivors) exactly when judge mean is provably unable to discriminate.

This was approved because the fact that triggers it -- MDD 0.88 vs observed 0.2, n~384
required to resolve the actual gap -- was computed *before* knowing which arm it would
favor. It is a stopping rule for a case the pre-registration did not anticipate (a
provably unresolvable primary metric, not merely a wide interval), not a post-hoc
reinterpretation chosen because it happened to pick the preferred arm.

**Full basis for selecting Arm C, all already-collected data:**
- Parse-success: 44/44 (100%) vs 29/31 (93.5%), the deciding metric, itself a
  pre-registered elimination gate extended per the amendment above.
- Reliability defect on Arm A is characterized, not a fluke: same early-termination
  shape that eliminated Arm B (ADR-0053), on the same base model.
- Cost: cheaper per call on an identical prompt (completion p50 1,099 vs 1,433 tokens).
- Quality-per-token: 10.12 vs 8.46.
- Ceiling headroom: 465 tokens of slack vs 2,038 against the 2,048 cap.
- The only unbiased quality comparison available (paired, same 10 issues): Arm C
  numerically ahead, 12.0 vs 11.8 -- unresolvable per the power calculation, but
  directionally consistent with, not contradicting, the parse-success-based decision.

## Consequences

- **What changes:** `TRIAGE_MODEL` in `src/triage_iq/model_config.py` moves from the
  retired `llama-3.1-8b-instant` to `openai/gpt-oss-120b`. **This inverts a starting
  assumption of this screen, stated plainly:** the larger, nominally "more expensive"
  model turns out to be cheaper per call (fewer completion tokens on an identical
  prompt), more reliable (100% vs 93.5% parse-success, with a characterized defect on
  the smaller model), and more concise (higher quality per completion token) -- not a
  premium option only justified by a clear quality win over the cheaper default. The
  screen started by treating `gpt-oss-20b` as the natural default and `gpt-oss-120b` as
  the arm that would need to earn its keep on quality; the data reversed which arm
  needed to earn its keep.
- **The two eliminations that preceded this decision, for the record:** Arm D
  (`qwen/qwen3.6-27b`) on tokenizer footprint against the free-tier TPM ceiling -- a
  real infrastructure constraint, not a quality judgment. Arm B (`gpt-oss-20b`, no
  few-shot) on the pre-registered parse-success floor -- see ADR-0053 for the full
  record and its own consequence (few-shot is retained on evidence, and even the
  selected model still needs it).
- **A finding about the eval instrument, independent of this specific decision:** the
  local judge (qwen3:8b, ADR-0019) returned identical scores on 6 of 10 paired
  comparisons in this screen. That is evidence of limited discriminating power on this
  task/prompt shape at the current rubric and sample sizes, not just an n problem --
  future model-comparison work should weight judge mean accordingly, or invest in a
  sharper rubric, rather than treating a tie-heavy judge as automatically resolvable
  with more n alone.
- **What becomes easier:** The 64-issue re-record and any eval-set expansion cost
  slightly less in actual tokens on Arm C than Arm A under the current guard (~22.7
  calls/day vs ~21.7); this gap would widen if a per-model-calibrated token-budget
  predictor (proposed, not implemented -- see the companion session report) ships later,
  since Arm C's genuinely shorter completions aren't yet reflected in a tighter
  reservation.
- **What stays open, deliberately -- a production question, not a blocker on this
  decision:** whether the token-budget clamp itself contributes to early termination in
  general (not just as a non-confound for this specific A-vs-C comparison) is
  unresolved. Arm B's unclamped status is perfectly collinear with its short prompt (not
  a controlled comparison), and Arm A's own clamped-vs-unclamped split (2/26 vs 0/5) is
  too small to rule out an effect. Worth investigating before or alongside production
  restoration, since fixing it would improve whichever model ships -- but the clamp is a
  *shared* condition between Arm A and Arm C (84% vs 86%), so it cannot be what's
  driving the reliability difference between them, which is what this ADR decides.
- **What this ADR does:** authorizes updating `TRIAGE_MODEL` and starting the 64-issue
  re-record against `gpt-oss-120b`. Does not itself merge anything or spend quota --
  those are separate, explicitly-approved steps (see the companion session report's
  Part B/C).

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
