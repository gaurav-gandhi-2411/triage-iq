# ADR-0054: Model selection rests on parse-success -- a pre-registered elimination gate promoted to decision authority, not a new metric

Status: Accepted 2026-08-30, corrected and RE-ACCEPTED on a revised basis 2026-09-03
(GG approved). **Selection unchanged (`openai/gpt-oss-120b`); its justification
changed** -- from parse-success (retracted, see below) to truncation headroom (a
direct, low-variance measurement -- see "Revised decision" below). See ADR-0055 for
the separate, larger finding this correction led to: the early-termination defect
itself was substantially a schema/output-contract defect, not a model quality
difference, and is now fixed for the model this ADR selects (ADR-0055 Part A/C).
Date: 2026-08-30

## Correction (2026-09-03)

**The "44/44 (100%) vs 29/31 (93.5%)" parse-success figures below are retracted as
stated.** They were never traceable to a committed artifact -- an exhaustive search of
this repo's full git history (every branch) and the authoring session's own scratchpad
found no results file, log, or computation supporting 44 or 31 as the attempt count for
either arm. Recovered instead: the raw per-call JSONL that actually produced this
screen (`part_d_screen_results.jsonl`, a prior session's scratchpad, cross-checked
issue-by-issue against §6's pre-registered sample table -- full match, 20/20 issues
present for each arm).

**Verified parse-success on the pre-registered 20-issue sample, from those raw
records:**
- Arm C (`openai/gpt-oss-120b`, few-shot): **20/20 (100%)**, zero failures.
- Arm A (`openai/gpt-oss-20b`, few-shot): **19/20 (95%)**, one failure (#12254,
  confirmed early-termination -- matches this ADR's own account of excluding #12254
  from the biased 11.80 judge-mean figure below, which is unaffected by this
  correction).

The direction of the original claim survives (Arm C still has a lower failure rate on
the pre-registered sample) but the **denominator was wrong by more than 2x for both
arms**, and the true counts are small enough that the conclusion changes. **Pooled
against the subsequent 64-issue re-record** (also raw, checkpoint-verified: 59/64
resolved, 5/64 permanently early-terminated under `openai/gpt-oss-120b` --
`k8s-12224`, `vscode-4996`, `k8s-12477`, `k8s-12248`, `k8s-13508`; `vscode-4996` is the
one re-record failure that was also in the screen sample, where it had succeeded --
non-determinism across separate live calls, not a contradiction):

| | failures / n | rate | 95% Wald CI |
|---|---:|---:|---|
| Arm C, screen only | 0/20 | 0% | [0%, 0%]\* |
| Arm C, record only | 5/64 | 7.81% | [1.24%, 14.39%] |
| **Arm C, pooled** | **5/84** | **5.95%** | **[0.89%, 11.01%]** |
| Arm A, screen only (no record exists) | 1/20 | 5.00% | [0%, 14.55%] |

\*Wald CI is degenerate at 0 successes; a Wilson/Jeffreys interval would show a
non-zero upper bound, not materially changing the overlap conclusion below.

**The pooled Arm C rate and Arm A's screen-only rate overlap almost completely.**
Parse-success does **not** cleanly resolve between the two models on the corrected,
verified numbers -- the same underpowered-comparison shape this ADR already used to
disqualify judge mean. **This ADR's Decision section's parse-success-as-tie-breaker
reasoning no longer holds as stated.** See the companion session's Priority 4 findings
for a revised basis (completion-token distribution and ceiling headroom, both
verified directly from raw `usage` data and *not* subject to the same small-n problem
as parse-success or judge mean) and Priority 3 findings for a separate, load-bearing
observation: **all 9 early-termination failures examined across both arms (5 from the
gpt-oss-120b re-record, 4 from this screen) fail on the identical set of fields** --
`grounding`, `grounding_status`, `declared_attribution`, `abstention_status` (100%,
9/9) and usually also `resolution_bucket`, `resolution_confidence_pct`,
`resolution_interval_conformal` (89%, 8/9) -- every one of which is either overwritten
post-hoc by `app.py` regardless of what the model emits, or already null-tolerant by
design (`TriagePlan`'s own field descriptions, `src/triage_iq/models/triage.py:187-244`).
This looks like a property of the output contract under Groq's `strict:true` decoding,
not a difference between the two models -- see the full writeup for the field-by-field
evidence.

### Revised decision (2026-09-03, GG APPROVED -- this is now the operative basis)

**Selection: KEEP `openai/gpt-oss-120b`, on truncation headroom.** Parse-success
(5.95% pooled vs 5% -- does not resolve) and quality-per-1K-tokens (numerator carries
the same small-n noise already rejected as a basis elsewhere in this engagement) are
explicitly DROPPED from the stated basis, not merely de-emphasized -- selection stands,
justification changes to what's below.

Re-grounded on the metrics that are direct measurements, not derived rates at n=20/31 --
computed independently from the same raw `part_d_screen_results.jsonl`, cross-checked
against this ADR's original figures (which are close but not exact matches; treating
the independently recomputed numbers below as authoritative since they're reproducible
from the raw file):

| metric | gpt-oss-120b | gpt-oss-20b | resolved? |
|---|---:|---:|---|
| Completion tokens, p50 (n=20/19) | 1,105 | 1,376 | **Yes** -- paired, same prompts, low-variance count data, no judge involved. |
| Completion tokens, max | 1,459 | 2,038 | **Yes** -- same reasoning. |
| Headroom vs. 2,048 cap (worst case) | 589 tokens | 10 tokens | **Yes** -- 20b's worst observed call in this sample was 10 tokens from the truncation cliff. |
| Quality per 1K completion tokens | 10.12 | 8.46 | **Partially** -- the token-count denominator is solid, but the judge-score numerator carries the same n=20/n=10 noise already shown insufficient to resolve raw judge mean. The *direction* is consistent with the token-count gap driving most of the ratio, not judge noise alone, but this is not an independently powered result -- treat as corroborating, not decisive on its own. |
| Cost per call (USD) | $0.15/M in, $0.60/M out (verified 2026-09-03, Groq's published pricing) | not verified for this model -- gpt-oss-20b's actual rate has not been checked | **Partially** -- gpt-oss-120b's own rate is now correct in `model_config.py` (`TRIAGE_PRICE_PROMPT_PER_MTOK`/`TRIAGE_PRICE_COMPLETION_PER_MTOK`, corrected from a stale blended 0.27 that wasn't even wired to the cost formula it was meant to feed -- see ADR-0055). gpt-oss-20b's rate is still unchecked, so a head-to-head dollar comparison remains unverified; not load-bearing for this decision since parse-success/cost were already dropped from the basis. |

**Approved basis for keeping `openai/gpt-oss-120b` (2026-09-03):** not parse-success
(does not resolve) and not judge-mean/quality-per-token (does not resolve, carries the
same small-n noise already rejected elsewhere in this engagement) --
**operational safety margin.** gpt-oss-20b's worst-case completion in this sample sat
10 tokens from the hard truncation ceiling; gpt-oss-120b's sat 589 tokens away. That
gap is a direct, low-noise measurement, and it matters mechanically: a completion that
hits the ceiling raises `TruncatedCompletionError` (PR #113) in production, not just in
this eval. Selection stands; this is the full and only basis going forward.

**Separately, and independent of which model ships -- now IMPLEMENTED, see ADR-0055:**
Priority 3's finding that early termination concentrates on fields the schema forces
as `required` under Groq's `strict:true` mode but that are either overwritten post-hoc
or already null-tolerant by design led to a schema-reduction fix
(`_strip_post_hoc_fields`, `src/triage_iq/models/triage.py`) that removed all 7 such
fields from the wire schema (18 -> 11 required). Live-validated 2026-09-03: all 5 of
the real `gpt-oss-120b` early-termination failures from the 64-issue re-record
succeeded on a genuine, cache-bypassed re-test under the reduced schema. A separate,
new failure mode was found on one issue (a mangled field name, not a missing field) --
the original defect class does not fully generalize to "zero failures ever," but the
specific defect this ADR's parse-success comparison was built on is resolved for the
model this ADR selects. Full detail, including why this doesn't reopen model selection
or ADR-0053's few-shot decision, in ADR-0055.

**A gap this correction surfaced in the checkpoint-keying fix (`d2b2b96`) -- now FIXED:**
`_compute_prompt_hash()` hashed only the system prompt and few-shot messages, not
`TriagePlan`'s JSON schema (the `response_format` sent to Groq). Fixed 2026-09-03 (Part
B, ADR-0055) to include the serialized wire schema in the hash -- confirmed this
correctly invalidates the 59 entries recorded under the pre-schema-reduction contract
(hash changed `5ddbe97c...` -> `0a15adc0...`; `load_checkpoint()` now treats all 64 old
entries as belonging to a different, non-comparable configuration).

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
