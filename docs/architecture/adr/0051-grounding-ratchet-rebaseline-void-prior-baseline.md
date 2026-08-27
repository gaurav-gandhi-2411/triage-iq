# ADR-0051 — Grounding ratchet re-baseline: prior baseline void, new baseline is a degradation

**Status:** WITHDRAWN (2026-08-27, same day, superseded by continued investigation — see below)
**Date:** 2026-08-27
**Decider:** Gaurav Gandhi

## Withdrawal note (2026-08-27)

This ADR is withdrawn, not corrected. Two independent findings overturned its premise
before it was ever merged:

1. **The model swap it re-baselines against (`openai/gpt-oss-20b`, PR #101/#106) is
   halted.** A same-day fallback-plan audit found 68.75% first-attempt JSON-parse failure
   and a 32% fully-degraded-fallback rate on `kubernetes/kubernetes` — a severe
   structured-output reliability regression, unrelated to grounding, that is reason enough
   on its own to not ship the swap. Re-baselining a gate against a model that isn't being
   deployed is moot.
2. **The underlying grounding measurement is itself under review.** `scripts/measure_grounding.py`
   (2026-07-24) predates ADR-0020's "honest override" declared-attribution affordance
   (2026-08-23) and never applies it — a real eval/production skew. This ADR's "3/11
   ungrounded, re-baseline to 3" framing assumed the coarse (no-override) check was the
   thing needing correction. Further review of `verify_declared_attribution()` found the
   override path itself is **unvalidated self-certification** (any non-empty free-text
   reason suffices, checked against nothing) — so neither number (3/11 coarse, or a
   hypothetical 0/11 "corrected" via the override exception) is currently trustworthy
   as a decision input. Applying the override exception uncritically, as this session
   first proposed, would have handed the halted model a pass specifically on the 3
   disputed cases.

`_GROUNDING_BASELINE` is restored to the pre-swap `llama-3.1-8b-instant` values
(`microsoft/vscode: 0/11`, `kubernetes/kubernetes: 0/53`) in `eval/test_invariants.py`.
The ratchet mechanism is unchanged. This ADR is kept (not deleted) as the record of why a
re-baseline was attempted and specifically why it was reversed — see the eval-methodology
soundness review this same session for the open question on the override check.

---

**Original content below, preserved for the record. Do not treat any of it — including
the "3/11", "27.3%", or "void baseline" framing — as current state.**

## Context

`test_grounding_ratchet_no_new_ungrounded_claims` (`eval/test_invariants.py`) gates any new
ungrounded claim against a fixed baseline (`_GROUNDING_BASELINE`), last set 2026-08-10 at
`kubernetes/kubernetes: 0/53`, `microsoft/vscode: 0/11`, measured against
`llama-3.1-8b-instant`.

Groq has since deprecated `llama-3.1-8b-instant` — it now returns HTTP 404 on every call.
PR #101/#106 swaps the triage/judge model to `openai/gpt-oss-20b`. Measured against the
same eval set on the new model (cassette-replayed, zero live LLM calls,
`scripts/measure_grounding.py`):

- `kubernetes/kubernetes`: 0/53 ungrounded — unchanged.
- `microsoft/vscode`: **3/11 ungrounded issues** (27.3%), up from 0/11. At the claim level,
  3/44 claims across those 11 issues are ungrounded. All 3 failures are on the
  `predicted_component` claim specifically — zero similar-issue citation hallucination in
  any of the 11 issues — so the 3 failures don't cluster on one issue; the issue-level rate
  (3/11) is the correct unit of analysis, not an inflated one.

A Fisher's exact test on the 2×2 table (3/11 vs 0/11) gives **p = 0.107**, one-sided —
**not significant at α = 0.05**. At n=11, this test cannot distinguish a real regression
from a single unlucky draw; the true underlying rate could be anywhere up to roughly 27%
(rule-of-three bound) and still be consistent with having observed 0/11 by chance under the
old baseline.

## The decision this ADR is actually about

The instinctive framing — "the model swap regressed grounding, evaluate whether to accept
the regression" — is wrong. **The prior baseline (0/11) was measured against a model that no
longer exists.** `llama-3.1-8b-instant` returns 404 on every call; there is no live
"baseline model" to hold to, and no path to producing another 0/11-consistent measurement
against it even for comparison purposes. A ratchet whose reference point is an unreachable
model is not gating anything — it is asserting a number that can never be re-verified.
"Hold the model swap and keep the old baseline" is therefore not an available option. This
is a **void baseline**, not an accepted regression: there is nothing left to regress
against, only a new floor to establish.

## Decision

1. Re-baseline `_GROUNDING_BASELINE["per_repo"]` in `eval/test_invariants.py` to the
   `gpt-oss-20b` measurement: `microsoft/vscode: {ungrounded_count: 3, n: 11}`,
   `kubernetes/kubernetes: {ungrounded_count: 0, n: 53}` (unchanged). The ratchet mechanism
   itself is untouched — it still fails on any *new* ungrounded issue beyond this reference
   point. Only the reference point moves, and only because the old one is unreachable, not
   because the new number is acceptable on its own terms.
2. Merge PR #110's `test_grounding_rate_non_inferior` alongside this change, rebased onto
   the same baseline. It does not replace the ratchet — both run every CI cycle. It adds a
   statistically valid second gate (one-sided Fisher's exact, 10pp non-inferiority margin,
   α=0.05) that will only become *load-bearing* once `eval_set.jsonl` reaches the required
   n≥75 issues/repo (currently vscode=11, k8s=53). Below that, it skips — and now emits an
   explicit `warnings.warn()` (not a silent skip) naming the current n and the required n on
   every run, so the gap stays visible instead of disappearing into a skip count nobody
   reads.

## Stated plainly, not softened

- **The prior baseline is void, not held.** There is no reachable model to re-measure it
  against. This is not "we chose to accept a regression against the old baseline" — the old
  baseline cannot be evaluated against anymore.
- **The new baseline is worse than the old one in absolute terms.** 0% → 27.3% ungrounded-issue
  rate for `microsoft/vscode`. This is a real increase in the raw number, full stop.
- **The increase is not statistically significant at the current sample size** (n=11,
  Fisher's exact p=0.107 one-sided). This means the data cannot currently distinguish "real
  regression" from "the same underlying rate as before, observed unluckily" — it does
  **not** mean the increase should be read as safe or immaterial. Absence of significance at
  n=11 is a statement about statistical power, not about the underlying risk.
- **Grounding protection is degraded until the eval set is expanded.** The ratchet still
  catches any *further* regression beyond 3/11, but the population it's checking is 7x
  smaller than the ~75/repo needed to detect a 10pp regression at 80% power. A real
  regression smaller than roughly 3-4 additional ungrounded issues in one deploy could land
  and neither test would catch it (see "what would actually be caught today," below). This
  state is accepted as time-boxed, not as adequate — see the eval-set-expansion scoping
  work tracked separately (2026-08-27 diagnostic session, Part D).

## What regression magnitude would actually be caught today

With the ratchet re-baselined to 3/11 for vscode and the non-inferiority test skipping
(underpowered, n=11 < 75):

- The **ratchet** catches anything **> 3/11** — i.e., a 4th (or more) ungrounded issue in
  any future vscode run against this same 11-issue eval set would fail CI immediately.
  Below that (0, 1, 2, or 3 ungrounded issues), it passes silently.
- The **non-inferiority test contributes nothing** right now — it cannot assert anything at
  n=11 and skips (loudly, per this ADR's change, but still skips).
- **Net effect: today's actual protection is "catch a 4th ungrounded issue out of 11,"
  which is a ~9pp jump on top of the new 27.3% baseline** (i.e., a regression to 4/11 =
  36.4% is the smallest jump that fails anything). A regression that keeps the count at or
  below 3/11 — including one that is a real, mechanism-backed regression, not noise — passes
  both gates. This is a materially weaker safety net than the pre-swap state (which caught
  any single ungrounded issue out of 11, a ~9pp jump off a 0% floor) for the simple reason
  that the floor itself moved up. This gap is the direct, intended consequence of accepting
  a void baseline rather than blocking indefinitely on a model that no longer exists, and is
  why eval-set expansion (Part D scoping, tracked separately) is the actual fix, not this
  re-baseline.

## Consequences

- CI on the `openai/gpt-oss-20b` model swap goes green without weakening or deleting either
  gate's mechanism.
- Grounding regression detection is measurably degraded (see above) until the eval set
  expansion lands. This is tracked, time-boxed, and not to be treated as a closed issue by
  virtue of CI being green.
- Any future re-baseline of `_GROUNDING_BASELINE` must go through the same ceremony: an ADR,
  not a silent number change, per the standing convention this ADR follows (see
  ADR-0039's blind-spot note in `eval/test_invariants.py` for the precedent on why silent
  baseline changes are the recurring failure mode here).

## Alternatives considered

- **Hold the model swap, keep the old baseline.** Not available — the old baseline's model
  is unreachable (404 on every call); there is nothing to hold to. Rejected as not a real
  option, not merely a worse one.
- **Delete or loosen the ratchet mechanism** (e.g., convert it to informational-only).
  Rejected per explicit instruction: a gate that can't fail loudly rots silently — this
  exact failure mode already happened once to this same test (see the 2026-08-10 comment in
  `eval/test_invariants.py`: the ratchet baseline drifted from the real eval set for weeks,
  masked by `continue-on-error`, until promoted to blocking).
- **Block the model swap entirely until the eval set is expanded.** Rejected: the model
  being swapped away from is already fully dead (404), so blocking the swap does not
  preserve any existing protection — it only leaves production on a completely non-functional
  model while the expansion work (estimated in the Part D scoping) is done. The swap and the
  re-baseline are accepted together as the only combination that both restores a working
  model and keeps an honest (if temporarily weaker) safety net in place.
