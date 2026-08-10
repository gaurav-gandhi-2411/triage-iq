# ADR-0044: Fabrication-rate hard gate — keep zero-tolerance, no per-repo slack

Status: Accepted
Date: 2026-08-10

## Context

Issue #48 asked to redo PR #34's hard-gate promotion (which pinned two specific known-bad
cases — k8s#13057, vscode#311836 — an approach ADR-0039 found fragile: neither case
reproduced after the classifier cutover, even though the grounding verifier itself was still
working correctly) against the rate-ratchet mechanism PR #47 replaced it with instead.

That promotion already happened, in a burst of closely-related work that landed on `main`
before this issue was picked up: PR #57 removed `continue-on-error: true` from both
`eval-gate.yml` jobs (`structural-invariants`, `quality-regression`), confirmed genuinely green
(not masked) on three consecutive live runs first. Both fabrication-detecting mechanisms are
now load-bearing, blocking CI checks:

1. **`test_grounding_ratchet_no_new_ungrounded_claims`** (`eval/test_invariants.py`,
   `structural-invariants` job) — `ungrounded_count <= baseline["ungrounded_count"]`, checked
   per-repo against `_GROUNDING_BASELINE` (currently 0/53 k8s, 0/11 vscode).
2. **`test_k8s_no_fabrication` / `test_vscode_no_fabrication`**
   (`eval/test_quality_regression.py`, `quality-regression` job) — `fabrication_rate == 0.0`,
   an absolute check with no baseline reference at all.

Both are, in effect, zero-tolerance today. What issue #48 asked for and PR #57 didn't settle:
**is zero-tolerance the honest bound**, given the explicit false-fire risk it named —
"a zero-tolerance gate on n=11 vscode will false-fire on one bad plan"? At n=11, a single new
hallucination is a 9.1% fabrication rate; a gate with no slack blocks on that first occurrence,
same as it would on a real, larger regression. Neither test's code had settled this
deliberately — the ratchet's bound is just whatever the last cassette happened to measure, and
the absolute check was written in ADR-0028 Phase B3 as an *informational* assertion, never
revisited for whether zero tolerance is still the right choice now that it blocks merges.

A second, smaller problem surfaced while reading `test_quality_regression.py` for this
decision: `_check_no_fabrication` and `_check_no_prose_number_contradiction`'s docstrings still
say "INFORMATIONAL ONLY... this file's CI job is continue-on-error:true" — true when written
(2026-07-11 / ADR-0042), false since PR #57. Stale documentation describing a check as
non-blocking when it now is is exactly the kind of silent-rot ADR-0043's postscript warned
about, just in a docstring instead of a workflow file.

## Decision

**Keep true zero-tolerance on both mechanisms. Do not add per-repo slack, including for
vscode's n=11.**

The false-fire risk issue #48 named is real but structurally bounded, not open-ended:

- Grounding is computed by **cassette replay** (`CassettePlayer(strict=True)`), zero live LLM
  calls, confirmed deterministic given a fixed plan (ADR-0019: 39/39 issues byte-identical
  grounding_status across two independent recordings of the same plans). An ordinary PR — code,
  prompt-unrelated changes, dependency bumps — **cannot** move `fabrication_rate` or
  `ungrounded_count` at all, because it doesn't touch the frozen cassette. The gate is inert
  noise-wise for the overwhelming majority of PRs that will ever run it.
- The only event that can change these numbers is a **cassette re-record**, and
  `record-cassette.yml` is `workflow_dispatch`-only, opens a PR rather than pushing directly
  (branch protection requires the `test` check, no bypass even for the workflow's own token),
  and that PR gets the same human review as any other. A re-record is already a deliberate,
  supervised event — not something that happens silently on an unrelated PR — so the moment a
  false-fire *could* happen is exactly the moment a human is already looking at the diff and in
  a position to tell a real regression from a new-cassette artifact.
- When that happens, the response is the same deliberate-investigate-then-re-derive workflow
  already used four times on this project (the manifest-drift fix, the `_load_classifier`
  import fix, the stale `_GROUNDING_BASELINE` hash fix, and the named-case-pin removal in
  ADR-0039 — each investigated first, then explicitly re-baselined with a written reason, never
  silently loosened). One extra investigation at re-record time is the accepted cost.
- Fabrication is explicitly framed in `_check_no_fabrication`'s own docstring as "a hard-fail
  correctness issue, not a soft quality miss" — the LLM inventing a similar-issue number or
  component that doesn't correspond to what it was actually given. A single such case in an
  11-issue sample is a real 9.1% hallucination rate on this project's most data-constrained
  repo (ADR-0017's data-ceiling finding), not noise to be smoothed over. Pre-baking slack (e.g.
  "allow 1/11") would mean a genuine new hallucination could land and stay invisible until a
  *second* one occurred — a worse failure mode than an occasional, cheap, human-supervised
  false-fire investigation. This matches the project's standing bias (ADR-0043: refused to
  write an inflated quality baseline; ADR-0039: refused to loosen a gate to make a proxy metric
  pass) — the tie-breaker order (rule 74: correctness before operability) says the same thing
  here.

This is a deliberate contrast with the *quality-regression* mean-score gate
(`_check_repo_quality`), which correctly *does* use a per-repo statistical band (vscode 0.45,
k8s 0.22, derived from measured judge jitter, ADR-0019) — that gate protects a continuous,
inherently noisy judge score where jitter is real and measured. Fabrication is a discrete,
deterministic-under-replay correctness signal with a different noise profile; the same slack
that's honest for one is not honest for the other.

**No collision with `test_k8s_quality_regression`.** That test's `xfail(strict=True)` marker
was already removed in the ADR-0043 baseline write (2026-08-10, same day) — it now compares
against the new post-cutover baseline (10.2642) rather than the frozen pre-cutover one, so it
passes normally. The two signals stay independently gated exactly as issue #48 asked:
fabrication/grounding is a ground-truth correctness check with no tolerance band; quality
regression is a judge-scored prose-quality check with a jitter-derived band. Neither reads or
overrides the other's threshold, and neither test file imports from the other.

**Docstrings fixed** in `eval/test_quality_regression.py` — `_check_no_fabrication` and
`_check_no_prose_number_contradiction` no longer say "INFORMATIONAL ONLY" /
"continue-on-error:true"; both now state plainly that the job is blocking (since PR #57) and
point here for the bound rationale.

## Consequences

- **What changes:** Nothing in test logic — both mechanisms were already effectively
  zero-tolerance and already blocking (PR #57). This ADR makes the bound choice explicit and
  fixes the two stale docstrings that described outdated (informational-only) CI behavior.
- **What becomes harder:** A cassette re-record that introduces even one new hallucination in
  vscode's 11-issue sample will block that PR until investigated — cannot be waved through by
  re-running CI.
- **What becomes easier:** Anyone reading either test file now sees an accurate description of
  what actually gates a merge, and a clear pointer to the reasoning if the bound is ever
  questioned again, instead of rediscovering it from scratch.
- **Reversible, cheap to revisit:** if a re-record ever produces a false-fire that's genuinely
  noise (not a real regression), the fix is the same as every prior instance — investigate,
  then deliberately re-derive `_GROUNDING_BASELINE`/`fabrication_rate`'s expected value with a
  written reason in the commit, same as the four prior precedents this ADR cites.

## Alternatives considered

- **Add fixed slack for vscode (e.g. allow ≤1/11)** — rejected: silently tolerates a real ~9%
  hallucination rate until a second occurrence, on the repo already flagged (ADR-0017) as the
  most data-constrained and statistically fragile. The false-fire cost it avoids is bounded to
  deliberate, human-supervised re-record events, not ordinary PRs — not worth trading away a
  real correctness signal to avoid an already-rare, already-supervised event.
- **Derive a statistical band the way `_check_repo_quality` does (2×SEM style)** — rejected:
  that band models genuine judge-scoring jitter on a continuous score. Fabrication is a
  discrete, replay-deterministic ground-truth signal (no live LLM call, confirmed byte-identical
  across replays) — there is no measured jitter to derive a band from; inventing one to feel
  consistent with the other gate would launder a fudge as a formula.
- **Leave the ratchet's bound as "whatever the last recording measured," undocumented** —
  rejected: this is what issue #48 explicitly asked not to leave undecided, and it's exactly the
  kind of undocumented judgment call this project's ADR discipline exists to close.
