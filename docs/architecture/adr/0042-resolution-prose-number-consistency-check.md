# ADR-0042 — Synthesis prose/number consistency: deterministic check added, not currently gating

**Status:** Accepted
**Date:** 2026-08-06
**Decider:** Gaurav Gandhi

## Context

ADR-0037's investigation into the k8s synthesis-quality regression surfaced, as a side finding,
a specific case (k8s-14756) where the LLM's free-text `expected_resolution_summary` directly
contradicted the numeric resolution interval it was generated alongside: the prose said
"typically 1 day or less for a straightforward configuration tweak" while
`expected_resolution_lower_days`/`expected_resolution_upper_days` were `[2.8, 21.6]` — the
claimed range doesn't even touch the actual interval's lower bound. GG's framing: this is a real
correctness defect (the model contradicting numbers it was directly handed in its own prompt),
distinct from ADR-0037's hedging-tone/confidence-framing investigation into the same synthesis
call, and worth measuring on its own rather than folding into that thread.

## Decision

**Measure first, deterministically, offline.** Built
`src/triage_iq/models/resolution_consistency.py::verify_resolution_consistency()` — same
dataclass-report pattern as `grounding.py::verify_plan_grounding()`. It extracts every "N
day(s)/hour(s)/week(s)/month(s)" time expression from the prose (handling explicit ranges,
"less than"/"within"/"or less" upper-bound-only claims, "more than"/"at least"/"or more"
lower-bound-only claims, and bare point estimates with a ±25% tolerance window), unions them into
an implied `(lo, hi)` day range, and flags a contradiction only when that implied range has
**zero overlap** with the actual `[lower_days, upper_days]` interval. A narrower prose estimate
that sits *inside* a wide interval (the common case — LLMs routinely narrow a wide numeric bound
into a more specific-sounding guess) is deliberately **not** flagged; that's a useful narrowing,
not a defect.

**Measured against the full current cassette** (`reports/lever4_prose_number_consistency.json`):
64 synthesis plans checked, 63 had an extractable time claim, **0 contradictions**. The
ADR-0037-cited k8s-14756 example does not reproduce in the currently active recording — its exact
quoted text ("configuration tweak") isn't present in `eval/cassettes/eval_cassette.json`, meaning
that specific instance was almost certainly captured from an earlier recording (no-fix/v1) ADR-0037
examined separately via an isolated-worktree replay, not the current v3 cassette this measurement
reads. Verified the extractor itself isn't silently under-matching: 98.4% of plans (63/64) had a
parseable time claim, and manual inspection of the extracted (implied, actual) pairs for a
15-plan sample showed sensible, correctly-parsed ranges throughout.

**Rate is not currently material — added the check anyway, informational only, not chasing a
zero-rate problem into a hard gate.** Wired `prose_number_contradiction_rate` into
`eval/run_eval.py::compute_scores()`'s `per_repo` output (same zero-extra-cost pattern as
`fabrication_rate`/`floor_fail_rate` — reads fields `TriagePlan` already produces, no new LLM
calls) and added `test_vscode_no_prose_number_contradiction` /
`test_k8s_no_prose_number_contradiction` to `eval/test_quality_regression.py`, asserting
`rate == 0.0` per repo. This file's CI job is already `continue-on-error: true` (informational),
matching the exact discipline `_check_no_fabrication` already established: build the standing,
zero-cost ratchet now, watch for regression, decide about hard-gating later if the rate ever moves
off zero for a real, current reason.

## Consequences

- **What changes:** a new importable module + two new informational test cases + one new
  per-repo metric. No production/serving code touched — this only runs inside the offline eval
  harness (`run_eval.py`), not `triage.py`'s live request path.
- **What becomes easier:** if a future prompt/model change reintroduces this defect, it's now
  visible in the eval report immediately rather than requiring another ADR-0037-style manual
  investigation to notice it.
- **What's deferred:** `test_vscode_no_prose_number_contradiction` /
  `test_k8s_no_prose_number_contradiction` currently `CassetteMissError` when run, for the same
  reason `test_grounding_ratchet_no_new_ungrounded_claims` does post-Cutover-A (ADR-0040's
  `eval_set.jsonl` refresh changed the synthesis cache key) — resolved by the already-deferred
  combined cassette re-record (retrieval + resolution + this lever, one pass), not touched here.
  Verified the check's own logic directly instead, via 11 unit tests
  (`tests/test_resolution_consistency.py`, including the exact k8s-14756 case) and the standalone
  cassette-content measurement script, both independent of the full pipeline replay.
- **Not done, deliberately:** did not add this as a live-serving field on `TriagePlan` (unlike
  `grounding_status`, which the live API does expose). The measured rate doesn't currently justify
  a schema/API surface change; the eval-harness-only wiring is proportionate to what was asked
  (measure, and if material, add to the deterministic verifier).

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Fix the k8s-14756 example specifically with prompt wording | GG's explicit framing: this is a correctness defect to catch deterministically, not a prompt-tuning problem — matches the project's `grounding.py` precedent (verify with code, don't hope the model self-corrects). |
| Skip building the check since the current rate is 0% | Rejected — the check is cheap (regex, zero LLM calls) and the whole point is catching a *future* regression before it needs its own investigation, not just describing today's snapshot. |
| Hard-gate immediately (assert-and-block) | No evidence yet that 0% is stable across cassette re-records; same "observe before hard-gating" discipline the project already applies to `fabrication_rate` and `floor_fail_rate`. |
| Expose as a live `TriagePlan` field / API surface | Bigger schema change than the measured rate justifies right now; can be added later exactly like `grounding_status` was, if a live-serving use case appears. |
