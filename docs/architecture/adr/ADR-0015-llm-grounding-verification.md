# ADR-0015: LLM Grounding Verification

Status: Accepted
Date: 2026-07-04

## Context

`TriageAssistant`'s synthesis step (System 4) asks an LLM to produce a `TriagePlan` that
includes a `predicted_component` and a list of `similar_issues` with citation-like
`relevance_note`s. Both claims are presented to the LLM as *inputs* (`classifier_top3` from
System 1, retrieved similar issues from System 2) but the LLM is free to restate, drop, or
invent values in its JSON output — nothing previously checked whether what it emitted matched
what it was actually shown.

This is a narrow but real trust gap: an LLM can name a component that never appeared in
`classifier_top3`, or cite a "similar issue" number that was never retrieved for this request
(hallucinated ref). Neither failure mode was detectable before this ADR — the `TriagePlan`
Pydantic schema validates shape (is `predicted_component` a string? is `similarity` in
`[0,1]`?) but not provenance (did this value come from an upstream deterministic system, or
did the LLM make it up?).

## Decision

Add a pure, deterministic verifier (`src/triage_iq/models/grounding.py:verify_plan_grounding`)
that checks a `TriagePlan`'s claims against the pipeline's own upstream signals for that
request:

- **Component grounding is strict `classifier_top3` membership, not label-space membership.**
  `predicted_component` (after `.strip()`) must exact-match one of the (at most 3) labels the
  classifier actually surfaced for this request — not just any label the classifier knows
  about. This is deliberately narrower than "is this a real component name"; it answers "did
  System 1 actually suggest this for this issue," which is the provenance question we care
  about. Match is case-sensitive; a case/whitespace-only mismatch (`"Editor"` vs `"editor"`)
  is treated as ungrounded, not silently normalized — see `test_case_and_whitespace_mismatch_is_ungrounded`.
- **Similar-issue grounding checks each cited ref against the actual retrieved set**
  (`signals["similar_raw"]`, System 2's output for this request), not against `plan.similar_issues`
  itself (which is the LLM's own claim, not ground truth).
- **Policy is FLAG, not strip.** An ungrounded claim is surfaced via a new `grounding_status`
  field, not silently removed from the response. Stripping would hide the failure from anyone
  inspecting a plan and would require deciding what to substitute; flagging preserves the raw
  LLM output while making its unreliability visible to both the API consumer and the UI.
- **Additive fields, reconstructed, not newly elicited.** Two new optional fields land on
  `TriagePlan`:
  - `grounding: GroundingAttribution | None` — `component_source` and `similar_issue_refs`,
    a **reconstruction** of what the LLM already said in this response (no new prompt turn).
  - `grounding_status: GroundingStatus | None` — the verifier's output: `component_grounded`,
    `component_reason`, `similar_issue_refs`, `ungrounded_refs`, `all_grounded`.

  Both default to `None` so existing consumers (and the schema-contract test,
  `eval/test_invariants.py::test_triage_plan_schema_contract`) are unaffected. **The synthesis
  prompt itself is unchanged this iteration** — grounding is computed post-hoc in
  `triage_with_metadata` by re-deriving the same values the LLM already emitted and diffing
  them against `signals["classifier_top3"]` / `signals["similar_raw"]`. This is explicitly not
  "ask the LLM to cite its sources" (a prompt-attribution change); it's "verify what it said
  against what it saw."

## Measured baseline

`scripts/measure_grounding.py` replayed the *current, unmodified* synthesis cassette
(`eval/cassettes/eval_cassette.json`) over all 60 eval-set issues (30 vscode + 30 k8s) through
`verify_plan_grounding` — zero live LLM calls. Result: **2/60 (3.3%) issues have at least one
ungrounded claim**, both in kubernetes/kubernetes:

- **#1678** — component grounded (`predicted_component="test"` is in `classifier_top3`), but
  one similar-issue ref (`13632`) was not in the actual retrieval set for this request.
- **#13435** — `predicted_component="cluster/bootstrap"` is not in `classifier_top3`
  (`['provider/gcp', 'platform/vagrant', 'example']`) — component hallucination.

Reported honestly as a baseline, not cherry-picked: microsoft/vscode has 0/30 ungrounded
claims in this measurement; kubernetes/kubernetes has 2/30. This measurement is committed at
`reports/grounding_measurement.json`.

## Eval-gate ratchet + known-case pin

Two new tests in `eval/test_invariants.py`, sharing a module-scoped `grounding_reports`
fixture (same cassette-replay pipeline as `scripts/measure_grounding.py`, factored into a
shared `compute_grounding_reports()` helper so the eval-gate test and the measurement script
never duplicate the wiring):

1. **`test_grounding_ratchet_no_new_ungrounded_claims`** — asserts the current ungrounded
   count is `<=` the recorded baseline (`_GROUNDING_BASELINE["ungrounded_count"] == 2`). This
   is the regression gate: it fails if grounding gets *worse*.
2. **`test_grounding_known_cases_still_flagged`** — asserts the two *specific* known-bad cases
   (#1678, #13435) are still individually caught by issue number, not just reflected in an
   aggregate count.

**Why both are needed — the no-op-verifier blind spot:** a verifier that always reports
`all_grounded=True` (a regression to a no-op, e.g. from a refactor that accidentally drops the
comparison logic) would produce an ungrounded count of 0, which trivially satisfies
`0 <= 2` — the ratchet alone would pass on a completely broken verifier. The pin test closes
this gap by requiring the two known cases to be caught *by name*, so a no-op verifier fails
loudly instead of silently passing. Both tests share a hash guard: they assert
`sha256(eval/eval_set.jsonl) == _GROUNDING_BASELINE["eval_set_hash"]` first and fail loudly
("re-derive `_GROUNDING_BASELINE` deliberately, do not silently compare across different
sets") if the eval set has changed, rather than comparing stale pinned issue numbers against a
set that may no longer contain them.

This mirrors the existing `_RECORDED_ECE` pattern in the same file and the
`current_scores`/`baseline` session-fixture pattern in `eval/test_quality_regression.py`.

## Explicit deferral: prompt-attribution change

A separate, more invasive change — modifying the synthesis prompt to explicitly ask the LLM to
cite which classifier label and which retrieved issue numbers it used — is **not** undertaken
in this iteration. Two reasons:

1. It would change the synthesis request, which changes the cassette key, which requires a
   full cassette re-record (`python eval/record_cassettes.py`) and a new
   `reports/eval_baseline.json` (per the ADR-0011 baseline-update procedure) — an expensive
   step to justify an *unmeasured* hypothesis.
2. There is no evidence yet that prompt-level attribution would reduce the 3.3% ungrounded
   rate below what this post-hoc verifier already catches. Making that change is only
   justified by a later measured A/B (current unchanged-prompt cassette vs. a
   re-recorded attribution-prompt cassette, both scored through the same grounding verifier)
   showing a real reduction — not by intuition that asking nicely produces more honest output.

This is a scoped, deferred follow-up, not a rejected idea.

## Honest limitation

**"Grounded" here means "traceable to this pipeline's own deterministic outputs for this
specific request"** — i.e., did the LLM's claim match something System 1 or System 2 actually
produced. It is **not** verification against world/ground truth: a `predicted_component` that
is in `classifier_top3` could still be the wrong component for the issue (the classifier
itself could be wrong); a cited similar-issue number that was genuinely retrieved could still
be irrelevant (retrieval could be wrong). This ADR closes the "did the LLM hallucinate a value
that was never shown to it" gap, not the "is the underlying prediction correct" gap — the
latter is what `eval/test_quality_regression.py`'s LLM-judge scoring already measures
separately.

## Consequences

- **What changes:** `TriagePlan` gains two additive, optional fields (`grounding`,
  `grounding_status`); `triage_with_metadata` computes them post-synthesis with zero prompt
  changes; the `/triage` API response includes them automatically via
  `response_model=TriagePlan` / `plan.model_dump()` (no manual wiring needed in `app.py`); the
  UI's "Under the hood" panel surfaces per-claim grounding markers.
- **What becomes harder:** nothing — this is purely additive. Existing consumers of the API or
  cached responses without these fields are unaffected (`None` default).
- **What becomes easier:** a hallucinated component or fabricated similar-issue citation is now
  visible in the response and caught by an automated eval-gate regression test, instead of
  being silently indistinguishable from a grounded claim.
- **What's still open:** the prompt-attribution question above; and this verifier does not
  (and is not intended to) evaluate whether the underlying classifier/retrieval predictions
  are themselves correct.

## Alternatives considered

- **Strip ungrounded claims from the response** — rejected. Hiding the LLM's actual output
  makes debugging harder and forces an unprincipled choice of substitute value; flagging
  preserves the raw output while making its provenance visible.
- **Label-space membership for component grounding** (any known component label, not just
  top-3) — rejected as too permissive: it would pass a component the classifier considered and
  ranked outside its own top-3, which is exactly the kind of "technically a real word, not
  actually what System 1 said" hallucination this ADR is meant to catch.
- **Case-insensitive / whitespace-normalized component matching** — rejected for the grounding
  verdict itself (kept only as a separate diagnostic signal, `case_insensitive_match_diagnostic`,
  in `scripts/measure_grounding.py`) so the strict grounding signal stays exact and auditable;
  the diagnostic lets us see how much of any future ungrounded rate is trivial capitalization
  drift versus real hallucination without softening the primary check.
- **Elicit attribution via a prompt change in this same iteration** — rejected per the deferral
  section above: unmeasured hypothesis, expensive cassette re-record, no evidence it improves
  on what post-hoc verification already catches.
