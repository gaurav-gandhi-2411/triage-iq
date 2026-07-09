# ADR-0022 — Structured Generation + Semantic Verification

**Status:** Accepted (verifier) / Not built (structured generation) — measure-first gate found
nothing for structured generation to fix, and the current synthesis model doesn't support it
anyway. The deterministic semantic verifier ships regardless — valuable independent of the rate,
per the spec.
**Date:** 2026-07-10
**Decider:** Gaurav Gandhi

## Context

Stage 4 synthesis (Groq `llama-3.1-8b-instant`) emits free-text JSON, parsed and validated
against `TriagePlan`. The existing ADR-0015 retry-cache path handles malformed JSON with a
re-prompt; `llm_status` reports which of `ok` / `parse_retry_succeeded` / `parse_failure`
occurred. This iteration asks two separate questions:

1. Would constraining generation to be schema-valid *at generation time* (structured output)
   measurably reduce malformed JSON, given the current rate?
2. Independent of (1): can a deterministic, pure-Python pass catch plans that are internally
   self-contradictory (not malformed JSON — valid JSON that says two contradictory things)?

**Measure first, because the answer to (1) determines whether to build it at all.** If the
malformed rate is already ~0%, there's nothing for structured generation to fix — an honest
finding, not a failure to find a problem.

## Results — the measure-first gate

`scripts/measure_synthesis_reliability.py` replays the clean, shipped-default cassette
(`eval/cassettes/eval_cassette.json`, `TRIAGE_PROMPT_INCLUDE_ATTRIBUTION` off) over all n=65
issues via `triage_with_metadata` — zero live calls. Full data: `reports/synthesis_reliability.json`.

| Repo | n | ok | parse_retry_succeeded | parse_failure | **malformed rate** |
|---|---|---|---|---|---|
| microsoft/vscode | 11 | 11 | 0 | 0 | **0.00%** |
| kubernetes/kubernetes | 54 | 54 | 0 | 0 | **0.00%** |
| **overall** | 65 | 65 | 0 | 0 | **0.00%** |

**Zero malformed-JSON events across all 65 recorded synthesis calls.** Every plan parsed as
valid JSON on the first attempt; the ADR-0015 retry path never fired in this recording.

### Groq structured-output availability — checked against the primary source, not assumed

A web search initially suggested `llama-3.1-8b-instant` might be deprecated — this was checked
directly against Groq's own model page (`console.groq.com/docs/model/llama-3.1-8b-instant`)
and **is false**: no deprecation notice exists there. The search summary conflated a different,
unrelated claim; flagging this here as a reminder that search summaries get verified against
the primary source before being acted on, same discipline as everywhere else in this project.

What **is** true, verified against `console.groq.com/docs/structured-outputs`: Groq's strict,
schema-constrained structured output (`response_format={"type": "json_schema", ...,
"strict": true}`, constrained decoding guaranteeing schema conformance) is supported **only**
for `openai/gpt-oss-20b` and `openai/gpt-oss-120b` (plus `meta-llama/llama-4-scout-17b-16e-instruct`
and `openai/gpt-oss-safeguard-20b` in best-effort, non-strict mode). **`llama-3.1-8b-instant` is
not on either list.** The only structured-output-adjacent feature it supports is JSON Object
Mode (`response_format={"type": "json_object"}`) — guarantees syntactically valid JSON, not
schema conformance, and Groq's own docs state it explicitly: "may not match your schema."

### Decision: structured generation is not built

Two independent reasons converge on the same answer:

1. **Nothing measured to fix.** 0/65 malformed — the existing retry-cache path (ADR-0015) is
   already handling whatever rare failures might occur in production at effectively 100%
   first-attempt success on this sample.
2. **Not available for the pinned model anyway.** Strict structured output requires
   `openai/gpt-oss-20b`/`120b`; changing the synthesis model is explicitly out of scope (it
   would invalidate the cassette and is a decision bigger than this ADR). The only thing
   available for `llama-3.1-8b-instant` — JSON Object Mode — targets syntactic validity, which
   is already at 100% on this measured sample; there is no headroom for it to demonstrate.

This is the honest either-outcome the spec asked for: **"malformed rate is already ~0" is a
valid, reportable finding, not a failure to find something to build.** Scoped down to
verifier-only, per the spec's own pre-authorized fallback.

## Decision: the semantic verifier ships (built regardless of the rate)

`src/triage_iq/models/plan_verify.py` — `verify_plan_consistency(plan) -> ConsistencyReport`.
Deterministic, pure Python, no LLM, no external signals (contrast `grounding.py`, which checks
the plan against upstream pipeline signals — this checks the plan **against itself**).
FLAG-not-strip: never raises on well-formed input, never mutates the plan. Wired into
`TriageAssistant.triage_with_metadata`, unconditionally — same as `grounding_status`, not
flag-gated, since it can never change synthesis output or block a response.

### Consistency rules (kept deliberately narrow — 2 rules, both on structured fields only)

1. **`priority_resolution_consistent`** — False iff `priority_guess == "high"` AND
   `resolution_bucket` is `"months"` or `"long"`. A plan claiming both maximum urgency and a
   multi-month timeline contradicts itself. **One-directional by design**: low/medium priority
   at any timeline, or high priority at hours/days/weeks, are all consistent — this flags only
   the single clearest contradiction, not a style preference about which priorities "should"
   pair with which timelines.
2. **`override_reason_consistent`** — False iff `declared_attribution.component_source ==
   "model_override"` but `component_override_reason` is blank (or whitespace-only). The schema
   (ADR-0020) already documents this field as *required* in that case — this rule enforces the
   schema's own stated contract. Vacuously true when `declared_attribution` is absent (field is
   optional/flag-gated, ADR-0020).

### Rules considered and explicitly NOT implemented

The spec's illustrative examples — "a next-step references a component not in the plan," "no
next-step references a nonexistent field" — were not built. Reliably extracting a component
reference from free-form `suggested_next_steps` text without an LLM is a substring/NLP problem:
a next-step like "coordinate with the networking team" mentioning a different component is
often legitimate cross-team coordination, not a contradiction, and a naive keyword match would
have real false-positive risk. That crosses the "flag clear contradictions, not stylistic
judgments" bar this project holds every heuristic to (same discipline as ADR-0021's rejection
of a fabricated priority-confidence proxy). Two precise rules beat five fragile ones.

### Measured inconsistency rate

Same replay, same n=65: **0/65 inconsistent** on both rules, both repos (see
`reports/synthesis_reliability.json`). Consistent with the honest-either-outcome framing: the
verifier finding nothing on this sample doesn't mean it's not worth having — it's a regression
gate for future prompt/model changes, same as `grounding_status` mostly returning
`all_grounded=True` in production while still being the thing that would catch a regression.

### TEST THE TEST — the ratchet demonstrably has teeth, not a vacuous assertion

Because the measured baseline is 0/65 inconsistent for both repos, there is no naturally
occurring bad case to pin by issue number (unlike the grounding ratchet's #13057/#311836).
Instead, the ratchet's failure path was exercised directly:

1. Ran the real ratchet check against real data: `k8s inconsistent_count=0, baseline=0` → pass.
2. **Injected** one synthetic case (`all_consistent=False`) into the replayed case list and
   re-ran the identical assertion logic: `AssertionError: kubernetes/kubernetes: inconsistent-plan
   count regressed: 1 > baseline 0` — the ratchet failed exactly as it should.
3. **Reverted** (removed the injected case), re-ran: pass again, `inconsistent_count=0`.

This proves `test_consistency_ratchet_no_new_inconsistent_plans` (`eval/test_invariants.py`)
would actually catch a future regression rather than trivially passing regardless of input.
Not committed as a permanent test fixture (it isn't real eval-set data) — recorded here as the
verification, mirroring how ADR-0019's jitter measurement was a direct one-time experiment.

### Additive schema

`ConsistencyStatus` (`priority_resolution_consistent: bool`, `override_reason_consistent: bool`,
`all_consistent: bool`) lands as `TriagePlan.consistency_status`, `None`-safe. Excluded from the
judge's `plan_json` in `eval/run_eval.py` (same pattern as `declared_attribution`/
`abstention_status`) — this harness *does* populate the field (unlike those two, this check runs
unconditionally inside `triage_with_metadata`), but excluding it keeps the judge cache key tied
to the byte-for-byte `plan_json` the baseline means were computed from, rather than silently
drifting when a new always-on field lands.

## Consequences

- **Nothing re-recorded, nothing re-baselined.** The verifier is scoring-only (reads existing
  fields, doesn't touch the synthesis prompt); `reports/eval_baseline.json` and
  `eval/cassettes/eval_cassette.json` are both unchanged — confirmed (`run_eval.py` still
  reproduces the exact baseline means and cassette hash).
- **Structured generation is not built, not flag-gated, not scaffolded** — there is no
  `TRIAGE_STRUCTURED_SYNTHESIS` flag in this codebase. Building an unused flag for a feature
  with no measured justification and no model support would be exactly the "no half-finished
  implementations" anti-pattern this project avoids elsewhere. If the synthesis model ever
  changes to one of the `openai/gpt-oss-*` models, or a future measurement shows a non-zero
  malformed rate, this decision should be revisited from scratch (a model change already forces
  a cassette re-record regardless).
- `consistency_status` is additive and always computed (like `grounding_status`) — no env var
  gates whether it's populated; nothing about it can change synthesis behavior or block a
  response (FLAG-not-strip).
- The consistency invariant (`eval/test_invariants.py::test_consistency_ratchet_no_new_inconsistent_plans`)
  guards against future regressions using a data-derived baseline (0/65), tied to
  `eval_set_hash` like the grounding ratchet — will loudly demand re-derivation if the eval set
  changes.
- Generalization caveat: the 0% rates above describe this exact prompt/model pair and this exact
  gold set. They are not a claim that malformed JSON or inconsistent plans can never happen in
  production — only that they didn't occur in this n=65 recorded sample. The retry-cache path
  (ADR-0015) and the new verifier both remain as safety nets for whatever this sample didn't
  surface.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Build structured generation anyway ("might help later") | No measured problem to fix (0% malformed) and the pinned model doesn't support strict schema mode — would be building an untested, unused code path against a hypothetical, violating "measure before building." |
| Switch synthesis model to `openai/gpt-oss-20b`/`120b` to unlock strict structured output | Explicitly out of scope — a model change invalidates the cassette and is a decision with its own cost/quality tradeoffs, bigger than this ADR's scope. |
| Implement the free-text next-step/component-reference rules from the spec's illustrative examples | Would require substring/NLP heuristics with real false-positive risk (e.g. legitimate cross-team mentions) — fails the "flag clear contradictions, not stylistic judgments" bar. Two precise, structured-field rules were chosen over several fragile text-based ones. |
| Pin known bad-case issue numbers for the consistency ratchet (mirroring the grounding ratchet) | Not possible honestly — the measured baseline is 0/65, there is no naturally occurring inconsistent case in this data to pin. TEST THE TEST (inject/observe-fail/revert) substituted as the teeth-proving mechanism instead. |
