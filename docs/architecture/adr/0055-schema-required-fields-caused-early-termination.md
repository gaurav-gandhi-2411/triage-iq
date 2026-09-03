# ADR-0055: The early-termination defect was substantially an output-contract defect, not a model defect

Status: Accepted
Date: 2026-09-03

## Context

ADR-0054's original (retracted) basis for selecting `openai/gpt-oss-120b` over
`openai/gpt-oss-20b` was a parse-success/early-termination gap. Investigating that gap
after the 44/44 retraction (see ADR-0054's correction) found something more fundamental:
every early-termination failure examined across BOTH models -- 5 from the
`openai/gpt-oss-120b` 64-issue re-record (`k8s-12224`, `vscode-4996`, `k8s-12477`,
`k8s-12248`, `k8s-13508`) and 4 from the bake-off screen (Arm B/no-few-shot:
`k8s-12287`, `vscode-4993`, `k8s-14835`; Arm A/`gpt-oss-20b`-few-shot: `k8s-12254`) --
failed on the identical subset of a fixed set of 7 fields, out of `TriagePlan`'s 18
schema-required fields:

`resolution_bucket`, `resolution_confidence_pct`, `resolution_interval_conformal`,
`grounding`, `grounding_status`, `declared_attribution`, `abstention_status`.

**Every one of these 7 is either overwritten post-hoc by
`TriageAssistant.triage_with_metadata` / `app.py` regardless of what the model emits
(6 of 7, confirmed by direct code read: `triage.py:578-597`, `app.py:329,360`), or
already treated as a compliance failure rather than a request failure when missing or
malformed (`declared_attribution`'s `tolerant_attribution` validator, 1 of 7).** The
model was never asked to derive real signal for these fields -- their own descriptions
say so explicitly ("Always emit exactly X", "Always emit null ... you cannot derive
this"). Groq's `strict: true` structured-output mode has no notion of an optional
property (`_force_strict_schema_requirements`'s own docstring, confirmed by trial before
this session): every key in `properties` is forced into `required` regardless of the
Pydantic model's own defaults. The model was being asked to spend generation budget,
under a hard schema gate, on 7 fields nothing downstream consumes -- and reliably
dropped a subset of them under constrained decoding, producing a 400
(`json_validate_failed`) instead of a usable completion.

**Fine-grained pattern (all 9 failures, field-by-field):** the 4 fields missing in
100% (9/9) of failures were `grounding`, `grounding_status`, `declared_attribution`,
`abstention_status` -- all four of `TriagePlan`'s `X | None` fields. 3 more
(`resolution_bucket`, `resolution_confidence_pct`, `resolution_interval_conformal`)
were missing in 8/9. The stopping point was not perfectly fixed (Arm B's 2 worst
failures dropped 15/18 fields, stopping right after `similar_issues`; the 5
`gpt-oss-120b` failures and Arm A's all stopped after `triage_summary`, dropping
exactly 7) -- but the *set* of fields ever missing, across every failure examined, was
always a subset of these same 7. `similar_issues` (also a defaulted field, via
`default_factory=list` rather than `default=...`) was never among them in any failure
-- confirmed empirically, not assumed; see the distinction this ADR's fix relies on,
below.

## Decision

**Strip these 7 fields from the wire schema entirely** (not just from `required` --
Groq's strict mode requires `required` to equal every key in `properties`, so the only
way to stop demanding a field is to remove it from `properties` too). Implemented as
`_strip_post_hoc_fields` (`src/triage_iq/models/triage.py`), called before
`_inline_nullable_object_refs`/`_force_strict_schema_requirements` in
`_build_triage_plan_response_format()`. The exclusion criterion is general, not a
hardcoded field list: any `TriagePlan` field with an explicit Pydantic `default=...`
(not `default_factory`) is excluded. This distinction is load-bearing, not
incidental -- `default=...` means "a fixed/derived value the model cannot affect
either way" (exactly this defect's 7 fields); `default_factory` means "the model
should try, an empty result is an acceptable fallback" (`similar_issues`, which
carries real signal and was never among the missing fields in any failure examined).
Confirmed via direct Pydantic field inspection that this criterion produces exactly
the 7 target fields and nothing else (`resolution_bucket`, `resolution_confidence_pct`,
`resolution_interval_conformal`, `grounding`, `grounding_status`,
`declared_attribution`, `abstention_status` all have `default=...`; `similar_issues`
has `default_factory` and is correctly excluded from the strip).

**Result: 18 -> 11 required fields.** Nested `$defs` no longer reachable
(`ConformalIntervalResult`, `GroundingAttribution`, `GroundingStatus`,
`DeclaredAttribution`, `AbstentionStatus`, `StageAbstention`) are pruned by the
existing `_prune_unreferenced_defs` reachability pass -- only `SimilarIssue` remains.
Wire schema JSON shrank 8,220 -> 2,417 chars (2,105 -> 621 cl100k tokens, a proxy
measurement, 70.5% smaller) -- **this does NOT translate into a measurable calls/day
improvement.** The existing token-budget guard's own calibration
(`_estimate_prompt_tokens`) is fit only against `messages` content and matches Groq's
real reported `prompt_tokens` with low error using that message-only estimate; if the
schema were counted in Groq's TPM/TPD accounting the same way, that calibration would
show a large systematic underestimate, which it does not. The schema's real-dollar or
real-quota cost, if any, is not visible in the same `usage.prompt_tokens` figure this
pipeline's guard is built around -- **unverified, not assumed zero either way.** The
genuine quota benefit is indirect: a failed call still generates real, billable
completion tokens before Groq's post-hoc schema validation rejects it (confirmed --
every `failed_generation` payload examined contained a substantial, mostly-complete
JSON body), so eliminating these failures recovers that wasted spend. At the
corrected/verified pooled `gpt-oss-120b` failure rate (5.95%, ADR-0054), this is a
modest, secondary effect, not the primary justification for this change.

**Prompt text is unchanged.** Confirmed via `git diff --stat`: only
`src/triage_iq/models/triage.py` changed for this fix; `src/triage_iq/prompts/
triage_prompt.py` was not touched. The model is asked to do the same task; only the
wire contract's demands changed.

**No behavior change for the 6 post-hoc-overwritten fields' final values** -- they
were already being overwritten after synthesis regardless of what the model emitted
(pre-existing code, unchanged by this fix); the only change is that the model is no
longer required to emit a placeholder for them first. `declared_attribution`'s
existing tolerant-parse behavior is also unchanged -- still `None` when absent or
malformed, still a compliance signal, not a request failure.

## Validation (2026-09-03, before any full re-record)

1. **Zero-quota dry-run gate:** `scripts/record_cassettes_dry_run_check.py` --
   PASSED against the reduced schema.
2. **Full test suite:** 302 passed (6 new tests added,
   `tests/test_wire_schema_excludes_post_hoc_fields.py`, pinning the 18->11 count,
   `required == properties`, the defaulted-field exclusion criterion generically (not
   just the 7 named fields), `similar_issues` untouched, and orphaned-`$defs` pruning).
3. **Small live validation, the 9 known-failing issues, under the CURRENT shipping
   config (`openai/gpt-oss-120b`, few-shot, reduced schema), cache bypassed
   (`cache=None`) to force a genuinely live call for every issue -- a stale cassette
   entry from the screen's Arm C run (old schema) would otherwise have silently
   replayed as a false-positive "still works," which is exactly what happened on a
   first attempt with the cassette enabled before this was caught (see "A related gap
   found live" below):**
   - **All 5 real `openai/gpt-oss-120b` re-record failures succeeded**
     (`k8s-12224`, `vscode-4996`, `k8s-12477`, `k8s-12248`, `k8s-13508`) --
     confirmed via genuinely new cache entries (verified by exact request-content
     match, not inferred from the summary line alone). **The specific defect that
     killed the 64-issue re-record and motivated this fix is resolved for all 5
     known instances of it.**
   - **`vscode-4993` (an Arm B/no-few-shot screen failure, not one of the 5 real
     `gpt-oss-120b` failures, but re-tested here under the actual shipping
     config for the first time) FAILED again -- a NEW, different failure mode: the
     model emitted a malformed key `"triage_summary layman"` instead of
     `triage_summary`, so schema validation correctly rejected it as missing the
     real key.** Reproduced independently twice (once in the cache-enabled run
     before the stale-cache issue was understood -- that run made a genuinely new
     live call for this specific issue, confirmed via cache-key analysis -- and once
     in the cache-bypassed re-run). **This is not the field-omission defect this ADR
     fixes; it is evidence the wire-contract simplification does not make the model
     perfectly reliable, only removes one specific, well-characterized failure mode.**
   - The remaining 3 issues (`k8s-12287`, `k8s-14835`, `k8s-12254`) could not be
     confirmed under the true reduced schema before Groq's daily quota was exhausted
     mid-run (`TPD: Limit 200000, Used 197689+`, 2026-09-03) -- their apparent
     "success" in the cache-enabled run was a stale replay of the screen's old-schema
     Arm C entry, not a genuine test, and must be re-run once quota resets before
     being counted either way.
4. **Per the working agreement's explicit instruction, this is a STOP, not a proceed:
   the full 64-issue re-record has NOT been started.** The original defect class is
   resolved for the 5 cases that motivated this ADR; a different defect was found on
   a 6th, previously-untested-under-shipping-config issue; 3 more remain unconfirmed
   on quota exhaustion. Recommend resuming validation (the 3 unconfirmed issues) once
   quota resets, and deciding on `vscode-4993`'s new failure mode (accept as a
   residual rate, or investigate further) before starting the full re-record.

**A related gap found live, now fixed:** the cassette's own cache key
(`CassettePlayer.compute_key` -> `LLMCache.compute_key`, keyed on
provider+model+messages+temperature+max_tokens) does not include `response_format`
either -- the same gap class as the checkpoint's `prompt_hash` (Part B below), in a
different mechanism. This nearly produced a false-positive validation result (3 of
the first 9 "successes" were stale old-schema replays, not new-schema tests) before
being caught by checking which cache entries were genuinely new. Not fixed in this
pass (the validation script now works around it with `cache=None`) -- flagging as a
separate, real gap in `eval/cassette.py`/`triage_iq/cache/llm_cache.py` for whoever
next changes the response contract and expects the cassette to notice.

## Correcting a related citation (Part D2 of the originating session's instructions)

The 68.75% first-attempt JSON-parse failure rate that halted the earlier
`openai/gpt-oss-20b` swap (PR #101/#106, 2026-08-27, and the same-day withdrawal of
ADR-0051) is **NOT** explained by this ADR's finding. That figure is explicitly
attributed, in this repo's own history, to a **`max_tokens=1024` token-budget
truncation** -- "recorded with `max_tokens=1024`, which truncated 44/64 (68.75%)" --
a distinct, already-diagnosed, already-fixed defect (the dynamic per-request token
budget guard and `TruncatedCompletionError`, PR #113, landed before this session).
Truncation (`finish_reason == "length"`, a genuine mid-generation cutoff) and this
ADR's defect (a syntactically complete, cleanly-closed JSON object that simply omits
non-derivable fields, rejected by Groq's post-hoc schema validator) are mechanically
different failures with different signatures in Groq's own error responses --
conflating them would be inaccurate, not merely imprecise citation.

**What DOES hold, confirmed by this session's own direct analysis of the raw error
payloads:** Arm B's actual elimination in the 2026-08-29/30 bake-off screen (3/8
failures, 62.5% success, `k8s-12287`/`vscode-4993`/`k8s-14835`) IS the schema-field
defect this ADR fixes -- all 3 failures' `failed_generation` payloads were examined
directly and show the identical missing-field pattern documented above. So the
broader point survives even though the specific number cited does not: **Arm B's
elimination (ADR-0053) traces to a schema artifact, not a `gpt-oss-20b`-without-
few-shot-specific model defect** -- see Consequences below for what this implies
about ADR-0053's few-shot decision, and what it does not.

## Consequences

- **What changes:** `TriagePlan`'s wire schema sent to Groq now requires 11 fields,
  not 18. `record_cassettes.py`'s checkpoint hash now covers the schema (Part B,
  below) -- a prompt-only hash would have silently mismatched requests made under
  different schemas as "the same recording."
- **What this does NOT do:** reopen ADR-0054's model selection (already re-grounded
  on truncation headroom, independent of this finding) or ADR-0053's few-shot
  decision (see below -- explicitly not reopened, a re-measurement question is
  raised but not acted on).
- **What becomes easier:** any future schema field addition automatically gets the
  same treatment -- `_strip_post_hoc_fields` is a general mechanism (any field with
  an explicit Pydantic `default=...`), not a hardcoded list, so a newly-added
  post-hoc-overwritten field is excluded from the wire schema without a code change
  at the exclusion site, only a regression-test update
  (`tests/test_wire_schema_excludes_post_hoc_fields.py`'s required-count pin).
- **What remains open:** `vscode-4993`'s new failure mode (malformed key name) is
  unexplained and unfixed -- this ADR does not claim the defect class is eliminated,
  only that the specific, well-characterized field-omission shape is resolved for
  the cases that motivated the investigation. The cassette's own schema-unaware
  cache key (found live, described above) is a separate, real gap, not fixed here.

## Alternatives considered

- **Keep all 18 fields required, add client-side retry-with-repair on a 400.**
  Rejected: treats the symptom (a rejected response) rather than the cause (asking
  for values nothing consumes), and a retry still burns real tokens on the same
  wasted fields every attempt.
- **Drop `strict: true` entirely, parse leniently.** Rejected, more invasive and
  loses Groq's structural JSON-schema enforcement for the fields that DO matter
  (a much larger behavior change, undermining the reliability structured output was
  adopted for in the first place -- see the 2026-08-28 commit history).
- **Fix only the specific 7 named fields, not the general default-based mechanism.**
  Rejected per the explicit instruction to fix the root cause, not just this
  instance -- a hardcoded 7-field list would silently miss the next
  post-hoc-overwritten field added to `TriagePlan`.
