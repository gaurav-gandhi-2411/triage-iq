# ADR-0052 — TriageIQ currently has no valid LLM-quality baseline

**Status:** Accepted
**Date:** 2026-08-28
**Decider:** Gaurav Gandhi

## Context

Two independent, unrelated failures have each destroyed the validity of one of TriageIQ's
two committed synthesis cassettes:

1. **main's cassette** (`llama-3.1-8b-instant`, ~129 entries): the model it was recorded
   against is deprecated by Groq. Confirmed live: a direct API call to
   `llama-3.1-8b-instant` returns `404 model_not_found`. This cassette cannot be extended,
   re-recorded, or re-verified against its own model ever again — there is nothing left to
   call.
2. **`fix/groq-deprecated-model-p0-reduced`'s cassette** (`openai/gpt-oss-20b`, PR #106,
   438 entries): recorded with `max_tokens=1024`, which truncated 44/64 (68.75%) of
   synthesis completions on first attempt (confirmed via live re-run: every failing
   entry's cached `usage.completion_tokens` equals exactly 1024, and raw content is clean
   JSON cut off mid-field, not malformed or reasoning-trace-contaminated). 18/64 issues
   (1 vscode, 17 k8s) ended up as fully degraded, predictor-only fallback plans after both
   attempts failed, scored by the judge as genuine model output. ADR-0051 (withdrawn same
   day) documents this in full.

## Decision — state the actual situation plainly

**TriageIQ has no valid grounding, fabrication-rate, or judge-quality baseline right now.**
Not "a degraded one" and not "the old one held over as a placeholder" — none. This is not
a bypass of process, and it is not being softened for narrative comfort: it is the
accurate current state, arrived at by two separate root-caused failures, neither of which
is recoverable by re-measuring the existing data.

- The `llama-3.1-8b-instant` baseline (README.md's "0.0% (0/53)" / "0.0% (0/11)"
  fabrication-rate table, `docs/PROJECT_STATE.md`'s "Judge 10.75/15" figure) was a real,
  validly-measured number **at the time it was recorded**. It is now unreproducible: the
  model that produced it no longer exists to be called, so it can never be re-verified,
  extended to more issues, or used as a like-for-like comparison point again. A number
  that can never be re-measured is not a baseline going forward — it's a historical record
  of a system that no longer exists.
- The `gpt-oss-20b` numbers (3/11 ungrounded, judge scores in the withdrawn
  `eval_baseline.json`) were never valid in the first place — they were measured against a
  cassette contaminated by a config defect (truncation) that changed what was actually
  being scored for over two-thirds of the sample.

**Therefore: whichever model Part B's bake-off selects, and whatever it measures under
the corrected configuration (proper `max_tokens` from the real completion-token
distribution, native structured output, `TruncatedCompletionError` making silent
truncation impossible going forward), will be the first genuinely valid quality baseline
this project has had.** Not a re-baseline of a prior number. Not a restoration. The first
one that was ever actually measured cleanly, end to end, on the model it will actually
ship with.

## Consequences

- Production stays down until that measurement exists and a model is selected on it —
  there is no fallback to "the old baseline" to serve traffic against, because the old
  baseline's model doesn't exist anymore either.
- Every existing doc/README claim tied to `llama-3.1-8b-instant`'s numbers is now
  historical record, not a defensible current-state claim. See the companion audit
  (2026-08-28 diagnostic session, Part D3) for the specific list and what needs updating
  once Part B concludes.
- The eval-set expansion work (scoped, not started — see the Part D-eval-expansion plan
  from the prior session) becomes more urgent, not less: the *first* real baseline this
  project gets will still be measured on the current small eval set (11 vscode / 53 k8s)
  until that expansion lands, so it inherits the same statistical-power limitations
  ADR's grounding-rate non-inferiority proposal (PR #110) already documented.

## Alternatives considered

- **Treat the withdrawn `gpt-oss-20b` numbers as "directionally useful" and keep them as a
  placeholder baseline.** Rejected: a number known to be measured under a broken config is
  not directionally useful, it's actively misleading — 68.75% of its inputs were degraded
  by a mechanism (truncation) that has nothing to do with model quality.
- **Hold the `llama-3.1-8b-instant` baseline as "the standard to beat" even though the
  model is dead.** Rejected: a standard measured against an uncallable model cannot be
  re-verified, extended, or used for a fair comparison — it's a number, not a baseline.
- **Soften this to "baseline temporarily unavailable, pending re-measurement."** Rejected
  per explicit instruction: this is not temporary unavailability of an otherwise-intact
  baseline, it's the accurate state that no valid baseline currently exists, full stop.

## Update 2026-08-29 — the `max_tokens=1024` defect is a still-shipped default, not just a
## recording-time incident, and the bake-off's own prompt changes are a second, separate
## comparability break

**Correction to how this ADR originally framed the `max_tokens=1024` truncation defect.**
The Context section above (and ADR-0051) described it as something that happened *at
recording time* for PR #106's cassette — implying it was a one-off mistake in a specific
recording run. That framing was incomplete. `TriageAssistant.__init__`'s constructor
default for `max_tokens` was `1024` on `main` **before** PR #106 existed, is unchanged by
PR #106, and remained `1024` in PR #113 (the token-budget-guard fix) until this session
found it: no call site anywhere in the codebase (`api/loader.py`, `eval/run_eval.py`,
`eval/record_cassettes.py`, `scripts/11_evaluate_triage.py`) ever overrode it. Observed
real completions run 1,031–1,919 tokens — above 1024 almost everywhere — so **any real
`/triage` call that reaches a live model has been truncating most of its completions all
along**, not just during the one PR #106 recording session this ADR originally singled
out. PR #113 fixes this by making the value an explicit, env-controlled setting
(`TRIAGE_MAX_TOKENS`, default `2048` — see `src/triage_iq/config.py`) that every real call
site now passes explicitly, instead of falling through to a constructor default nobody
could see.

**This means the defect diagnosed here was live in production, continuously, and is not
specific to any one cassette.** It happened to be *caught* via PR #106's cassette because
that recording session inspected `usage.completion_tokens` directly; main's own
`llama-3.1-8b-instant` cassette was never audited for the same thing before the model was
retired, so whether main's historical judge/fabrication numbers were themselves
truncation-degraded is now unknowable — one more reason (beyond the model being
uncallable) that those numbers cannot be treated as a real baseline.

**Second, separate comparability break, on top of the one this ADR already declares.**
Fixing `max_tokens` changes `TriageAssistant._call_llm_verbose`'s cache key for every
call, because `LLMCache.compute_key` includes `max_tokens` as an input (confirmed: strict
replay of `eval/cassettes/eval_cassette.json` against the fixed code raises
`CassetteMissError` on the very first call). Separately, Part B's few-shot-removal arm (if
it wins the bake-off) changes the prompt text itself, which also changes every cache key.
Both changes are individually necessary and both are being made deliberately, not
discovered as an accident after the fact — but together they mean:

- **No number produced by the Part C bake-off is comparable to any number this project has
  ever recorded before** — not `llama-3.1-8b-instant`'s historical numbers (already voided
  above), not PR #106's withdrawn `gpt-oss-20b` numbers (already voided above), and not
  even a hypothetical "just fix `max_tokens` and re-run the old prompt" measurement, since
  that alone already breaks the cache key against every existing cassette.
- This is consistent with, not an exception to, this ADR's core decision: whichever model
  and prompt shape the bake-off selects **is** the first genuinely valid baseline, measured
  under the corrected `max_tokens` handling, under whichever prompt shape (few-shot or not)
  the pre-registered criteria in `docs/eval/bakeoff_prereg_2026-08-29.md` select. Once that
  baseline exists, `eval/cassettes/eval_cassette.json` and `reports/eval_baseline.json` get
  re-recorded against it and become the new comparison point for everything after. Nothing
  before it is comparable to anything after it, by construction, and that is intentional.
