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

## Update 2026-09-05 — Resolved: the baseline this ADR called for now exists

**Note on how this update was written:** the session that produced it worked in a long-lived
worktree branch that forked before this ADR's 2026-08-29 update and never merged `main`
again until closing out — it independently drafted a second "ADR-0052" file
(`0052-no-valid-eval-baseline-resolution.md`) documenting the same resolution below without
knowing this file already existed. Caught and reconciled by merging `main` before opening any
PR; the duplicate file is deleted, its content folded in here. Recorded plainly per this
project's own honest-documentation standard — a near-duplicate-ADR mistake is exactly the
kind of process gap worth naming, not quietly absorbing.

**No step could produce a trustworthy baseline on its own — each was a precondition for the
next:**

- **ADR-0054** found the model-selection basis itself (parse-success "44/44 vs 29/31") was
  never traceable to a committed artifact and didn't match the recovered raw data (20/20 vs
  19/20) — selection was re-grounded on truncation headroom instead.
- **ADR-0055** found the early-termination defect was substantially a wire-schema defect
  (asking the model to emit 7 fields nothing downstream consumes), not a model quality
  difference — fixed by stripping those fields from the wire schema (18 → 11 required).
- **ADR-0056** found the eval gold set's component labels were drawn from a broader taxonomy
  than the deployed classifier could ever emit — training-data staleness, not genuine
  rarity — making the existing "ungrounded" signal unreliable for a material fraction of
  vscode's eval population.
- **ADR-0057** shipped a retrain resolving the taxonomy gap, found the previously-reported
  "accuracy regression" didn't survive an apples-to-apples remeasurement (the old classifier
  is worse on a fair comparison, not better), and landed `declared_attribution` (ADR-0020)
  properly into the wire schema.
- **ADR-0058** proved the grounding/fabrication gate with a negative control (it had only
  ever been observed passing, never failing, until this), sized the vscode ratchet honestly
  given real observed sampling variance, and resolved the two residual gate failures found
  while closing this ADR (a stale `_RECORDED_ECE` constant, and `MANIFEST.sha256` drift
  pending the Phase 4 GCS publish — see that ADR for both).

**Baseline set from a clean, complete recording** (`eval/record_cassettes.py`, 34
quota-paced iterations over ~14 hours real time, `openai/gpt-oss-120b`, 47-class retrained
classifier, 12-required-field schema with `declared_attribution` restored,
`TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1`):

| Metric | vscode | k8s | Overall |
|---|---:|---:|---:|
| n | 11 | 53 | 64 |
| Judge mean (/15) | 12.4545 (83.0%) | 12.0189 (80.1%) | 12.0938 (80.6%) |
| Fabrication rate | 0.0% | 0.0% | 0.0% (0/64) |
| Floor-fail rate | 0.0% | 7.55% | — |
| Component-ungrounded | 0/11 | 0/53 | 0/64 |
| Fallback plans | 0 | 0 | 0/64 |
| Truncated completions | 0 | 0 | 0/64 |
| Permanently early-terminated | 0 | 0 | 0/64 |
| `declared_attribution` non-null | 11/11 | 53/53 | 64/64 (100%) |

**Recording quality: strictly better than the prior (invalid) attempt.** The previous
64-issue re-record (attribution off, pre-ADR-0057 schema) resolved 59/64 with 5 permanently
early-terminated. This recording resolved 64/64 with zero early terminations, zero fallback
plans, zero truncations.

Written to `reports/eval_baseline.json` (`eval/run_eval.py --update-baseline`). Not a
before/after regression check against the prior baseline — model, classifier, schema, and
prompt all changed simultaneously; there is no valid prior baseline this compares against
(that was this ADR's entire premise). This is a fresh floor, not a delta.

**`eval/test_invariants.py`'s `_GROUNDING_BASELINE` required NO numeric change.** Its
recorded values (0 ungrounded, n=53/n=11) already matched this recording's real result
exactly, and `eval_set_hash` is unchanged — only the provenance comment was stale, attributing
the constant to the now-invalid prior measurement.

**What becomes easier:** future model/prompt/schema changes have a real floor to compare
against, and the specific chain of unverified-precedent problems this ADR closes (parse-
success figures with no artifact, a taxonomy gap masquerading as fabrication, a stale
grounding ratchet silently failing since 2026-08-06) has a documented, one-time fix rather
than being rediscovered piecemeal again. See ADR-0058 for the gate-proof and ratchet-sizing
work that followed directly from this baseline, including the vscode statistical-power
finding (n≈73 needed for a 5%-ceiling hard gate) and the two residual gate fixes.

**What stays open:** everything ADR-0057 already listed as invalidated-but-not-yet-edited
(README claims, `docs/architecture/adr/0044`'s cited figures) — none of those are edited by
this update either, per the working agreement's explicit "do not edit these docs until
told."
