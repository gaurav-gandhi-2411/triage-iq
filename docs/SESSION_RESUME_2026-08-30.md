# Resume note — TriageIQ, 2026-08-30 session

**If you are a fresh session with no memory of the prior conversation, read this file
first, then resume exactly as described below.** Production has been down since
~2026-08-16 (Groq retired `llama-3.1-8b-instant`); this session got as far as selecting
a replacement and is mid-way through re-recording the eval cassette against it.

## 2026-09-03 (final): 9/9 confirmed — READY to start the full re-record, NOT started

`k8s-14835` (the last unconfirmed issue) succeeded on retry — genuinely live,
`cache_hit=False`. **All 9 original early-termination issues are now confirmed under
the reduced schema:** 8/8 clean, 1 (`vscode-4993`) at 1 fail / 2 live attempts (the
rare, non-reproducible malformed-key defect diagnosed in ADR-0055 Part A — not a
blocker per A5, carried forward as a residual rate).

**Parts A and B are both complete. READY to start the full 64-issue re-record.**
Per the explicit working agreement (D4/B3), this session reports readiness and STOPS
— does not start it. Next session/explicit approval: run
`scripts/run_recording_unattended.py` (already built and tested, currently paused).
Expect 3-4 days of quota-paced wall-clock per the D2 estimate in this session's report
— today's quota is exhausted (confirm via a probe call before assuming it has reset).

## 2026-09-03 (still later same day): 8/9 confirmed, 1 pending, do not start re-record yet

**vscode-4993's new failure mode diagnosed (ADR-0055 Part A):** malformed key
(`"triage_summary layman"` instead of `triage_summary`). Confirmed via direct
`jsonschema` validation against our own schema (installed into `.venv` for this check
only, not a new project dependency) that our schema is valid and correctly rejects
this payload for TWO reasons (missing required key + disallowed additional
property) — Groq's `strict: true` constrained decoding is not fully constraining
property *names* at the token level (a Groq-side limit), not a bug in our schema or
parser. **Not reproducible**: retried `vscode-4993` itself (succeeded) plus 2 other
vscode issues (both succeeded) — 1 fail in 2 live attempts on the same issue, looks
like a rare stochastic decoding glitch, not systematic. Not special-cased —
`record_cassettes.py` already fails closed on any schema-validation failure
regardless of cause.

**Validation status: 8/9 of the original early-termination issues confirmed
genuinely live under the reduced schema** (`k8s-12224`, `vscode-4996`, `k8s-12477`,
`k8s-12248`, `k8s-13508`, `k8s-12287`, `k8s-12254` all succeed;`vscode-4993` succeeds
1/2 with the new rare failure mode above). **`k8s-14835` is the ONLY one still
unconfirmed** — blocked by Groq TPD exhaustion three separate times today (429s, not
schema errors), most recently `Used 199737/200000`. A final background retry is
queued for it alone; check `eval/cassettes/unattended_logs/`-adjacent scratch output
or just re-run `scripts/scratch/validate_9_early_terminations.py` (TARGET_IDS already
set to just `k8s-14835`) once quota allows.

**Do NOT start the full 64-issue re-record until `k8s-14835` is confirmed** (per the
working agreement's explicit B3 gate — report READY only after A and B are both
complete). If it succeeds: 9/9 effectively resolved (the one new failure mode is a
rare residual rate, not a blocker per A5). If it fails with a genuine schema error
(not TPD): report immediately, do not proceed.

## 2026-09-03 (later same day): schema fix implemented, live-validated, quota exhausted mid-validation

**Model selection is now settled** (GG approved, ADR-0054): `openai/gpt-oss-120b` on
truncation headroom. **Schema fix implemented and mostly validated** (ADR-0055):
`_strip_post_hoc_fields` (`src/triage_iq/models/triage.py`) removes 7 fields from the
wire schema, 18 → 11 required. Checkpoint `prompt_hash` now covers the schema too
(`eval/record_cassettes.py`) — confirmed this correctly invalidates the 64 entries
recorded under the old schema (hash `5ddbe97c...` → `0a15adc0...`; a fresh resume
would need to re-record all 64 from scratch under the new schema).

**Do NOT start the full 64-issue re-record yet.** Live validation (cache bypassed to
force genuinely new calls) found:
- **All 5 real `gpt-oss-120b` re-record failures now succeed** (`k8s-12224`,
  `vscode-4996`, `k8s-12477`, `k8s-12248`, `k8s-13508`) — confirmed via genuinely new
  cassette entries, not stale replays.
- **`vscode-4993` still fails — a NEW, different failure mode** (a malformed field
  name, `"triage_summary layman"` instead of `triage_summary`, not a missing field).
  Reproduced twice independently. Not fixed by this change.
- **3 issues unconfirmed** (`k8s-12287`, `k8s-14835`, `k8s-12254`) — Groq's daily
  quota was exhausted mid-validation (`TPD: Limit 200000, Used 197689+`,
  2026-09-03 ~12:08). Their earlier apparent "success" was a stale cache replay under
  the OLD schema (the cassette's own cache key also doesn't cover `response_format` —
  a separate, real gap, not yet fixed, see ADR-0055), not a genuine test.

**To resume validation once quota resets** (a fresh day, or check Groq's stated wait):
`scripts/scratch/validate_9_early_terminations.py` (gitignored, scratch — re-create
from this session's transcript if needed, or write a fresh one) with `cache=None` to
force genuinely live calls. Confirm the 3 remaining issues, decide what to do about
`vscode-4993`'s new failure mode (accept as a residual rate and proceed, or
investigate further), THEN start the real 64-issue re-record via
`scripts/run_recording_unattended.py` (already built, paused — should work unchanged
against the new schema/hash).

**GROQ_API_KEY note:** this worktree has no `.env` — load it from the main repo
(`C:\Users\gaura\ml-projects\triage-iq\.env`) before any live script, e.g.:
`export GROQ_API_KEY=$(grep "^GROQ_API_KEY=" /c/Users/gaura/ml-projects/triage-iq/.env | cut -d= -f2-)`

## 2026-09-03: RECORDING PAUSED — model selection under revision, do not resume

**The unattended recorder is stopped** (it self-halted on its own after reaching 64/64
accounted for: 59 resolved, 5 permanently early-terminated — see below — no process
running, checkpoint file verified valid). **Do not restart
`scripts/run_recording_unattended.py` or `eval/record_cassettes.py` until GG approves
the revised model selection** in ADR-0054's 2026-09-03 correction section
(`docs/architecture/adr/0054-model-selection-underpowered-judge-mean.md`). The
original "44/44 vs 29/31" parse-success basis for selecting `gpt-oss-120b` was
retracted — untraceable to any committed artifact, and the raw records that were
eventually recovered (a prior session's scratchpad,
`.../0b8faa64-7bcf-4ede-8dbe-a941bbcb6980/scratchpad/part_d_screen_results.jsonl`) show
20/20 vs 19/20, not 44/44 vs 29/31. Full accounting, corrected statistics, and a
proposed revised basis (completion-token distribution / ceiling headroom, not
parse-success) are in the ADR.

**Checkpoint validity if `gpt-oss-120b` is re-confirmed as-is:** the 59 resolved + 5
dead entries in `eval/cassettes/recording_checkpoint.json` are keyed by
`(issue_id, model, prompt_hash)` — confirmed still `model=openai/gpt-oss-120b`,
`prompt_hash=5ddbe97c4b47958f`. If the revised selection keeps this model and prompt
unchanged, all 59 resolved entries remain valid on resume; nothing would need
re-recording. **Caveat found this session:** `prompt_hash` covers only the system
prompt and few-shot text, NOT the JSON schema (`response_format`) sent to Groq — if
Priority 3's proposed schema reduction (see ADR-0054) is ever implemented, the
checkpoint would NOT detect that change on its own. Fix the hash or clear the
checkpoint explicitly alongside any future schema edit.

## Where things stand

- **Worktree:** `C:\Users\gaura\ml-projects\triage-iq-wt-bakeoff`, branch
  `chore/bakeoff-screen-2026-08-29`. All work this session is committed **locally on
  this branch only** — nothing pushed, nothing merged to `main`.
- **Model selected:** `openai/gpt-oss-120b` (few-shot prompt), per **ADR-0054**
  (`docs/architecture/adr/0054-model-selection-underpowered-judge-mean.md`, Status:
  Accepted). `src/triage_iq/model_config.py`'s `TRIAGE_MODEL` already updated.
- **In progress:** re-recording `eval/cassettes/eval_cassette.json` (64 issues) against
  the new model via `eval/record_cassettes.py`. Both the cassette and
  `eval/cassettes/recording_checkpoint.json` were deliberately cleared this session (the
  old files held the retired model's data; old data is fully recoverable via git history
  if ever needed — see commit `c3693ef`).

## How to resume the re-record

```
cd "C:\Users\gaura\ml-projects\triage-iq-wt-bakeoff"
# load GROQ_API_KEY from the main repo's .env first, e.g.:
#   cd "C:\Users\gaura\ml-projects\triage-iq" && set -a && source .env && set +a && cd - 
.venv/Scripts/python.exe eval/record_cassettes.py
```

This is **fully resumable** — it reads `recording_checkpoint.json` and skips whatever's
already done. Just re-run the same command; no flags needed.

**Expect it to fail repeatedly with a TPD (tokens-per-day) rate limit.** Groq's error
message includes an exact `Please try again in Xm Ys` — wait that long (plus ~1 min
buffer) and re-run the same command. This is a **rolling** window, not a fixed daily
reset (confirmed directly this session: the `Used` figure decreases gradually over
tens of minutes, it does not reset all at once). Observed pace this session: quota was
already ~98% consumed by earlier work before the re-record even started, so each resume
attempt has only gotten 0-3 issues through before hitting the wall again. **This will
likely take hours to ~1-2 days of wall-clock, mostly spent waiting on quota, not
compute.** Closing the machine between attempts is fine — nothing is lost, just resume
the same command later.

**As of this note:** 4/64 issues checkpointed (`k8s-13257`, `k8s-14054`, `k8s-13096` —
genuine successes; `k8s-12224` — a real early-termination failure, `plan=None`, see
below). A background wait-then-retry was in flight when this note was written; check
`recording_checkpoint.json`'s `done` count to see if it made further progress before the
session ended.

## 2026-09-03 update: reconciliation findings + unattended recorder running

**Checkpoint-keying fix is now committed** (`d2b2b96`, includes the 36/64 issues recorded
under it so far). The fix works as designed: the **main repo checkout**
(`C:\Users\gaura\ml-projects\triage-iq`)'s `eval/cassettes/recording_checkpoint.json`
still holds 64 untagged (pre-fix) entries — verified directly (`model`/`prompt_hash` keys
absent on every entry). Any `record_cassettes.py` run started from that checkout will
hit `load_checkpoint()`'s `sys.exit(1)` ("STOP: ... predates the (issue_id, model,
prompt_hash) keying fix") immediately. **That is the fix working, not a bug** — do not
delete or re-tag that file to make it pass; it correctly cannot be trusted to belong to
`openai/gpt-oss-120b`.

**Early-termination reconciliation against ADR-0054 (Priority 1) — reported, not
resolved unilaterally, per the working agreement:**
- 3/36 record-time early terminations so far: `k8s-12224`, `vscode-4996`, `k8s-12477`
  (all `plan=None`, schema-incomplete JSON — same defect class as ADR-0053).
- Sample membership (VERIFIED against `docs/eval/bakeoff_prereg_2026-08-29.md` §6, read
  from the `docs/bakeoff-prereg` branch — that file is not on this branch's history):
  **`vscode-4996` IS one of the screen's 20 pre-registered issues** (repo=vscode,
  number=4996, row present in §6's table). **`k8s-12224` and `k8s-12477` are NOT** in
  that sample — easy to mis-scan against `k8s-12254`, which is a different, in-sample
  issue. So the screen's sample was not representative of 2 of these 3 failures by
  construction (never tested), but it *did* include `vscode-4996`.
- **Request shape unchanged between screen and record** (VERIFIED via `git log`):
  `_PROMPT_SIZE_SAFETY_MARGIN=200` and the schema-field-description fixes landed at
  `1564293` (2026-08-30 00:11), *before* the v3 harness commit (`acc2d7d`, 09:36) that
  produced the screen's Arm C numbers — both used the current, fixed request shape.
  Nothing under `src/triage_iq/` changed between ADR-0054's acceptance (`99c9799`,
  13:48) and now, other than the checkpoint-keying fix itself (bookkeeping only, no
  request-shape effect).
- **ADR-0054's "44/44 (100%)" figure has no committed artifact backing it** — searched
  every branch for a results JSON/log; none exists. The `docs/bakeoff-prereg` branch's
  own append-only log stops at 10:53 (`479b022`) recording Arm C at **19/20, 1 issue
  still missing**, with D7's explicit note that further retries that day "would just
  fail identically." ADR-0054 (`685934e`, 12:41) then reports 44/44 complete ~1h48m
  later with no intervening commit recording how the gap closed. **BELIEVED, not
  VERIFIED** — flagging per rule 65b, not asserting the number is wrong, only that it
  isn't traceable to a committed artifact the way this engagement's own working
  agreement calls for.
- **Pooled rate, taking ADR-0054's 44/44 at face value:** gpt-oss-120b 3/80 = 3.75%
  (95% Wald CI [0.00%, 7.91%]) vs gpt-oss-20b (Arm A, screen only, per ADR) 2/31 = 6.45%
  (95% CI [0.00%, 15.10%]) — **the intervals overlap almost completely; not resolved.**
  Taking the record-only rate alone (no benefit of the unverified screen figure):
  gpt-oss-120b is 3/36 = 8.33%, *higher* than gpt-oss-20b's screen-measured 6.45%.
  **Read: the parse-success/early-termination gap ADR-0054 called "categorically
  resolved" does not survive the record data — at these sample sizes the two models
  are not distinguishable on this metric either, the same underpowered-comparison shape
  the ADR itself used to disqualify judge mean.** Not re-opening the model selection —
  reporting this for GG to decide, per the working agreement.

**Unattended recorder running** (Priority 2): `scripts/run_recording_unattended.py`,
launched as a detached background process (verified orphaned from its launcher — will
survive this CC session or terminal closing; **not** verified to survive the machine
sleeping or shutting down — `powercfg` on this machine exposes no lid-close-action
setting to check, state this as unconfirmed rather than assumed). It re-invokes
`record_cassettes.py`, waits out Groq's stated rate-limit window (parsed from the error
text, +60s buffer, falling back to 30 min if unparseable) on a TPD/rate-limit hit, and
hard-stops (no retry) on a degraded/fallback synthesis, a truncated completion, a
checkpoint/model mismatch, or any subprocess outcome it doesn't recognize — fail-closed,
per this engagement's own guard-design rule. It computes "done" from the checkpoint file
directly (resolved + permanently-dead == 64), not from the subprocess's exit code, since
`record_cassettes.py` itself exits 1 forever once the only issues left are the
permanently-skipped dead ones — reading that literally would be an infinite retry trap.
Status file: `eval/cassettes/RECORDING_STATUS.txt` (plain text, human-readable, updated
every iteration — check it directly, no CC session needed). Per-iteration logs:
`eval/cassettes/unattended_logs/`. PID file: `eval/cassettes/unattended_recorder.pid`.

To stop it manually: `Stop-Process -Id (Get-Content eval/cassettes/unattended_recorder.pid)`.

## A known issue you'll need to decide on: `k8s-12224`

This issue got a genuine early-termination failure from gpt-oss-120b (schema-incomplete
JSON, same defect class documented in ADR-0053) during live recording — NOT a bug, real
model behavior. `record_cassettes.py` checkpointed it as done with `plan=None` (no
`tpd_hit` flag), which means **the script will never retry it automatically** — it's
permanently skipped on every future resume. The script's own completeness assertion at
the very end (after all 64 are attempted) will fail loudly on this
(`sys.exit(1)`, "INCOMPLETE: 1/64 issues missing a judge score") — that's expected, not
a new bug.

Decide before declaring the recording complete:
1. **Retry just this one issue** — delete its entry from
   `recording_checkpoint.json`'s `done` dict, then re-run; temperature=0/seed=42 usually
   reproduces the same failure, but LLM serving isn't always perfectly deterministic
   across server instances, so a retry might succeed.
2. **Accept it as a genuine, reported data point** — per this session's working
   agreement ("any non-zero fallback rate fails the gate you built — report it, do not
   work around it"), silently retrying-until-it-passes would defeat the point. If it's
   still failing after one clean retry attempt, report the real early-termination rate
   (currently 1/64 so far) rather than engineering around it.

Either way, **report the decision and the reasoning, don't silently pick one.**

## What happens after the recording completes

Per the working agreement already established this session (do not deviate without
telling the user):

1. Report the fallback-plan rate, early-termination rate, and truncation rate in the
   full 64-issue recording (this is Part C4 of the session's task list).
2. Run the full eval/judge pipeline, establish `reports/eval_baseline.json`, and set
   `_GROUNDING_BASELINE` in `eval/test_invariants.py` from the real measurement — this
   is the project's first genuinely valid baseline (see ADR-0052).
3. Close ADR-0052 against the actual baseline.
4. Report which README/`docs/PROJECT_STATE.md` claims can be replaced with measured
   numbers vs. remain permanently unreproducible (measured on retired models). **Do not
   edit those docs until told to.**
5. Report the full merge-and-deploy sequence (which PRs, what order, CI status per PR)
   and **wait for explicit approval before merging anything.**

**Zero spend, Secret Manager and IAM untouched — these constraints hold for the rest of
this work, not just this session.**

## Everything already committed this session (for context, not action needed)

On `chore/bakeoff-screen-2026-08-29`, roughly in order: cassette `allow_record` guard
(prevents record-on-miss outside the sanctioned path), token-budget margin 100→200 +
6 schema-field description fixes (a live 400 was caused by a percent-vs-fraction units
bug in a field description), bake-off screen harness v3 with a mandatory zero-quota
dry-run gate (`scripts/bakeoff_screen_harness.py`), ADR-0053 (few-shot retained on
evidence), ADR-0054 (model selection), the `TRIAGE_MODEL` update, two more fixes to
`eval/record_cassettes.py` found by a dry-run gate applied to it specifically
(`scripts/record_cassettes_dry_run_check.py`) before any of this session's real spend on
the re-record: (a) a degraded/truncated completion no longer gets silently checkpointed
as done, (b) the synthesis stop-check now covers sustained rate limits generally, not
just TPD-worded messages. Full detail is in the conversation this session had, and in
each commit's own message — `git log --oneline` on this branch tells the story in order.
