# Resume note — TriageIQ, 2026-08-30 session

**If you are a fresh session with no memory of the prior conversation, read this file
first, then resume exactly as described below.** Production has been down since
~2026-08-16 (Groq retired `llama-3.1-8b-instant`); this session got as far as selecting
a replacement and is mid-way through re-recording the eval cassette against it.

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
