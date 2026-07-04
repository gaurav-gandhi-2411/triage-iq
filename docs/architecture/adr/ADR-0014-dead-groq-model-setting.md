# ADR-0014: Wire or delete `groq_model_triage` config setting

Status: **Accepted — Option B implemented**
Date: 2026-06-21 (decided), 2026-07-04 (implemented)

## Problem

`config.py` defines `groq_model_triage: str = "llama-3.1-8b-instant"` (readable via env var
`GROQ_MODEL_TRIAGE`), but `loader.py` never reads it. `TriageAssistant` is constructed without a
`model=` kwarg, so it falls through to its own hardcoded default (`"llama-3.1-8b-instant"`).

The env var `GROQ_MODEL_TRIAGE` set on Cloud Run has zero effect. Discovered 2026-06-21 while
tracing the synthesis model chain during eval-gate cassette work.

**Why this is a real trap, not just a clean-up note:** Today it is harmless only because the dead
config default equals the hardcoded one (`"llama-3.1-8b-instant"` in both places). A future
operator who adds `GROQ_MODEL_TRIAGE=llama-3.3-70b-versatile` to Cloud Run env expecting a model
upgrade would see no error, no warning, and no change — prod keeps serving 8B while every config
inspection implies 70B. Same failure class as the model-artifact drift that motivated ADR-0013
(config says X, prod serves Y, no alarm), but at the wiring layer rather than the artifact layer.
The MANIFEST drift guard does not catch this.

## Options

**Option A — Wire it:** Pass `cfg.groq_model_triage` from `app.py` through `ModelStore.load_all`
to `TriageAssistant(model=...)`. The env var then actually controls the synthesis model.

- Con: model is not a safe runtime variable. Changing synthesis model requires re-recording
  eval cassettes and re-approving baseline means (`--update-baseline`). A hot env-var swap
  would silently invalidate the cassette (all keys derive from model name — CI would hit
  `CassetteMissError` on the next push). This makes `GROQ_MODEL_TRIAGE` a footgun that looks
  like an operational lever but breaks the eval gate.

**Option B — Delete it (recommended):** Remove `groq_model_triage` from `config.py`. The model
stays as a code constant in `triage.py`. To change synthesis model: edit `triage.py`, re-record
cassette locally, run eval, approve means, commit all three in one PR.

- Pro: no inert env var, no misleading control surface, no footgun.
- Pro: model changes require the same workflow as any other eval-affecting change — explicit,
  reviewed, cassette+baseline updated atomically.
- Con: can't switch models without a code change + redeploy. Acceptable given current usage.

## Decision

**Accepted.** Option B (delete) implemented 2026-07-04, after the eval-gate PR (#9) merged.

## Implementation (Option B) — done

1. Deleted `groq_model_triage` from `config.py` (and its `GROQ_MODEL_TRIAGE` reference in
   `README.md`'s environment-variable table).
2. No change to `loader.py` or `triage.py` — the hardcoded default stays as-is.
3. This ADR updated to `Status: Accepted`.
4. Re-verified `GROQ_MODEL_TRIAGE` not set in Cloud Run env immediately before implementing
   (2026-07-04, via `gcloud run services describe`) — confirmed absent.
