# ADR-0060 — Split synthesis from judging in the cassette recorder

Status: Accepted
Date: 2026-09-23 (split `ef83006`; status file `06e219a`; judge provenance + B4/B5 `ad7529f`;
B4 tests `7926687`; B5 link marking — this ADR's commit series)
Decider: Gaurav Gandhi

## Context

`eval/record_cassettes.py` originally did synthesis (Groq, `openai/gpt-oss-120b`) and judging
(local Ollama `qwen3:8b`, ADR-0019) per issue in one process. Synthesis is paced by Groq's
200k tokens-per-day rolling window, so a 64-issue recording spends most of its multi-day
wall-clock sleeping on quota. The judge sat resident the whole time: ~5 GB RAM and ~5.5 GB of
this machine's 8 GB VRAM held idle for a call that takes seconds. The 2026-09-22/23 recording
run was killed by the harness's system-RAM protection at 23/64 partly because of this.

## Decision

### Split into modes

`record_cassettes.py --mode {full,synthesis,judge}` (default `full` preserves the old
behaviour). `synthesis` never imports `TriageJudge` or touches Ollama; `judge` loads no
classifier/predictor/retriever and makes zero Groq calls. Both share `_synthesize_one()` /
`_judge_one()` with `full` so per-issue logic cannot diverge.

`scripts/run_recording_unattended.py --mode …` drives either loop: synthesis at
BelowNormal priority with BLAS threads capped at 2 (set in the child env before numpy import);
judge only when `nvidia-smi` shows ≥ 6 GiB free VRAM, re-checking rather than contending. The
judge pass unloads its model (`keep_alive=0`) when it ends. `RECORDING_STATUS.txt` is written
by `record_cassettes.py` after every issue (not per subprocess pass) and always reports both
synthesis-done and judged counts.

### Checkpoint semantics (Phase 2c)

A synthesis success is checkpointed with `plan` set and `judge_pending: true`.
`_pending_judge_entries()` and `_fully_judged_count()` are the single definition of
"synthesis-done" vs "fully done", shared by `run_judge()`, the status writer, the unattended
driver, and tests — a completion summary can never report N/64 while some are unjudged.

### B3 — Judge-entry provenance

A judge call has no classifier/index dependency of its own, so it is stamped (`judge_provenance`)
with what it actually depends on: the parent synthesis entry's cassette key, the judge model,
the rubric hash (`compute_judge_prompt_hash()`), and the parent's `artifact_hashes` **inherited**
via `get_provenance()`, never independently asserted. `test_cassette_provenance_matches_current_artifacts`
checks all four. The 23 judge entries recorded before this existed were deleted and their
checkpoint judge state reset — re-judged, not backfilled (backfilling provenance after the fact
is what ADR-0059 exists to stop).

### B4 — Judge-config invalidation via separate checkpoint fields (accepted deviation)

A judge-model or rubric change must invalidate judging. It is **not** folded into the
`(issue_id, model, prompt_hash, artifact_hash)` composite key: that key gates whether
*synthesis* is done, and coupling them would force a quota-burning full re-synthesis for a
change that never touched Groq. Instead each judged entry records `judge_model_used` and
`judge_prompt_hash_used`; `_judge_config_current()` gates every "judged" count and the judge
work queue on them matching the current config.

**Fail-closed requirement:** a score under a different judge model, a different rubric hash,
or with no stamp at all is re-queued and never counted as judged — in `_fully_judged_count`,
`write_live_status`, and the unattended driver's terminal check — and `run_judge` re-judges and
re-stamps it. Pinned by `tests/test_record_cassettes_checkpoint.py` (mutation-checked: forcing
`_judge_config_current` to `True` fails all three B4 tests).

### B5 — `synthesis_cache_key` backfill, marked as derived (accepted deviation)

The judge needs its parent synthesis key, but the checkpoint only started carrying
`synthesis_cache_key` in `ad7529f`. For the 48 entries synthesized before that, the key was
recovered by **unique content match** of each checkpoint plan against the cassette's synthesis
responses. This recovers an existing, deterministic fact (which cassette entry that call
produced) that a code gap failed to record — it is not provenance fabrication.

It is still derived, not observed, so every checkpoint entry carries
`link_source: "backfilled_content_match"` (those 48, identified as exactly the entries holding
a key at `ad7529f`) or `"recorded"` (written by the live synthesis call). Anything consuming the
link can distinguish the two.

## Consequences

- Ollama is resident only for the minutes the judge batch runs; VRAM free measured 2512 → 8020
  MiB after the explicit unload.
- Judging becomes a separate, repeatable, zero-Groq-cost step; a rubric change re-judges 64
  issues locally instead of re-synthesizing them.
- Two fields (`judge_model_used`, `judge_prompt_hash_used`) now carry correctness weight outside
  the composite key. Any new "judged" counter must go through `_judge_config_current()`; the
  tests cover the three that exist today, not future ones.
- The ideal was never-needing-backfill; B5 exists because the split shipped before the key was
  captured. `link_source` keeps that visible instead of letting 48 derived links read as
  observed.
- `--mode full` still exists and is untested against the RAM/VRAM concern by design.

## Alternatives considered

- **Keep one process, unload Ollama between calls.** Every judge call would pay a cold load and
  ADR-0019's cold-start variance, breaking cassette reproducibility.
- **Restart the shared Ollama server with global `OLLAMA_KEEP_ALIVE=0` /
  `OLLAMA_MAX_LOADED_MODELS=1`.** Disrupts other sessions using that server; a scoped unload or a
  dedicated server on another port achieves the same for this job only.
- **Judge config in the composite key.** Rejected above (B4).
- **Re-synthesize the 48 instead of backfilling (B5).** Correct but costs ~1.5 days of Groq
  quota (48 × ~6.4k tokens against a 200k/day window) to recover a fact already determinable
  from the cassette.

## Incident (2026-09-23): the B3/B5 cleanup wrote mojibake into 48 entries

The same `ad7529f` data cleanup that implemented B3/B5 rewrote `eval_cassette.json` and
`recording_checkpoint.json` after reading them as cp1252 rather than UTF-8. Every non-ASCII
character in the 48 entries then present (em-dash, arrow, non-breaking hyphen) was stored as
mojibake in requests, responses and checkpoint plans. Keys were untouched, so replay still hit,
while serving corrupted plan text. The 16 entries synthesized afterwards were clean. Found by
the prompt-parity test (the cassette held two distinct system prompts).

Repaired in `f1efca7` by the exact inverse transform, verified:
- byte-exact against `71bd580` for every entry that exists there;
- by zero-call regeneration of all 48 user prompts.

`test_cassette_and_checkpoint_have_no_mojibake` now guards the class. It returns 0 at `71bd580`
and 383/164 corrupted strings at `ad7529f`. B5's content match itself was unaffected: it
compared equally-corrupted copies. For any future checkpoint/cassette surgery: explicit
`encoding="utf-8"` on every read and write, and re-run the invariant suite before committing.
