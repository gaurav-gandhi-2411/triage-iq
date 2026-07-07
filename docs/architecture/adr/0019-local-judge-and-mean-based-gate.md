# ADR-0019 — Local Judge (qwen3:8b) + Mean-Based Quality Gate

**Status:** Accepted
**Date:** 2026-07-07
**Decider:** Gaurav Gandhi

## Context

ADR-0018 disclosed that the n=60 gold set's judge scores were train-contaminated and needed
re-baselining on a verified-clean n=65 subset (54 kubernetes/kubernetes / 11 microsoft/vscode).
The org hosting the production judge (`llama-3.3-70b-versatile`, Groq) turned out to share a daily
token budget with an unrelated, autonomously-scheduled project (`agentgauge`) — a structural
collision discovered while trying to schedule the re-record. Paying for Groq's Dev tier was
considered and priced (~$0.09 one-time) but rejected in favor of a zero-cost local judge, since the
project's stated default is a $0-cost path for every LLM-dependent system.

This ADR covers two decisions made together because the second was only discoverable by attempting
the first: (1) moving the judge to a local Ollama model, and (2) redesigning the quality-regression
gate after direct experiments proved byte-identical re-recording is not achievable with any
inference stack available to this project.

## Decision 1 — Local judge: qwen3:8b (Ollama), GPU-only

### Model selection and determinism testing

`TriageJudge` never had Ollama routing before this ADR — despite qwen having been used previously
(ADR-0002), that was `qwen/qwen3-32b` via **Groq**, never local. Added `provider="ollama"` support
(`_ollama_completion`, `think=False`, `keep_alive=-1`, seeded `options`).

The largest available local model, `qwen3:30b-a3b` (18GB), does not fit this machine's 8GB GPU and
runs 67%/33% CPU/GPU split. Tested directly: two identical calls (same prompt, `temperature=0`,
`seed=42`) diverged at character 34 of the output — CPU/GPU floating-point non-associativity breaks
determinism for this model on this hardware. `qwen3:8b` (5.6GB) fits entirely in 8GB VRAM,
confirmed 100% GPU / 0% CPU. Verified byte-identical across 5 back-to-back calls, and quality
sample-checked on 3 real issues across both repos (numeric scores tracked actual
match/correctness correctly; free-text rationale is more formulaic than the old 70B's, confirmed
internal-only — never surfaced via `/eval` or the API — so this is an acceptable quality trade for
zero cost).

### GPU exclusivity finding (flagged, not fixed here)

Before recording, `ollama ps` showed an unrelated `llama3.1:8b` already loaded on the GPU (100%,
5.3GB) — not from this session's work, source unidentified. Stopped before recording. This is the
same "unattended process consuming a shared resource" pattern as the `agentgauge` Groq-org
collision, a second instance in the same session, different resource. **Systemic finding, not
addressed here:** something on this machine loads Ollama models unprompted; if it fires mid-run it
forces the CPU/GPU split that breaks determinism. Portfolio-wide GPU/API-key/budget isolation
across projects is a real infra-hygiene item for a separate session.

## Decision 2 — Byte-identical re-recording is not achievable; the gate changes accordingly

### What was tried and failed

**Groq synthesis (`llama-3.1-8b-instant`), with an explicit `seed=42`:** three live calls, identical
prompt, all three produced different output, diverging within the first 100 characters (a
`component_confidence: 0.44` vs `0.45` rounding difference cascading into different downstream
tokens). Groq's documented "best-effort, not guaranteed" caveat for `seed` is confirmed true in
practice. **`seed` was reverted from the cache-key computation** (kept only on the live API call,
harmlessly) after it broke replay of the already-recorded cassette — a real, caught side effect of
adding it.

**qwen3:8b judge, isolated back-to-back (5 calls, 2 previously-divergent issues):** 10/10
byte-identical. Confirmed the divergence is sequence-state contamination (something in how Ollama
serves many different sequential prompts against one loaded model), not intrinsic sampling
randomness.

**qwen3:8b judge, strict decoding (`top_k=1, top_p=0`) at full 65-issue scale, two full passes:**
59/65 (90.8%) byte-identical, 6/65 (9.2%) still diverged. Stricter decoding does not eliminate it.

**Conclusion: neither Groq nor the best available local model can reproduce a recording
byte-for-byte across independent runs, and no configuration tried fixes either one.** This is not a
bug to chase further — it's a hard property of the available inference stack. Byte-identical
re-recording was the wrong bar for a gate to depend on.

### What actually matters: are the MEANS stable?

Compared attempt1 (complete, n=65) against attempt2 (independent re-recording, stopped at n=57 by
an unrelated Groq synthesis TPD wall) on their 57 common issues:

| Repo | n | attempt1 mean | attempt2 mean | diff |
|---|---|---|---|---|
| kubernetes/kubernetes | 46 | 10.5000/15 | 10.4565/15 | 0.0435 |
| microsoft/vscode | 11 | 8.3636/15 | 8.3636/15 | 0.0000 |

Despite ~9-12% of individual issues jittering (measured: std=0.748 on the /15 total-score scale,
mean diff ≈0 — symmetric noise, no systematic bias, max single-issue swing of 3 points), **the
per-repo means are stable within hundredths of a point.** The jitter washes out at the aggregate
level. This is the reproducibility that actually matters for a baseline.

Also verified: `verify_plan_grounding()` (used by the grounding ratchet) is pure Python with no LLM
call — confirmed deterministic given a fixed plan (39/39 issues with byte-identical plans across the
two attempts also had byte-identical `grounding_status`, zero exceptions). Grounding's instability,
where it exists, traces entirely to the plan input changing (Groq synthesis nondeterminism), not to
any nondeterminism in the grounding logic itself.

### Gate redesign

**The cassette REPLAY invariant is unchanged and stays zero-tolerance.** `test_cassette_hash_matches_baseline`
still requires the committed cassette's bytes to match the baseline's hash exactly, and replaying
that fixed cassette is fully deterministic (verified: two consecutive `compute_scores()` calls in
one process, byte-identical, always). This was never broken — the confusion was conflating "replay
is exact" with "re-recording reproduces the same bytes," which is a different, unmet claim.

**What moves is the RE-RECORD-comparison semantics**, i.e., what happens when a *future* re-record
produces a *new* cassette and its scores need judging against the old baseline. `test_quality_regression.py`'s
`_check_repo_quality` now compares per-repo MEAN against baseline with a **per-repo tolerance band**,
derived directly from the measured jitter above, not guessed:

```
SEM = measured_std / sqrt(n)      (n = the actual comparison sample size: k8s 46, vscode 11)
band = 2 x SEM
```

| Repo | SEM | band |
|---|---|---|
| kubernetes/kubernetes | 0.1103 | **0.22** |
| microsoft/vscode | 0.2255 | **0.45** |

Full derivation is stored in `reports/eval_baseline.json["threshold"]`, not just this document —
`std_per_issue_total_score: 0.748`, the source description, and the exact formula, so the number is
auditable, not asserted.

**The check is one-directional**, unchanged in design from before: `drop = baseline_mean -
current_mean`; fails only if `drop > band`. A negative drop (an improvement) never trips it — this
was already true of the pre-existing pooled-threshold code and is preserved exactly, just with a
per-repo band instead of a pooled near-zero epsilon.

**vscode's band (0.45) is genuinely wider than k8s's (0.22) because n=11 makes its mean less
statistically stable — this is the honest signal of the vscode data-ceiling finding (ADR-0017), not
a fudged tolerance to make the gate pass.** Its regression sensitivity is real, coarser, and stated
as such in `eval_baseline.json["threshold"]["vscode_note"]`, not hidden.

**Grounding ratchet/pins get no tolerance band** — they're computed by cassette replay against the
already-committed, fixed plan (verified deterministic above), so the zero-tolerance replay invariant
already covers them correctly. Only a *future* re-record's grounding numbers would need the same
plan-level jitter treatment the judge got; the ratchet as implemented today does not, because it
never re-generates a plan.

## New baseline (clean n=65, local qwen3:8b judge)

| Repo | n | mean |
|---|---|---|
| kubernetes/kubernetes | 54 | 10.5185/15 |
| microsoft/vscode | 11 | 8.3636/15 |
| Overall | 65 | 10.1538/15 |

**This is a new baseline, not a corrected version of the old n=60/Groq-70B one.** Two things changed
at once — train-contamination removal (ADR-0018) and the judge model (this ADR) — and their effects
cannot be separated without re-running the old judge on the clean set, which would cost real Groq
spend for a comparison whose only purpose is intellectual curiosity. The delta between old and new
numbers should not be read as "how much contamination inflated the score" — it conflates two causes
and is explicitly not decomposed. `reports/eval_baseline.json["judge"]["note"]` states this.

Grounding baseline re-derived on the same clean n=65 cassette: k8s 1/54 ungrounded (issue #13057,
component axis), vscode 1/11 ungrounded (issue #311836, component axis). The old pins (#1678
similar-issue axis, #13435 component axis, both from the contaminated n=60 set) are gone: #1678
isn't in the clean n=65 set, and no similar-issue-axis hallucination exists in the current committed
cassette to pin. Both new pins are component-axis — not a weaker test by design, just what's
actually in the committed recording.

`eval/test_invariants.py::test_retrieval_top_k`'s hardcoded probe issues (`#2093, #4223` vscode,
`#11079` k8s) were also part of the contaminated original 60 and no longer exist in `eval_set.jsonl`
— replaced with `#311284, #311836` (vscode) and `#14054, #13257` (k8s), each verified directly at
5/5 exact top-5 match before committing.

## Consequences

- The judge is now zero-cost and reproducible-without-a-key, closing the org-sharing collision risk
  from ADR-0018's follow-up entirely — no live key means no daily-budget contention with any other
  project.
- The gate's real invariant (replay is exact) is unchanged and still zero-tolerance. What's now
  explicit, data-derived, and documented is the tolerance for the *different* claim of re-record
  reproducibility, which was previously implicit and effectively unmet (a near-zero epsilon that
  only "worked" because nobody had re-recorded from scratch to test it against).
- vscode's judge-mean sensitivity is now formally coarser than k8s's, stated plainly rather than
  papered over with a shared threshold.
- The free-text judge rationale is measurably more formulaic than the old 70B judge's — a real,
  disclosed quality trade for zero cost and full reproducibility of the numbers that are actually
  gated on.
- Systemic follow-up not resolved here: identify what loads Ollama models onto this machine's GPU
  unprompted (second instance of the "unattended process consuming a shared resource" pattern this
  session, after the `agentgauge` Groq-org collision) — separate session.

## Alternatives considered

| Alternative | Reason rejected |
|---|---|
| Pay for Groq Dev tier (~$0.09 one-time) to remove the TPD wall | Zero-cost is the project's stated default routing profile; the local path works and the cost isn't the blocker once the judge choice is settled — this ADR is about which judge, not about buying past a temporary wall. |
| Keep chasing byte-identical re-recording (different decode settings, different models) | Directly disproven: strict decoding didn't fix the residual 9% divergence, and Groq's replica-level nondeterminism isn't a client-side knob at all. Continuing would be optimizing for a bar that can't be met. |
| Weaken the replay gate itself to add slack | Rejected — replay is genuinely deterministic and verified so; weakening it would be fixing a problem that doesn't exist there while leaving the real gap (re-record tolerance) undocumented. |
| Pooled (not per-repo) tolerance band | Would let a vscode-concentrated regression hide under k8s's larger, more stable n — same reasoning ADR-0015's per-repo grounding ratchet was built on. |
| Multi-sample-and-median the judge to force exact reproducibility | Viable but not chosen: N x cost for a determinism guarantee the mean-based gate already provides at 1x cost, given the means are already proven stable. Left as a documented option if future evidence shows means are less stable than measured here. |
