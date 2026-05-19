# ADR-0002: Use qwen/qwen3-32b as the cross-family judge for eval validation

Status: Accepted
Date: 2026-05-18

## Context

The baseline eval (audit §F, `reports/triage_results.json`) uses `llama-3.3-70b-versatile`
as the LLM-as-judge. The production triage model is `llama-3.1-8b-instant`. Both are in the
**Llama family** (Meta), creating a potential same-family bias: a 70B Llama model may score
outputs from an 8B Llama model more charitably than a judge from a different architecture
would, inflating the reported 73% judge score.

To test this hypothesis, we need a cross-family judge that:
1. Is NOT in the Llama family.
2. Has free-tier rate limits sufficient to run 180 judge calls (60 issues × 3 systems)
   in a single session.
3. Produces reliable structured JSON output (the judge schema is a 6-field object with
   integer scores and a rationale string).
4. Uses the existing `GROQ_API_KEY` (no new credential required).

### Candidate evaluation (2026-05-18) — three iterations

**Attempt 1 — gemma2-9b-it (Groq)**

Selected initially: Google Gemma 2 9B, same GROQ_API_KEY, 15K TPM. Invalidated on the
first run — every call returned HTTP 400 `model_decommissioned`. Model removed from Groq
production before the workstream started. 0 scores collected.

**Attempt 2 — gemini-2.5-flash (Google AI Studio)**

Selected as replacement: Google family (cross-family from Llama), 250K TPM advertised.
Actual free tier: **20 RPD (requests per day)** for `gemini-2.5-flash`. Two test runs
exhausted the daily quota after 5 scored calls. 180 calls requires 9 days at this rate.
`gemini-1.5-flash` (the 1,500 RPD model) was also tried but is deprecated and removed
from the API as of 2026-05-18. Requires separate `GEMINI_API_KEY`.

**Attempt 3 — qwen/qwen3-32b (Groq)**

| Property | Value |
|---|---|
| Family | Qwen (Alibaba) — cross-family from Llama |
| Provider | Groq production |
| Key | Same `GROQ_API_KEY` |
| TPM (free) | 6,000 |
| RPD (free) | 1,000 |
| JSON reliability | Confirmed via smoke test |

For 180 calls at ~1,000 tokens/call = ~180,000 tokens:
- 6K TPM → 10s minimum between calls, ~30 min total.
- 1K RPD covers 180 calls in one session.
- Same Groq backoff infrastructure already proven stable in the main triage runs.

**TPD caveat (observed 2026-05-18):** In practice, Qwen3-32B in thinking mode emits
1,000–2,000 thinking tokens per call even at `temperature=0`. This multiplied actual
token consumption to ~3× the nominal estimate and exhausted Groq's daily token quota
after ~131/180 calls. Disabling thinking via `/no_think` suffix (appended to the last
user message) reduces per-call output significantly and is now applied automatically
in `_groq_completion` when `"qwen" in model.lower()`.

**Implication for scale:** Groq free-tier TPD makes this judge unsuitable for eval sets
larger than ~60 issues. As the gold set grows, a judge without a TPD-style ceiling will
be needed. Gemini 2.5 Flash (Google AI Studio) has 1,500 RPD with no documented TPD
ceiling and is the leading candidate for future evals — pending free-tier quota
verification (our 2026-05-18 test showed only 20 RPD for `gemini-2.5-flash`; the
1,500 RPD figure applies to `gemini-1.5-flash` which is now deprecated).

## Decision

Use **`qwen/qwen3-32b`** (Alibaba Qwen 3 32B, production tier on Groq) as the
cross-family judge. Set `--judge-delay 10` (6 calls/min → 6,000 TPM, at the cap).

The judge model is configurable via `--judge-model` CLI arg and `TRIAGE_JUDGE_MODEL`
env var in `scripts/11_evaluate_triage.py`. Each judge model gets its own checkpoint file
(`data/judge_scores_checkpoint_{model_slug}.jsonl`). Provider defaults to `groq`.

## Consequences

- **What changes:**
  - `src/triage_iq/evaluation/triage_eval.py`: `TriageJudge` also accepts
    `provider="gemini"` and `gemini_api_key` (added during Attempt 2). Default is
    `provider="groq"`, unchanged.
  - `scripts/11_evaluate_triage.py`: `--judge-model`, `--judge-provider`, `--output-file`
    args added. `_is_tpd_error` narrowed to Groq-specific keywords (no bare "429") so
    Gemini per-minute 429s don't trigger the TPD fast-exit path.
  - `google-genai>=1.0` added to `requirements.txt` (retained — may be useful for future
    Gemini evals if daily quota is upgraded).
- **What stays the same:** Default behavior (no CLI flags) is unchanged — still uses
  `llama-3.3-70b-versatile` via Groq.
- **What becomes easier:** Future judge swaps require only a CLI flag.

## Alternatives considered

- **gemma2-9b-it (Groq)** — originally selected; invalidated by model decommission.
- **gemini-2.5-flash (Google AI Studio)** — 20 RPD free tier exhausted after 5 calls;
  insufficient for 180-call eval. Requires separate `GEMINI_API_KEY`.
- **gemini-1.5-flash** — 1,500 RPD historically; deprecated and removed from API.
- **mistral-saba-24b (Groq)** — 6K TPM same as Qwen; weaker JSON reliability for
  structured scoring tasks; no quality advantage documented.
- **DeepSeek V3 (OpenRouter)** — requires `OPENROUTER_API_KEY` not provisioned.

---

## 2026-05-19 Update #1 — Groq TPD confirmed non-viable at n=60; Gemini 20 RPD wall confirmed

The qwen3-32b run completed only 132/180 judge calls (the `/no_think` fix added
in the original ADR reduced thinking-mode token burn, but the rolling TPD window
meant only 1 new score was added in a 24h period). **Groq TPD is definitively
non-viable for judge work at n=60.**

An attempt was made to pivot to `gemini-2.5-flash` via Google AI Studio free tier.
The API returned HTTP 429 with the following error after 6 calls total:

```
Quota exceeded for: generativelanguage.googleapis.com/generate_content_free_tier_requests
quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier
quotaValue: 20
```

**The 1,500 RPD figure cited in the original ADR is wrong for `gemini-2.5-flash`.**
It applies to `gemini-1.5-flash` (now deprecated and removed from the API). New
free-tier keys for `gemini-2.5-flash` have **20 RPD**. Upgrading to a paid Google
AI Studio project is required to reach 1,500 RPD. Historical archive:
`reports/_archive/2026-05-19-gemini-20rpd-confirmed.md`.

## 2026-05-19 Update #3 — W1.2 Cohere eval complete; cache eliminates re-cost concern

Full 180/180 judge calls (60 issues × 3 systems) with `command-a-03-2025` completed
for the W1.2 calibration eval.

| Run | Wall time | Cohere calls | Result |
|---|---|---|---|
| Warm-up (cold cache) | 1974.5s | 180 (live) | 10.83/15 |
| Verify (warm cache, W2.A) | 9.2s | 0 (cache hits) | 10.83/15 |

W1.2 vs W1.1 delta: +0.43 (+2.89pp). See ADR-0004 verdict section for per-dimension detail.

The W2.A LLM response cache (ADR-0005) retroactively eliminates the re-cost concern: future
Cohere re-runs against a warm cache cost 0 Cohere tokens. The 18% of monthly trial budget
consumed by the W1.2 warm-up is non-recurring for identical prompts.

**Llama-70b retrofit:** Pending Groq TPD reset. Will use warm triage cache so only judge
calls consume tokens (~180K, no triage tokens). Result will be appended to ADR-0004 verdict.

---

## 2026-05-19 Update #2 — Cohere Command A (Trial) is the working free-tier judge path

After both Groq and Gemini free tiers proved insufficient, **Cohere Command A
(`command-a-03-2025`)** via the Cohere Trial API completed all 180 judge calls
without rate-limit failures.

| Property | Value |
|---|---|
| Family | Cohere (cross-family from Llama) |
| Provider | Cohere API (`api.cohere.com/v2/chat`) |
| SDK | `cohere==6.1.0` (`ClientV2`, structured output via `response_format`) |
| Trial quota | 1,000 calls/month, 20 RPM |
| Judge delay | 6s/call (comfortably under 20 RPM) |
| 180 calls used | 18% of monthly budget |
| Structured output | `response_format={"type":"json_object","json_schema":...}` |
| Schema note | Cohere's validator rejects `minimum`/`maximum` on integer fields; stripped via `_cohere_sanitize_schema()` before passing |
| Run time | ~30 minutes, 0 failures |

Full results: `reports/triage_results_judge_cohere_command_a.json`.
Cross-family comparison: `reports/judge_comparison.md`.
Verdict: ADR-0003.
