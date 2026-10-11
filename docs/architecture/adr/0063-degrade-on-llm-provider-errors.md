# ADR-0063 — Degrade gracefully (HTTP 200, `_degraded`) on LLM provider-side failures

Status: Accepted (code on a PR; not merged, not deployed)
Date: 2026-10-08
Decider: Gaurav Gandhi

## Context

Production, 2026-10-07 20:16-20:18 UTC: Groq returned HTTP 429 (`groq.RateLimitError`, daily
token limit exhausted) and `POST /triage` returned HTTP 500 with no useful body. Traceback path:
`app.py triage` -> `triage_with_metadata` -> `_call_llm_verbose` -> `_groq_completion` ->
`groq/_base_client.py` raising `RateLimitError`.

`_call_llm_verbose` already degraded to the signals-only fallback plan for
`TruncatedCompletionError`, `SchemaValidationError` and the pre-call token-budget guard. Any
other provider error propagated to `app.py`'s generic `except Exception` and became a 500. The
Groq free-tier daily budget is shared with other projects, so exhaustion is a realistic
production event, not a corner case. The signals (classifier component, retrieved similar issues,
resolution estimate) are computed before the LLM call and are still correct when the LLM is
unreachable.

A second, compounding defect: `_groq_completion` retried `RateLimitError` six times with
5/10/20/40/60 s-class backoff (about 108 s of `time.sleep`) before raising. For a *daily* quota
that wait can never succeed, so the request held a worker for ~2 minutes and then failed anyway
(this is the 20:16-20:18 window in the logs).

## Decision

1. Provider-side failures that are not the caller's fault degrade to the existing signals-only
   fallback plan: HTTP 200, `_degraded: true`, `_llm_status: "degraded_provider_error"`,
   `_llm_status_reason` in `rate_limited_tpd | rate_limited_rpd | rate_limited_tpm |
   rate_limited_rpm | rate_limited | provider_timeout | provider_unavailable`.
   Covered: `RateLimitError` (429), `APIConnectionError`, `APITimeoutError`,
   `InternalServerError` and any other `APIStatusError` with status >= 500. Both
   `_groq_completion` call sites are covered (synthesis call and the parse-retry call).
2. The degrade-vs-raise split:
   - **Degrade** (provider's fault, transient, not actionable by the caller): 429, connection,
     timeout, 5xx.
   - **Keep raising** (HTTP 500 as before): `AuthenticationError` (401), `PermissionDeniedError`
     (403), and every other 4xx (400, 404, 413, 422). A rejected key must be loud: hiding it
     behind a plausible-looking plan would let a mis-rotated secret serve degraded answers
     indefinitely while every dashboard stays green. `/health?deps=1` and the deploy smoke test
     assert the key is accepted, and they only work as controls if the request path does not
     mask the same failure. 400/422 mean our request was malformed (our bug); degrading would
     convert a deterministic code defect into silent quality loss. (The existing
     `json_validate_failed` 400 and `response_format` 400 handling is unchanged.)
3. No new sleeping in the request path. TPD/RPD 429s now fail fast (no retry): a daily quota does
   not clear in seconds. Per-minute 429s, connection errors and 5xx keep the pre-existing
   in-method retry/backoff, unchanged (the offline recording scripts depend on it); see
   Consequences.
4. The provider's "try again in ..." text is logged at WARNING with repo, status code, error code
   and reason, truncated to 200 chars, in logs only. The response carries only the short reason
   code, never provider text.
5. A degraded result is never cached: the degrade returns before the `cache.set` call, in both
   the synthesis and parse-retry branches.
6. Metrics: `triage_requests_total{status="provider_degraded"}` (distinct from `fallback`, which
   keeps meaning "model output unusable"), a new
   `triage_llm_provider_errors_total{reason}`, and the existing `triage_llm_fallback_total`
   still increments. `_degraded` is true for this status.
7. The provider-outage fallback plan additionally lists the retrieved similar issues (real
   retrieval output, `relevance_note` says the LLM summary is unavailable). The older degrade
   paths keep `similar_issues=[]` so their behaviour and the eval are unchanged.
8. The deploy smoke test still asserts `_degraded is False`. A rate-limited candidate **must**
   still fail the gate; that is deliberate. Only its diagnostics changed: it prints the HTTP code
   and body on a non-200 (instead of `curl -f` exit 22) and prints `_llm_status_reason` on a
   degraded response.

UI: `triage-iq-ui` treats `_llm_status` as an opaque string (`src/components/UnderTheHood.tsx`
renders it with underscores replaced by spaces) and does not branch on specific values, so the new
value does not break rendering. The UI does not read `_degraded`/`_llm_status_reason`; a degraded
plan renders as a normal plan with the fallback summary text and status "degraded provider error".
Surfacing a banner is a separate UI change (out of scope here).

## Consequences

- Positive: a Groq outage or quota exhaustion yields a useful, honest answer (component, similar
  issues, resolution estimate) instead of a 500; alerts can distinguish provider outage from bad
  model output; daily-quota exhaustion no longer pins a worker for ~108 s.
- Negative / risk: degradation can hide an outage from naive "no 5xx" alerting. Mitigation: the
  `provider_degraded` status label and `triage_llm_provider_errors_total` exist so an alert can be
  written on them; `_degraded` stays true so clients can tell.
- Residual latency: per-minute 429s, connection errors and 5xx still back off up to ~108 s inside
  the request before degrading. Capping the request-path retry budget (while preserving the
  patient schedule for batch/recording callers) is a worthwhile follow-up; not done here to keep
  this change to one concern and the recording path untouched.
- Fallback summary text for this path still says "LLM response unparseable; estimate from
  predictor only." in `expected_resolution_summary` (pre-existing string shared by all fallback
  paths); the specific cause is in `triage_summary`/`priority_rationale` and `_llm_status_reason`.
- The eval cassette is unaffected: replay never reaches a provider call.

## Alternatives

- **Fall back to a second Groq model (e.g. `openai/gpt-oss-20b`, whose daily budget is separate).**
  Recommended as a follow-up, not implemented. It would keep full-quality-ish plans during a
  120b quota outage. It needs its own eval: the 20b model has different schema-adherence,
  grounding and priority behaviour, and the committed baseline (ADR-0061) was measured on the
  production model only. Shipping it unmeasured would be an unevaluated prompt/model change
  serving real users; it also needs its own cache-key and `_model` reporting semantics (the
  response says which model answered) and a decision on whether the smoke test should accept it.
- **Queue and retry later.** Requires async job infrastructure and a way to deliver the result;
  the endpoint is synchronous and interactive. Over-built for the need.
- **Retry with (longer) backoff in-request.** Already the pre-existing behaviour and the cause of
  the ~2-minute hang; useless against a daily quota.
- **Return 503 + Retry-After instead of 200.** Honest for machines, but the UI/clients get no
  plan at all although the signals are valid; and it reproduces "outage = error page" for users.
  200 with `_degraded: true` follows the contract already established by the other degrade paths.
- **Degrade on 401/403 too.** Rejected: see Decision 2.
