# ADR-0005 — App-side LLM Response Cache

**Status:** Accepted  
**Date:** 2026-05-19  
**Decider:** Gaurav Gandhi

---

## Context

Every `/triage` request and every eval script run calls Groq for each issue, even when
the exact same prompt was seen seconds or minutes ago. Two specific pain points drove this:

1. **Eval re-runs waste TPD.** The TriageIQ Groq account is shared between the triage
   model (`llama-3.1-8b-instant`) and the judge model (`llama-3.3-70b-versatile`). A
   60-issue eval run spends ~120K triage tokens before the judge can score anything.
   On the rolling ~100K TPD limit, the judge hits quota before it starts (see W1.2
   incident in the baseline audit).

2. **Wall-clock latency.** Identical inputs during development and demo loops take 1–3 s
   each when a cached response could be returned in <5 ms.

The fix must be additive — production paths must work identically with the cache disabled.

---

## Decision

Introduce an opt-in, SQLite-backed LLM response cache (`LLMCache`) keyed on SHA-256 of
a canonical JSON representation of each request.

**Key design choices:**

| Choice | Decision | Rationale |
|---|---|---|
| Storage backend | SQLite (WAL mode) | Zero new infrastructure, file-portable, survives process restarts, inspectable with standard tools |
| Cache key | SHA-256 of canonical `json.dumps(payload, sort_keys=True)` | Order-independent, deterministic, schema-versionable |
| Key payload | `schema_version`, `provider`, `model`, `messages`, `temperature`, `max_tokens` + provider-specific extras | All fields that affect the response must be in the key; anything that doesn't must stay out |
| Cohere extra field | `response_format` (sanitized JSON schema) | Cohere structured output depends on the schema; same messages + different schema → different response |
| Default | `TRIAGE_LLM_CACHE_ENABLED=false` | Opt-in — production behavior unchanged unless cache is explicitly enabled |
| Thread safety | `threading.Lock` guards all SQLite reads and writes | Python's `sqlite3` module is not thread-safe for shared connections; WAL alone is insufficient |
| Prometheus metrics | `triage_llm_cache_hits_total`, `triage_llm_cache_misses_total` (counters), `triage_llm_cache_size_bytes`, `triage_llm_cache_entries` (gauges refreshed on `/metrics` scrape) | Cache behaviour must be observable in the same Prometheus scrape as the rest of the API |
| Schema versioning | `schema_version: "v1"` field in key payload | Bump to invalidate all cached entries when the key format changes |
| Cloud Run | Cache is per-instance, ephemeral disk | Acceptable for Stage A — warm cache is a latency/TPD bonus, not a correctness requirement; cold starts are handled gracefully |

**Files added / modified:**

| File | Change |
|---|---|
| `src/triage_iq/cache/llm_cache.py` | New `LLMCache` class |
| `src/triage_iq/cache/__init__.py` | Module init |
| `src/triage_iq/config.py` | `llm_cache_enabled: bool = False`, `llm_cache_path: Path` |
| `src/triage_iq/models/triage.py` | `cache=` param on `TriageAssistant.__init__`; `_call_llm_verbose` returns 5-tuple `(plan, raw, usage, status, cache_hit)` |
| `src/triage_iq/evaluation/triage_eval.py` | `cache=` param on `TriageJudge.__init__`; `score()` checks/writes cache for all three providers |
| `src/triage_iq/api/app.py` | 4 new Prometheus metrics; cache init in `lifespan`; gauge refresh in `/metrics`; hit/miss counter in `/triage` |
| `src/triage_iq/api/loader.py` | `cache=` param forwarded to `TriageAssistant` |
| `scripts/11_evaluate_triage.py` | Cache init from env var; passed to `TriageAssistant` and `TriageJudge`; stats printed at end |
| `scripts/13_cache_admin.py` | Admin CLI: `stats`, `clear`, `clear-provider`, `clear-model` |
| `tests/test_llm_cache.py` | Unit tests (key determinism, miss/hit cycle, idempotent set, stats, clear ops, thread safety) |

---

## Validation results (T1 gate, 2026-05-19)

End-to-end validation against the W1.2 Cohere gold set (60 issues, 180 judge calls).

| Run | Cache state | Wall time | Cohere API calls | Judge score | Hit rate |
|---|---|---|---|---|---|
| Warm-up | Cold (empty DB) | 1974.5s | 180 (live) | 10.833/15 | — |
| Verify | Warm (180 entries) | **9.2s** (214×) | 0 (all cached) | 10.833/15 | **100%** |

Score drift across all 6 dimensions: **0.000** (identical JSON responses from cache).
Thread safety: 200 concurrent writes across 8 workers, 0.96s, 0 errors, 200 DB rows.
Key isolation: retry-path messages (6-turn list) produce a different SHA-256 than the
first-call messages (4-turn list) — no collision between primary and retry cache entries.

---

## Consequences

**Good:**
- Eval re-runs with the same triage checkpoint consume 0 new API tokens for plans already
  seen — reduces judge-side TPD burn from ~120K to near-zero on checkpoint-based reruns.
- `/triage` repeat requests (identical issue text) served from cache in <5 ms vs ~1-3 s live.
- Fully observable through existing Prometheus + Grafana setup.
- Zero production risk when disabled (default).

**Bad / watch:**
- SQLite on Cloud Run ephemeral disk means cache is cold on every instance start. For
  a high-traffic service a shared Redis layer would be necessary. Stage A accepts this.
- Cache hit rate depends on prompt determinism. Any change to `SYSTEM_PROMPT`,
  few-shot examples, or issue preprocessing invalidates all prior entries silently
  (key hash changes). This is correct behaviour but requires a manual `clear` after
  prompt changes in dev.
- `_call_llm_verbose` return type changed from 4-tuple to 5-tuple. Any code that unpacks
  this tuple must be updated (three tests in `test_api.py` were updated as part of this change).

**Not done (Stage B candidates):**
- TTL-based expiry (entries never expire in Stage A).
- Shared Redis cache across Cloud Run instances.
- Per-request cache bypass header.
