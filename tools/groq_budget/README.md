# groq_budget: one Groq token ledger for every project on this machine

Groq's free-tier daily pool (TPD) is a rolling 24 h window per model, shared by every key and
project in the organisation. Nothing on Groq's side says what is left, so every project that
calls Groq asks this ledger first and tells it afterwards. Standard library only; one file
(`groq_budget.py`); works from any project via a path or a copy.

2026-10-11 is why it exists: one benchmark key made about 2,100 calls (about 300K tokens on
gpt-oss-120b and 230K on gpt-oss-20b) while TriageIQ made a few dozen, and the production deploy
smoke test then failed on a 429. Each project kept its own private ledger; none could see the others.

## Allocations (gpt-oss-120b, defaults; other models: cap 200K, reserve 40K, no project caps)

| Slice | Tokens / rolling 24 h | Rule |
|---|---|---|
| Production reserve | 40,000 | Never allocated. Covers Cloud Run traffic and deploy smoke tests, which this ledger cannot see. |
| `benchmark` | at most 100,000 | Hard project cap. |
| `triageiq-recorder` | the remainder of the 160,000 allocatable | No fixed cap: allocatable minus what everyone else spent. |
| `external-observed` | whatever Groq reports | Added when a 429 shows a larger "Used" than the ledger knew. |

The check is two-sided: total spend on the model plus the next call must fit in the 160,000
allocatable, and a capped project's own spend plus the next call must fit its cap. Override per model
in `<root>/config.json`, for example
`{"openai/gpt-oss-120b": {"production_reserve": 60000, "projects": {"benchmark": 80000}}}`.

## Where it lives

`$GROQ_BUDGET_DIR`, else `~/.groq_budget/` (`ledger.json`, `recording.json`, `config.json`). Writers
take a lock file and replace the ledger atomically, so concurrent processes and worktrees are safe.
Entries carry `ts, model, project, tokens, source` and optionally `cached_tokens`.

## Integrating a project (Python)

```python
import sys
sys.path.insert(0, r"C:\Users\gaura\ml-projects\triage-iq\tools\groq_budget")  # or copy groq_budget.py
import groq_budget as gb

led = gb.Ledger()
MODEL, PROJECT = "openai/gpt-oss-120b", "benchmark"

d = led.wait_until_allowed(PROJECT, MODEL, est=3_000, timeout_s=3600)   # blocks while recording/exhausted
if not d.allowed:
    raise SystemExit(d.detail)                                          # d.reason says why, d.retry_after_s when

resp = client.chat.completions.create(model=MODEL, messages=msgs)       # your Groq call
u = resp.usage
cached = getattr(getattr(u, "prompt_tokens_details", None), "cached_tokens", None)
led.record(PROJECT, MODEL, u.prompt_tokens + u.completion_tokens, source="my_script:item-17", cached_tokens=cached)

# on a Groq 429:
led.observe_429(str(exc), MODEL)     # folds Groq's "Used" figure in; returns the tokens added
```

Record TOTAL tokens (prompt + completion). Count the call whether or not it hit Groq's prompt cache:
whether cached tokens count against the daily pool is still being measured (TriageIQ A3), so the
ledger errs conservative.

## Command line (any language)

```
python groq_budget.py check  --project benchmark --model openai/gpt-oss-120b --est 3000   # exit 0 allow, 4 wait
python groq_budget.py wait   --project benchmark --model openai/gpt-oss-120b --est 3000 --timeout 3600
python groq_budget.py record --project benchmark --model openai/gpt-oss-120b --tokens 2750 [--cached 1800] [--source x]
python groq_budget.py observe-429 --model openai/gpt-oss-120b --text "<429 message>"      # or pipe it on stdin
python groq_budget.py status --model openai/gpt-oss-120b
python groq_budget.py recording start|stop|show --project triageiq-recorder --model openai/gpt-oss-120b
python groq_budget.py migrate-legacy        # imports ~/.triageiq/groq_tpd_ledger.json; idempotent
```

## Recording-active flag

A project that re-records (TriageIQ cassettes) runs `recording start`; until `recording stop`,
other projects' `check`/`wait` for those models answer `WAIT [recording_active]`. The flag expires
after its TTL (default 6 h) and is ignored if its process has died, so a crash cannot block everyone.
Only the project that set it can clear it.

## Limits to know

- It cannot see spend that bypasses it (production, CI, a script that does not call it). The
  production reserve and `observe_429` are the safety net, not a substitute for integration.
- `observe_429` holds the gap between Groq's "Used" and the ledger as 24 hourly slices across the trailing window (the spend happened at unknown times in it), so it drains hour by hour; a newer observation replaces the previous one for that model and can correct it downwards. Before 2026-10-12 it was one block that blocked for 24 h.
- The `benchmark` key's own pacing under the 8,000 tokens-per-minute limit is the caller's job; this ledger only does the daily pool.
