from __future__ import annotations

import os
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

"""Stop a Groq recording pass before it spends the production serving budget.

The Groq free-tier TPD budget (200,000 tokens/day, org-wide, rolling 24 h) is shared by every
recording pass, other projects' benchmark keys and the production /triage endpoint. On
2026-10-07 a 53-call re-record spent all of it and the next production deploy's smoke test failed
with a 429 (DECISION_LOG D28). Owner rule: the recorder stops at 80 pct of the daily cap, which
keeps at least 30K tokens for deploy smoke tests.

Since 2026-10-12 this module is a thin adapter over the machine-wide ledger in
`tools/groq_budget/` (~/.groq_budget): per-model rolling-24 h accounting with per-project
attribution, a 40K production reserve that is never allocated, a 100K cap for the review-iq
`benchmark` project, the recorder taking the remainder, and a recording-active flag that makes
other projects wait while TriageIQ records. The public API below is unchanged. The first use
imports the old ~/.triageiq/groq_tpd_ledger.json history (idempotent; that file is left alone).

Why a ledger and not a probe: Groq exposes no "tokens remaining today" read, and a cheap probe
request does NOT reveal it. A first design asked for max_tokens=45000 and read the 429; on
2026-10-08 that probe returned OK at Used 198,924 of 200,000 (Groq only counts a bounded slice of
max_tokens against the daily budget), so it certified headroom that did not exist. Instead every
call is appended to a ledger shared by all worktrees and projects, and the gate sums the trailing
24 h. Consumers that do not call the ledger (production, CI smoke tests) live in the production
reserve. A real TPD 429 is ground truth: it is logged with its Used figure and the recorder waits
(record_cassettes.py / run_recording_unattended.py), and observe_tpd_429 folds any Used the
ledger did not know about into the ledger.

Mechanism, measured 2026-10-08 ~05:25Z: prompts of 1,399 and 4,279 tokens (max_tokens=16) were
ADMITTED with the ledger at ~199K, so the real Used had already fallen below the ledger's
even-spacing seed. A probe can only prove headroom for the tokens it actually sends (and spends
them when admitted), never for a 30K reserve. The ledger errs conservative (BELIEVED: the seed
overstates what is still inside the window), which only delays the recorder.
"""

DAILY_CAP_TOKENS = 200_000
# 80 pct of the cap is the most the recorder may spend in any rolling 24 h; the other 40K is for
# production and deploy smoke tests (the owner asked for >= 30K). Since 2026-10-12 this is the
# ALLOCATABLE pool of the shared ledger (cap minus the 40K production reserve), shared with other
# projects (the review-iq benchmark key is capped at 100K); the recorder takes the remainder.
RECORDER_CAP_TOKENS = 160_000
# Largest request one synthesis call makes (prompt + max_tokens), measured at ~4.9K on 2026-10-07.
NEXT_CALL_TOKENS = 5_000
WINDOW = timedelta(hours=24)

STOP_MARKER = "=== BUDGET RESERVE STOP ==="
LEDGER_ENV = "TRIAGEIQ_GROQ_LEDGER"  # optional: pin the ledger FILE (tests, one-off runs)
LEGACY_LEDGER = Path.home() / ".triageiq" / "groq_tpd_ledger.json"
MODEL = "openai/gpt-oss-120b"  # the model the recorder's synthesis calls use
PROJECT = "triageiq-recorder"

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "groq_budget"))
import groq_budget  # noqa: E402

_MIGRATED = False


class BudgetReserveError(RuntimeError):
    """Raised when this tooling has already spent its share of the rolling 24 h budget."""


def ledger_path() -> Path:
    override = os.environ.get(LEDGER_ENV)
    return Path(override) if override else groq_budget.root_dir() / "ledger.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _ledger(path: Path | None = None) -> groq_budget.Ledger:
    """The machine-wide ledger. The first default-location use imports the old per-TriageIQ
    history once (idempotent; the old file is untouched)."""
    global _MIGRATED
    led = groq_budget.Ledger(path=path or ledger_path())
    if not _MIGRATED and path is None and LEDGER_ENV not in os.environ:
        _MIGRATED = True
        led.migrate_legacy(LEGACY_LEDGER)
    return led


def record_usage(
    tokens: int,
    source: str,
    now: datetime | None = None,
    path: Path | None = None,
    cached_tokens: int | None = None,
) -> None:
    """Append one call's total tokens to the shared ledger (entries older than 48 h are dropped).

    `cached_tokens` (prompt tokens Groq served from its prompt cache, when reported) is stored
    beside `tokens` for the cached-token study. It does NOT change what the gate sums: whether
    Groq's TPD counts cached tokens is exactly what the study has to establish at the next TPD
    429 (compare Groq's 'Used' with the ledger's total and its non-cached total).
    """
    _ledger(path).record(PROJECT, MODEL, tokens, source, cached_tokens=cached_tokens, now=now)


def spent_last_24h(now: datetime | None = None, path: Path | None = None) -> int:
    """Trailing-24 h tokens on the recorder's model across ALL projects (the pool is shared)."""
    return _ledger(path).spent(MODEL, now=now)


def check_budget(now: datetime | None = None, path: Path | None = None) -> str:
    """Raise BudgetReserveError if one more call would push the shared 24 h pool past the cap."""
    led = _ledger(path)
    d = led.check(PROJECT, MODEL, NEXT_CALL_TOKENS, now=now)
    if d.allowed:
        spent = led.spent(MODEL, now=now)
        return f"budget ok: {spent} of {RECORDER_CAP_TOKENS} allocatable tokens used in the last 24 h"
    raise BudgetReserveError(d.detail)


parse_tpd_error = groq_budget.parse_tpd_error


def observe_tpd_429(text: str, now: datetime | None = None, path: Path | None = None) -> int:
    """Fold a real Groq TPD 429 into the ledger and return the tokens added (0 if not a TPD 429).

    The 429 body is the only ground truth for the org-wide Used figure. Spend this tooling did
    not make (production /triage, other projects) is invisible to the ledger; when Groq reports a
    Used larger than the ledger's trailing-24 h sum, the difference is recorded as
    `external-observed`, held as 24 hourly slices across the trailing window so it drains like real
    usage does (2026-10-12: one 24 h block had kept the ledger shut all day). A newer observation
    replaces the previous one.
    """
    return _ledger(path).observe_429(text, MODEL, now=now)


def start_recording(ttl_s: int = 6 * 3600, path: Path | None = None) -> None:
    """Set the recording-active flag: other projects' gpt-oss-120b checks wait until stop_recording()."""
    _ledger(path).start_recording(PROJECT, [MODEL], ttl_s=ttl_s)


def stop_recording(path: Path | None = None) -> None:
    _ledger(path).stop_recording(PROJECT)


# A multi-day recording outlives any fixed TTL (2026-10-30: the 6 h flag lapsed while the 150-set
# recording slept on the budget). The holder renews the flag well inside its TTL instead; a crash
# still clears it within one TTL, and sooner via the pid check.
HEARTBEAT_TTL_S = 2 * 3600
HEARTBEAT_INTERVAL_S = 15 * 60


def start_recording_heartbeat(
    interval_s: float = HEARTBEAT_INTERVAL_S,
    ttl_s: int = HEARTBEAT_TTL_S,
    path: Path | None = None,
) -> threading.Event:
    """Set the flag now and renew it every `interval_s` from a daemon thread; set the returned
    event to stop renewing. A failed renewal is printed and retried next interval, never raised:
    the recording itself must not die because the ledger file was briefly locked."""
    start_recording(ttl_s, path)
    stop = threading.Event()

    def _beat() -> None:
        while not stop.wait(interval_s):
            try:
                start_recording(ttl_s, path)
            except Exception as exc:  # noqa: BLE001 -- see docstring
                print(
                    f"recording-flag renewal failed (retrying in {interval_s}s): {exc}", flush=True
                )

    threading.Thread(target=_beat, name="recording-flag-heartbeat", daemon=True).start()
    return stop
