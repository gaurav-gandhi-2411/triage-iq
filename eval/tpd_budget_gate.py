from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

"""Stop a Groq recording pass before it spends the production serving budget.

The Groq free-tier TPD budget (200,000 tokens/day, org-wide, rolling 24 h) is shared by every
recording pass, gg-portfolio and the production /triage endpoint. On 2026-10-07 a 53-call
re-record spent all of it and the next production deploy's smoke test failed with a 429
(DECISION_LOG D28). Owner rule: the recorder stops at 80 pct of the daily cap, which keeps at
least 30K tokens for deploy smoke tests.

Why a ledger and not a probe: Groq exposes no "tokens remaining today" read, and a cheap probe
request does NOT reveal it. A first design asked for max_tokens=45000 and read the 429; on
2026-10-08 that probe returned OK at Used 198,924 of 200,000 (Groq only counts a bounded slice of
max_tokens against the daily budget), so it certified headroom that did not exist. Instead every
call this tooling makes is appended to a ledger shared by all worktrees, and the gate sums the
trailing 24 h. Consumers outside this tooling (production, gg-portfolio) are not in the ledger;
they live in the 20 pct the cap leaves untouched. A real TPD 429 is ground truth: it is logged
with its Used figure and the recorder waits (record_cassettes.py / run_recording_unattended.py).
"""

DAILY_CAP_TOKENS = 200_000
# 80 pct of the cap is the most the recorder may spend in any rolling 24 h; the other 40K is for
# production and deploy smoke tests (the owner asked for >= 30K).
RECORDER_CAP_TOKENS = 160_000
# Largest request one synthesis call makes (prompt + max_tokens), measured at ~4.9K on 2026-10-07.
NEXT_CALL_TOKENS = 5_000
WINDOW = timedelta(hours=24)

STOP_MARKER = "=== BUDGET RESERVE STOP ==="
LEDGER_ENV = "TRIAGEIQ_GROQ_LEDGER"


class BudgetReserveError(RuntimeError):
    """Raised when this tooling has already spent its share of the rolling 24 h budget."""


def ledger_path() -> Path:
    override = os.environ.get(LEDGER_ENV)
    return Path(override) if override else Path.home() / ".triageiq" / "groq_tpd_ledger.json"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _read(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))["entries"]


def _write(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"entries": entries}, indent=1) + "\n", encoding="utf-8")


def record_usage(tokens: int, source: str, now: datetime | None = None, path: Path | None = None) -> None:
    """Append one call's total tokens to the shared ledger (entries older than 48 h are dropped)."""
    p = path or ledger_path()
    t = now or _now()
    keep = [e for e in _read(p) if datetime.fromisoformat(e["ts"]) > t - 2 * WINDOW]
    keep.append({"ts": t.isoformat(), "tokens": int(tokens), "source": source})
    _write(p, keep)


def spent_last_24h(now: datetime | None = None, path: Path | None = None) -> int:
    t = now or _now()
    return sum(
        e["tokens"]
        for e in _read(path or ledger_path())
        if datetime.fromisoformat(e["ts"]) > t - WINDOW
    )


def check_budget(now: datetime | None = None, path: Path | None = None) -> str:
    """Raise BudgetReserveError if one more call would push the 24 h ledger past the cap."""
    t = now or _now()
    p = path or ledger_path()
    spent = spent_last_24h(t, p)
    if spent + NEXT_CALL_TOKENS <= RECORDER_CAP_TOKENS:
        return f"budget ok: {spent} of {RECORDER_CAP_TOKENS} recorder tokens used in the last 24 h"
    # When enough old spend expires for one more call, for the message only.
    need = spent + NEXT_CALL_TOKENS - RECORDER_CAP_TOKENS
    freed, free_at = 0, None
    for e in sorted(_read(p), key=lambda e: e["ts"]):
        ts = datetime.fromisoformat(e["ts"])
        if ts > t - WINDOW:
            freed += e["tokens"]
            if freed >= need:
                free_at = ts + WINDOW
                break
    when = free_at.strftime("%Y-%m-%d %H:%M UTC") if free_at else "unknown"
    raise BudgetReserveError(
        f"{STOP_MARKER}\nLedger: {spent} tokens spent by this tooling in the last 24 h; the cap is "
        f"{RECORDER_CAP_TOKENS} (80 pct of {DAILY_CAP_TOKENS}) so the rest stays for production and "
        f"deploy smoke tests. Room for one more call from about {when}."
    )


_USED_RE = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")


def parse_tpd_error(text: str) -> tuple[int, int, int] | None:
    """(limit, used, requested) from a Groq TPD 429 message, or None if it is not one."""
    if "tokens per day" not in text.lower():
        return None
    m = _USED_RE.search(text)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
