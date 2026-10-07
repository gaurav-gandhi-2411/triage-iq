from __future__ import annotations

import re
from collections.abc import Callable

"""Stop a Groq recording pass before it spends the production serving budget.

The Groq free-tier TPD budget (200,000 tokens/day, org-wide) is shared by every recording pass,
gg-portfolio and the production /triage endpoint. On 2026-10-07 a 53-call re-record spent all of
it and the next production deploy's smoke test failed with a 429 (DECISION_LOG D28). Owner rule:
the recorder stops at 80 pct of the daily cap, which keeps at least 30K tokens for deploy smoke
tests.

Groq exposes no "tokens remaining today" read, but it rejects a request whose REQUESTED tokens
(prompt + max_tokens) exceed what is left, naming Limit/Used in the error. So one tiny call that
asks for `RESERVE_TOKENS + NEXT_CALL_TOKENS` answers "is there still at least the reserve left after
the next synthesis call?" without spending more than a handful of tokens when the answer is yes.
"""

DAILY_CAP_TOKENS = 200_000
# 20 pct of the cap stays untouched (>= the 30K deploy-smoke-test reserve the owner asked for).
RESERVE_TOKENS = 40_000
# Largest request one synthesis call makes (prompt + max_tokens), measured at ~4.9K on 2026-10-07.
NEXT_CALL_TOKENS = 5_000
PROBE_REQUEST_TOKENS = RESERVE_TOKENS + NEXT_CALL_TOKENS

STOP_MARKER = "=== BUDGET RESERVE STOP ==="

_LIMIT_RE = re.compile(r"Limit (\d+), Used (\d+), Requested (\d+)")


class BudgetReserveError(RuntimeError):
    """Raised when the remaining daily budget is below the protected reserve."""


def parse_tpd_error(text: str) -> tuple[int, int, int] | None:
    """(limit, used, requested) from a Groq TPD 429 message, or None if it is not one."""
    if "tokens per day" not in text.lower():
        return None
    m = _LIMIT_RE.search(text)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def check_budget(probe: Callable[[int], None]) -> str:
    """Raise BudgetReserveError if fewer than RESERVE_TOKENS would remain after the next call.

    `probe(max_tokens)` performs one tiny Groq chat call asking for that many max_tokens and
    raises the provider exception on a 429. Anything that is NOT a TPD 429 propagates unchanged
    (fail closed: a probe that cannot answer must stop the run, never be read as headroom).
    Returns a one-line description of what was learned.
    """
    try:
        probe(PROBE_REQUEST_TOKENS)
    except Exception as exc:  # noqa: BLE001 - provider-specific type; classified below
        parsed = parse_tpd_error(str(exc))
        if parsed is None:
            raise
        limit, used, _ = parsed
        raise BudgetReserveError(
            f"{STOP_MARKER}\nGroq TPD used {used}/{limit}; fewer than {PROBE_REQUEST_TOKENS} tokens "
            f"remain, so the {RESERVE_TOKENS}-token reserve for deploy smoke tests would be spent. "
            "Resume after the rolling 24 h window frees budget."
        ) from exc
    return f"budget ok: at least {PROBE_REQUEST_TOKENS} tokens remain"


def groq_probe(api_key: str, model: str) -> Callable[[int], None]:
    """Probe callable bound to a key/model; max_retries=0 so a 429 surfaces immediately."""
    from groq import Groq

    client = Groq(api_key=api_key, max_retries=0, timeout=30)

    def _probe(max_tokens: int) -> None:
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": "Say OK."}], max_tokens=max_tokens
        )

    return _probe
