from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import tpd_budget_gate as g  # noqa: E402

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


def _spend(path: Path, n: int, tokens: int, start: datetime, step_min: int = 1) -> None:
    for i in range(n):
        g.record_usage(tokens, "t", now=start + timedelta(minutes=i * step_min), path=path)


def test_empty_ledger_allows_a_call(tmp_path: Path) -> None:
    assert "ok" in g.check_budget(now=NOW, path=tmp_path / "l.json")


def test_stops_at_80_percent_of_the_daily_cap_keeping_40k(tmp_path: Path) -> None:
    # Owner rule 2026-10-08: recorder stops at 80 pct; >= 30K stays for deploy smoke tests.
    assert g.RECORDER_CAP_TOKENS == 0.8 * g.DAILY_CAP_TOKENS
    assert g.DAILY_CAP_TOKENS - g.RECORDER_CAP_TOKENS >= 30_000
    p = tmp_path / "l.json"
    _spend(p, 31, 5_000, NOW - timedelta(hours=3))  # 155K spent, 160K cap, 5K next call: still ok
    g.check_budget(now=NOW, path=p)
    g.record_usage(1_000, "t", now=NOW - timedelta(hours=2), path=p)  # 156K + 5K > 160K
    with pytest.raises(g.BudgetReserveError) as ei:
        g.check_budget(now=NOW, path=p)
    assert g.STOP_MARKER in str(ei.value)
    assert "156000" in str(ei.value)


def test_spend_older_than_24h_expires(tmp_path: Path) -> None:
    p = tmp_path / "l.json"
    _spend(p, 40, 5_000, NOW - timedelta(hours=30))  # 200K, all older than 24 h
    assert g.spent_last_24h(now=NOW, path=p) == 0
    g.check_budget(now=NOW, path=p)


def test_message_says_when_room_returns(tmp_path: Path) -> None:
    p = tmp_path / "l.json"
    start = NOW - timedelta(hours=10)
    _spend(p, 32, 5_000, start, step_min=10)
    with pytest.raises(g.BudgetReserveError) as ei:
        g.check_budget(now=NOW, path=p)
    assert (start + timedelta(hours=24)).strftime("%Y-%m-%d") in str(ei.value)


def test_parse_tpd_error() -> None:
    msg = "429 on tokens per day (TPD): Limit 200000, Used 198924, Requested 4895. Please try again in 5m."
    assert g.parse_tpd_error(msg) == (200000, 198924, 4895)
    assert g.parse_tpd_error("rate_limit_exceeded on tokens per minute (TPM): Limit 8000") is None


def test_gate_and_ledger_are_wired_into_the_recorder() -> None:
    root = Path(__file__).resolve().parents[1]
    rec = (root / "eval" / "record_cassettes.py").read_text(encoding="utf-8")
    assert rec.count("_enforce_budget_reserve(groq_key)") == 2
    assert "tpd_budget_gate.record_usage(" in rec
    assert "tpd_budget_gate.STOP_MARKER in output" in (
        root / "scripts" / "run_recording_unattended.py"
    ).read_text(encoding="utf-8")


FIXTURE_429 = Path(__file__).parent / "fixtures" / "groq_tpd_429_2026-10-08.txt"


def test_recorded_429_body_parses() -> None:
    """The real Groq body from 2026-10-08 01:12 UTC (kept verbatim, truncated where logged)."""
    assert g.parse_tpd_error(FIXTURE_429.read_text(encoding="utf-8")) == (200000, 198924, 5502)


def test_a_probe_cannot_override_the_ledger(tmp_path: Path) -> None:
    """Regression for the 2026-10-08 probe: a max_tokens=65000 request was ADMITTED while the
    org had Used 198,924 of 200,000, because Groq admits on prompt tokens plus a bounded slice
    of max_tokens. The gate therefore reads only the ledger; no network call is involved, and
    an org that Groq reports as full keeps it closed even when this tooling spent little."""
    p = tmp_path / "l.json"
    _spend(p, 4, 5_000, NOW - timedelta(hours=1))  # tooling spent only 20K
    g.check_budget(now=NOW, path=p)  # tooling-only view: open
    added = g.observe_tpd_429(FIXTURE_429.read_text(encoding="utf-8"), now=NOW, path=p)
    assert added == 198_924 - 20_000
    with pytest.raises(g.BudgetReserveError) as ei:
        g.check_budget(now=NOW, path=p)
    assert g.STOP_MARKER in str(ei.value)


def test_observe_ignores_non_tpd_and_known_spend(tmp_path: Path) -> None:
    p = tmp_path / "l.json"
    assert g.observe_tpd_429("429 on tokens per minute (TPM): Limit 8000", now=NOW, path=p) == 0
    _spend(p, 40, 5_000, NOW - timedelta(hours=2))  # ledger already accounts for 200K
    assert g.observe_tpd_429(FIXTURE_429.read_text(encoding="utf-8"), now=NOW, path=p) == 0
