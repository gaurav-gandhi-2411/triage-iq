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


# --- shared machine-wide ledger (2026-10-12) ---------------------------------------------------
def test_default_ledger_imports_the_legacy_history_once(tmp_path: Path, monkeypatch) -> None:
    import json

    legacy = tmp_path / "old.json"
    entries = [
        {"ts": (NOW - timedelta(hours=h)).isoformat(), "tokens": 6_000, "source": f"record_cassettes:{h}"}
        for h in range(3)
    ]
    legacy.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    monkeypatch.delenv(g.LEDGER_ENV, raising=False)
    monkeypatch.setenv("GROQ_BUDGET_DIR", str(tmp_path / "shared"))
    monkeypatch.setattr(g, "LEGACY_LEDGER", legacy)
    monkeypatch.setattr(g, "_MIGRATED", False)
    assert g.spent_last_24h(now=NOW) == 18_000
    assert g.spent_last_24h(now=NOW) == 18_000  # second use does not import again
    assert json.loads(legacy.read_text(encoding="utf-8"))["entries"] == entries  # legacy untouched


def test_recorder_spend_is_visible_to_other_projects_on_the_shared_ledger(tmp_path: Path) -> None:
    import groq_budget as gb

    p = tmp_path / "l.json"
    _spend(p, 31, 5_000, NOW - timedelta(hours=3))  # recorder spent 155K of the 160K allocatable
    d = gb.Ledger(path=p).check("benchmark", g.MODEL, 6_000, now=NOW)
    assert not d.allowed and d.reason == "pool_exhausted"


def test_recording_flag_is_wired_into_both_recorder_entry_points() -> None:
    root = Path(__file__).resolve().parents[1]
    rec = (root / "eval" / "record_cassettes.py").read_text(encoding="utf-8")
    run = (root / "scripts" / "run_recording_unattended.py").read_text(encoding="utf-8")
    assert "tpd_budget_gate.start_recording()" in rec and "GROQ_BUDGET_RECORDING_PARENT" in rec
    assert "tpd_budget_gate.start_recording_heartbeat()" in run
    assert "atexit.register(heartbeat.set)" in run
    assert 'env["GROQ_BUDGET_RECORDING_PARENT"] = "1"' in run


def test_heartbeat_renews_the_flag_past_its_ttl_and_stops_on_request(tmp_path: Path) -> None:
    import time

    p = tmp_path / "l.json"
    led = g._ledger(p)
    stop = g.start_recording_heartbeat(interval_s=0.05, ttl_s=1, path=p)
    try:
        first = led.active_recording(g.MODEL)
        assert first is not None
        time.sleep(1.5)  # longer than the 1 s TTL: only renewal keeps the flag alive
        rec = led.active_recording(g.MODEL)
        assert rec is not None and rec["expires_ts"] > first["expires_ts"]
        assert rec["since_ts"] == first["since_ts"]  # renewal keeps the original start time
    finally:
        stop.set()
    time.sleep(1.3)  # renewals stopped: the flag now lapses on its TTL
    assert led.active_recording(g.MODEL) is None
    g.stop_recording(p)
