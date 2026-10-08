from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import tpd_budget_gate as g  # noqa: E402

TPD_429 = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model `openai/gpt-oss-120b` "
    "in organization `org_x` service tier `on_demand` on tokens per day (TPD): Limit 200000, "
    "Used 170000, Requested 45000. Please try again in 1h2m3s.'}}"
)


def test_headroom_passes_and_probes_with_reserve_plus_next_call() -> None:
    seen: list[int] = []
    msg = g.check_budget(seen.append)
    assert seen == [g.RESERVE_TOKENS + g.NEXT_CALL_TOKENS] == [45_000]
    assert "ok" in msg


def test_tpd_429_stops_with_marker_and_reports_usage() -> None:
    def probe(_: int) -> None:
        raise RuntimeError(TPD_429)

    with pytest.raises(g.BudgetReserveError) as ei:
        g.check_budget(probe)
    text = str(ei.value)
    assert g.STOP_MARKER in text
    assert "170000/200000" in text


def test_reserve_is_at_least_30k_and_at_most_20_percent_of_cap() -> None:
    # Owner rule 2026-10-08: reserve >= 30K for deploy smoke tests; recorder stops at 80 pct.
    assert g.RESERVE_TOKENS >= 30_000
    assert g.RESERVE_TOKENS >= 0.2 * g.DAILY_CAP_TOKENS


def test_non_tpd_errors_propagate_not_read_as_headroom() -> None:
    # Fail closed (98a): a probe that cannot answer must stop the run, never count as headroom.
    def probe(_: int) -> None:
        raise ConnectionError("getaddrinfo failed")

    with pytest.raises(ConnectionError):
        g.check_budget(probe)


def test_per_minute_429_is_not_mistaken_for_daily_exhaustion() -> None:
    def probe(_: int) -> None:
        raise RuntimeError("429 rate_limit_exceeded on tokens per minute (TPM): Limit 8000")

    with pytest.raises(RuntimeError) as ei:
        g.check_budget(probe)
    assert not isinstance(ei.value, g.BudgetReserveError)


def test_both_synthesis_loops_call_the_gate_before_synthesizing() -> None:
    src = (Path(__file__).resolve().parents[1] / "eval" / "record_cassettes.py").read_text(
        encoding="utf-8"
    )
    assert src.count("_enforce_budget_reserve(groq_key)") == 2
    assert "tpd_budget_gate.STOP_MARKER in output" in (
        Path(__file__).resolve().parents[1] / "scripts" / "run_recording_unattended.py"
    ).read_text(encoding="utf-8")
