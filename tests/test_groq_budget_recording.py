"""Shared machine-wide Groq budget ledger (tools/groq_budget/groq_budget.py)."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "groq_budget"
sys.path.insert(0, str(TOOLS))
import groq_budget as gb  # noqa: E402

M120 = "openai/gpt-oss-120b"
M20 = "openai/gpt-oss-20b"
NOW = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)


@pytest.fixture
def led(tmp_path: Path) -> gb.Ledger:
    return gb.Ledger(tmp_path / "root")


def _spend(led: gb.Ledger, project: str, model: str, n: int, tokens: int, start: datetime) -> None:
    for i in range(n):
        led.record(project, model, tokens, "t", now=start + timedelta(minutes=i))


def test_recording_flag_makes_other_projects_wait_but_not_the_owner(led: gb.Ledger) -> None:
    led.start_recording(gb.RECORDER, [M120])
    d = led.check("benchmark", M120, 1_000)
    assert not d.allowed and d.reason == "recording_active"
    assert led.check(gb.RECORDER, M120, 1_000).allowed
    assert led.check("benchmark", M20, 1_000).allowed  # a different model is not affected
    assert led.stop_recording(gb.RECORDER)
    assert led.check("benchmark", M120, 1_000).allowed


def test_only_the_owner_can_clear_the_flag(led: gb.Ledger) -> None:
    led.start_recording(gb.RECORDER, [M120])
    assert not led.stop_recording("benchmark")
    assert led.active_recording(M120) is not None


def test_expired_or_dead_recording_flag_does_not_block(led: gb.Ledger) -> None:
    led.start_recording(gb.RECORDER, [M120], ttl_s=60)
    assert led.active_recording(M120, now=datetime.now(timezone.utc) + timedelta(seconds=120)) is None
    led.start_recording(gb.RECORDER, [M120], pid=2**22 + 12345)  # no such process
    assert led.active_recording(M120) is None
    assert led.check("benchmark", M120, 1_000).allowed


def test_wait_returns_when_the_flag_clears(led: gb.Ledger) -> None:
    led.start_recording(gb.RECORDER, [M120], ttl_s=60)
    d = led.wait_until_allowed("benchmark", M120, 1_000, timeout_s=0.2, poll_s=0.05)
    assert not d.allowed  # still recording after the timeout
    led.stop_recording(gb.RECORDER)
    assert led.wait_until_allowed("benchmark", M120, 1_000, timeout_s=1, poll_s=0.05).allowed


def _cli(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "GROQ_BUDGET_DIR": str(root)}
    return subprocess.run(
        [sys.executable, str(TOOLS / "groq_budget.py"), *args], capture_output=True, text=True, env=env
    )


def test_cli_check_record_status(tmp_path: Path) -> None:
    root = tmp_path / "cli"
    assert _cli(root, "check", "--project", "benchmark", "--model", M120, "--est", "1000").returncode == 0
    assert _cli(root, "record", "--project", "benchmark", "--model", M120, "--tokens", "99500").returncode == 0
    r = _cli(root, "check", "--project", "benchmark", "--model", M120, "--est", "1000")
    assert r.returncode == 4 and "WAIT [project_cap]" in r.stdout
    s = _cli(root, "status", "--model", M120)
    assert "benchmark: 99500 / 100000" in s.stdout
