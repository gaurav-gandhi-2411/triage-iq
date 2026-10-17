"""A Groq 429 "Used" observation must age out progressively, not block for 24 h as one lump.

Defect (2026-10-12): at 14:49 UTC a TPD 429 reported Used 196,411 against a ledger that knew 17,257. The
179,154 gap was recorded as ONE entry stamped 14:49, so it could not leave the 24 h window before 14:49 the
next day, although real usage (spent at unknown earlier times) drains hour by hour. The ledger stayed shut
all day and a pre-registered measurement was refused by a number that was no longer true.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools" / "groq_budget"
sys.path.insert(0, str(TOOLS))
import groq_budget as gb  # noqa: E402

M120 = "openai/gpt-oss-120b"
M20 = "openai/gpt-oss-20b"
T0 = datetime(2026, 10, 12, 14, 49, tzinfo=timezone.utc)  # the real 429


def _body(used: int) -> str:
    return f"tokens per day (TPD): Limit 200000, Used {used}, Requested 5146. Please try again in 24m."


@pytest.fixture
def led(tmp_path: Path) -> gb.Ledger:
    return gb.Ledger(tmp_path / "root")


def _todays_case(led: gb.Ledger) -> int:
    """The ledger as it was at 14:49: 17,257 recorded by this tooling, then the 429 with Used 196,411."""
    led.record(gb.RECORDER, M120, 17_257, "record_cassettes:today", now=T0 - timedelta(hours=6))
    return led.observe_429(_body(196_411), M120, now=T0)


def test_the_observation_is_exactly_the_gap_at_the_moment_it_is_made(led: gb.Ledger) -> None:
    external = _todays_case(led)
    assert external == 196_411 - 17_257
    assert led.spent(M120, now=T0) == 196_411  # ground truth reproduced
    assert led.by_project(M120, now=T0)[gb.EXTERNAL] == external


def test_the_gap_is_held_as_24_buckets_inside_the_window_not_one_block(led: gb.Ledger) -> None:
    _todays_case(led)
    ext = [e for e in led._read() if e["project"] == gb.EXTERNAL]
    assert len(ext) == gb.EXTERNAL_BUCKETS == 24
    assert all(T0 - timedelta(hours=24) < datetime.fromisoformat(e["ts"]) < T0 for e in ext)
    assert sum(e["tokens"] for e in ext) == 196_411 - 17_257  # remainder lands in a bucket, nothing is lost


def test_todays_stale_block_now_drains_hour_by_hour(led: gb.Ledger) -> None:
    _todays_case(led)
    used = [led.spent(M120, now=T0 + timedelta(hours=h)) for h in range(0, 25)]
    # the old behaviour: still 196,411 at +23 h. Now it falls continuously...
    assert all(a >= b for a, b in zip(used, used[1:], strict=False))  # never rises on its own
    assert used[12] < 0.60 * used[0]
    assert used[23] < 0.15 * used[0]
    assert used[24] == 0  # ...and is gone by the time the 429's own window has fully passed


def test_the_ledger_reopens_hours_after_the_429_not_a_day_later(led: gb.Ledger) -> None:
    _todays_case(led)
    allowed_at = next(
        h for h in range(0, 25) if led.check(gb.RECORDER, M120, 5_000, now=T0 + timedelta(hours=h)).allowed
    )
    assert 4 <= allowed_at <= 8  # 160K allocatable: roughly 37K of the 179K gap must have drained (~5 buckets)
    assert not led.check(gb.RECORDER, M120, 5_000, now=T0).allowed  # still shut right after the 429


def test_a_newer_observation_replaces_the_old_one_and_can_correct_downwards(led: gb.Ledger) -> None:
    _todays_case(led)
    t1 = T0 + timedelta(hours=3)
    led.observe_429(_body(120_000), M120, now=t1)  # Groq now says only 120K is in the window
    assert led.spent(M120, now=t1) == 120_000
    assert sum(1 for e in led._read() if e["project"] == gb.EXTERNAL) == 24  # replaced, not stacked


def test_a_ledger_above_groqs_figure_drops_the_stale_estimate(led: gb.Ledger) -> None:
    _todays_case(led)
    led.record("benchmark", M120, 150_000, "later", now=T0 + timedelta(hours=1))
    assert led.observe_429(_body(100_000), M120, now=T0 + timedelta(hours=2)) == 0
    assert gb.EXTERNAL not in led.by_project(M120, now=T0 + timedelta(hours=2))


def test_other_projects_and_other_models_are_untouched(led: gb.Ledger) -> None:
    led.record("benchmark", M120, 30_000, "b", now=T0 - timedelta(hours=2))
    led.record("benchmark", M20, 9_000, "b20", now=T0 - timedelta(hours=2))
    led.observe_429(_body(100_000), M20, now=T0 - timedelta(hours=1))  # an observation on 20b first
    led.observe_429(_body(196_411), M120, now=T0)
    by120 = led.by_project(M120, now=T0)
    assert by120["benchmark"] == 30_000 and by120[gb.EXTERNAL] == 166_411
    assert led.spent(M20, now=T0 - timedelta(hours=1)) == 100_000  # the 20b observation survived the 120b one


def test_non_tpd_429_is_ignored(led: gb.Ledger) -> None:
    assert led.observe_429("tokens per minute (TPM): Limit 8000, Used 7000, Requested 1500", M120, now=T0) == 0
    assert led._read() == []


def test_cli_prints_the_held_external_total(tmp_path: Path) -> None:
    import os
    import subprocess

    env = {**os.environ, "GROQ_BUDGET_DIR": str(tmp_path / "cli")}
    r = subprocess.run(
        [sys.executable, str(TOOLS / "groq_budget.py"), "observe-429", "--model", M120, "--text", _body(50_000)],
        capture_output=True,
        text=True,
        env=env,
    )
    assert r.returncode == 0 and "50000" in r.stdout


def test_cloud_snapshot_merge_does_not_resurrect_a_superseded_observation(tmp_path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "groq_budget"))
    from gcs_state import merge_ledgers

    old = gb.Ledger(tmp_path / "old")
    old.observe_429(_body(196_411), M120, now=T0)
    new = gb.Ledger(tmp_path / "new")
    new.observe_429(_body(120_000), M120, now=T0 + timedelta(hours=3))
    merged = merge_ledgers({"entries": old._read()}, {"entries": new._read()})
    ext = [e for e in merged["entries"] if e["project"] == gb.EXTERNAL]
    assert len(ext) == 24 and sum(e["tokens"] for e in ext) == 120_000  # only the newer observation survives
