"""Shared machine-wide Groq budget ledger (tools/groq_budget/groq_budget.py)."""

from __future__ import annotations

import json
import multiprocessing
import os
import sys
import time
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


def test_empty_ledger_allows(led: gb.Ledger) -> None:
    d = led.check("benchmark", M120, 5_000, now=NOW)
    assert d.allowed and d.reason == "ok"


def test_production_reserve_is_never_allocated(led: gb.Ledger) -> None:
    cfg = led.config_for(M120)
    assert cfg["production_reserve"] == 40_000
    assert cfg["allocatable"] == 160_000
    _spend(led, gb.RECORDER, M120, 31, 5_000, NOW - timedelta(hours=3))  # 155K
    assert led.check(gb.RECORDER, M120, 5_000, now=NOW).allowed  # 160K exactly
    led.record(gb.RECORDER, M120, 1_000, now=NOW - timedelta(hours=2))  # 156K + 5K > 160K
    d = led.check(gb.RECORDER, M120, 5_000, now=NOW)
    assert not d.allowed and d.reason == "pool_exhausted"
    assert gb.STOP_MARKER in d.detail and "156000" in d.detail


def test_benchmark_is_capped_at_100k(led: gb.Ledger) -> None:
    _spend(led, "benchmark", M120, 19, 5_000, NOW - timedelta(hours=3))  # 95K
    assert led.check("benchmark", M120, 5_000, now=NOW).allowed
    led.record("benchmark", M120, 1_000, now=NOW - timedelta(hours=2))  # 96K + 5K > 100K
    d = led.check("benchmark", M120, 5_000, now=NOW)
    assert not d.allowed and d.reason == "project_cap"
    assert d.retry_after_s > 0


def test_recorder_gets_the_remainder_after_benchmark(led: gb.Ledger) -> None:
    _spend(led, "benchmark", M120, 20, 5_000, NOW - timedelta(hours=4))  # 100K, at its cap
    # recorder has no fixed cap: 160K allocatable - 100K benchmark = 60K left for it
    _spend(led, gb.RECORDER, M120, 11, 5_000, NOW - timedelta(hours=3))  # 55K
    assert led.check(gb.RECORDER, M120, 5_000, now=NOW).allowed
    led.record(gb.RECORDER, M120, 1_000, now=NOW - timedelta(hours=2))
    assert not led.check(gb.RECORDER, M120, 5_000, now=NOW).allowed


def test_models_have_separate_pools(led: gb.Ledger) -> None:
    _spend(led, "benchmark", M20, 32, 5_000, NOW - timedelta(hours=3))  # 20b full
    assert not led.check("benchmark", M20, 5_000, now=NOW).allowed
    assert led.check("benchmark", M120, 5_000, now=NOW).allowed


def test_spend_older_than_24h_expires(led: gb.Ledger) -> None:
    _spend(led, "benchmark", M120, 32, 5_000, NOW - timedelta(hours=30))
    assert led.spent(M120, now=NOW) == 0
    assert led.check("benchmark", M120, 5_000, now=NOW).allowed


def test_retry_after_says_when_room_returns(led: gb.Ledger) -> None:
    led.record(gb.RECORDER, M120, 160_000, now=NOW - timedelta(hours=20))
    d = led.check(gb.RECORDER, M120, 5_000, now=NOW)
    assert not d.allowed and 3 * 3600 < d.retry_after_s <= 4 * 3600 + 1


def test_user_config_overrides_allocations(led: gb.Ledger) -> None:
    led.root.mkdir(parents=True)
    (led.root / "config.json").write_text(
        json.dumps({M120: {"production_reserve": 60_000, "projects": {"benchmark": 20_000}}}), encoding="utf-8"
    )
    cfg = led.config_for(M120)
    assert cfg["allocatable"] == 140_000 and cfg["projects"]["benchmark"] == 20_000


# ---------------------------------------------------------------- recording-active flag
# ----------------------------------------------------------------------- Groq 429 folding
_BODY = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier `on_demand` "
    "on tokens per day (TPD): Limit 200000, Used 198924, Requested 5000. Please try again in 1h."
)


def test_observe_429_folds_external_spend(led: gb.Ledger) -> None:
    _spend(led, gb.RECORDER, M120, 10, 5_000, NOW - timedelta(hours=3))  # 50K known
    added = led.observe_429(_BODY, M120, now=NOW)
    assert added == 198_924 - 50_000
    assert led.spent(M120, now=NOW) == 198_924
    assert led.by_project(M120, now=NOW)[gb.EXTERNAL] == added
    assert not led.check(gb.RECORDER, M120, 5_000, now=NOW).allowed


def test_observe_ignores_non_tpd_and_known_spend(led: gb.Ledger) -> None:
    assert led.observe_429("Rate limit reached ... tokens per minute (TPM): Limit 8000", M120, now=NOW) == 0
    _spend(led, "benchmark", M120, 40, 5_000, NOW - timedelta(hours=3))  # ledger already above Groq's Used
    assert led.observe_429(_BODY, M120, now=NOW) == 0


# ------------------------------------------------------------ storage, history, concurrency
def test_cached_tokens_are_stored_and_total_unchanged(led: gb.Ledger) -> None:
    led.record("benchmark", M120, 6_000, "x", cached_tokens=4_000, now=NOW)
    entry = json.loads(led.path.read_text(encoding="utf-8"))["entries"][0]
    assert entry["cached_tokens"] == 4_000 and led.spent(M120, now=NOW) == 6_000


def test_legacy_entries_read_as_recorder_on_120b(led: gb.Ledger) -> None:
    led.root.mkdir(parents=True)
    legacy = {"entries": [{"ts": (NOW - timedelta(hours=1)).isoformat(), "tokens": 6_000, "source": "old"}]}
    led.path.write_text(json.dumps(legacy), encoding="utf-8")
    assert led.spent(M120, gb.RECORDER, now=NOW) == 6_000


def test_migrate_legacy_preserves_history_and_is_idempotent(led: gb.Ledger, tmp_path: Path) -> None:
    src = tmp_path / "old.json"
    entries = [
        {"ts": (NOW - timedelta(hours=h)).isoformat(), "tokens": 6_000 + h, "source": f"record_cassettes:{h}"}
        for h in range(5)
    ]
    src.write_text(json.dumps({"entries": entries}), encoding="utf-8")
    before = src.read_text(encoding="utf-8")
    assert led.migrate_legacy(src) == 5
    assert led.migrate_legacy(src) == 0
    assert led.spent(M120, gb.RECORDER, now=NOW) == sum(e["tokens"] for e in entries)
    assert src.read_text(encoding="utf-8") == before  # the legacy file is never modified


def _hammer(args: tuple[str, int]) -> None:
    root, i = args
    led = gb.Ledger(Path(root))
    for k in range(25):
        led.record(f"p{i}", M120, 10, f"{i}-{k}")


def test_concurrent_processes_lose_no_entries(tmp_path: Path) -> None:
    root = str(tmp_path / "root")
    with multiprocessing.get_context("spawn").Pool(4) as pool:
        pool.map(_hammer, [(root, i) for i in range(4)])
    assert len(json.loads((Path(root) / "ledger.json").read_text(encoding="utf-8"))["entries"]) == 100


def test_stale_lock_from_a_dead_holder_is_taken_over(led: gb.Ledger) -> None:
    led.root.mkdir(parents=True)
    lock = led.root / "ledger.lock"
    lock.write_text("99999")
    old = time.time() - 120
    os.utime(lock, (old, old))
    led.record("benchmark", M120, 10, now=NOW)  # would hang without stale-lock recovery
    assert led.spent(M120, now=NOW) == 10


# ------------------------------------------------------------------------------------ CLI
