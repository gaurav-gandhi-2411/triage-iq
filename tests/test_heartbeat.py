"""Unit tests for the heartbeat/staleness guard (scripts/heartbeat.py)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, "scripts")
from heartbeat import Heartbeat, check_staleness  # noqa: E402


def test_beat_creates_file_and_parent_dirs(tmp_path: Path):
    log_path = tmp_path / "nested" / "hb.jsonl"
    hb = Heartbeat(log_path)
    hb.beat(step=1, note="warmup")
    assert log_path.exists()


def test_beat_appends_one_line_per_call(tmp_path: Path):
    log_path = tmp_path / "hb.jsonl"
    hb = Heartbeat(log_path)
    hb.beat(step=1)
    hb.beat(step=2)
    hb.beat(step=3)
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3


def test_check_staleness_missing_file_reports_not_exists(tmp_path: Path):
    result = check_staleness(tmp_path / "never_written.jsonl", max_age_minutes=10)
    assert result.exists is False
    assert result.fresh is False
    assert result.age_seconds is None


def test_check_staleness_empty_file_reports_not_exists(tmp_path: Path):
    log_path = tmp_path / "empty.jsonl"
    log_path.touch()
    result = check_staleness(log_path, max_age_minutes=10)
    assert result.exists is False
    assert result.fresh is False


def test_check_staleness_recent_beat_is_fresh(tmp_path: Path):
    log_path = tmp_path / "hb.jsonl"
    hb = Heartbeat(log_path)
    hb.beat(step=5, note="loss=0.1234")
    result = check_staleness(log_path, max_age_minutes=10)
    assert result.exists is True
    assert result.fresh is True
    assert result.age_seconds < 5
    assert result.last_record["step"] == 5
    assert result.last_record["note"] == "loss=0.1234"


def test_check_staleness_old_beat_is_stale(tmp_path: Path):
    log_path = tmp_path / "hb.jsonl"
    hb = Heartbeat(log_path)
    hb.beat(step=1)
    # Rewrite the single line with a timestamp far in the past instead of sleeping in a test.
    import json

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    record["ts"] = time.time() - 3600
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = check_staleness(log_path, max_age_minutes=10)
    assert result.exists is True
    assert result.fresh is False
    assert result.age_seconds >= 3600


def test_check_staleness_uses_last_line_not_first(tmp_path: Path):
    log_path = tmp_path / "hb.jsonl"
    hb = Heartbeat(log_path)
    hb.beat(step=1)
    hb.beat(step=2)
    hb.beat(step=3)
    result = check_staleness(log_path, max_age_minutes=10)
    assert result.last_record["step"] == 3


def test_check_staleness_just_under_threshold_is_fresh(tmp_path: Path):
    """599s old against a 10min (600s) threshold must still read as fresh."""
    import json

    log_path = tmp_path / "hb.jsonl"
    hb = Heartbeat(log_path)
    hb.beat(step=1)
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    record["ts"] = time.time() - 599
    log_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    result = check_staleness(log_path, max_age_minutes=10)
    assert result.fresh is True
