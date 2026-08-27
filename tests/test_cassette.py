"""Unit tests for eval/cassette.py's CassettePlayer, focused on the request-storage
schema addition (2026-08-27) and backward compatibility with pre-existing cassettes."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "eval"))

from cassette import CassettePlayer, CassetteMissError  # noqa: E402

_MESSAGES = [{"role": "user", "content": "hello"}]


@pytest.fixture
def cassette_path(tmp_path: Path) -> Path:
    return tmp_path / "test_cassette.json"


def test_set_then_get_round_trips(cassette_path: Path):
    c = CassettePlayer(cassette_path, strict=True)
    key = c.compute_key("groq", "m", _MESSAGES)
    c.set(key, "groq", "m", _MESSAGES, {"content": "hi", "usage": {}})
    assert c.get(key) == {"content": "hi", "usage": {}}


def test_get_request_returns_stored_messages(cassette_path: Path):
    c = CassettePlayer(cassette_path, strict=True)
    key = c.compute_key("groq", "m", _MESSAGES)
    c.set(key, "groq", "m", _MESSAGES, {"content": "hi", "usage": {}})
    req = c.get_request(key)
    assert req is not None
    assert req["provider"] == "groq"
    assert req["model"] == "m"
    assert req["request_messages"] == _MESSAGES


def test_strict_miss_raises(cassette_path: Path):
    c = CassettePlayer(cassette_path, strict=True)
    with pytest.raises(CassetteMissError):
        c.get("nonexistent-key")


def test_non_strict_miss_returns_none(cassette_path: Path):
    c = CassettePlayer(cassette_path, strict=False)
    assert c.get("nonexistent-key") is None


def test_backward_compat_reads_pre_schema_entry(cassette_path: Path):
    """A cassette written before the request-storage addition stores the response dict
    directly (no wrapper) -- get() must still return it correctly."""
    c = CassettePlayer(cassette_path, strict=True)
    old_format_key = "old-style-key"
    c._entries[old_format_key] = {"content": "legacy response", "usage": {}}
    c._save()

    c2 = CassettePlayer(cassette_path, strict=True)
    assert c2.get(old_format_key) == {"content": "legacy response", "usage": {}}
    # get_request on a pre-schema entry: the entry exists but has no stored request --
    # this is a real (expected) None, not a bug.
    assert c2.get_request(old_format_key) is None


def test_persists_across_reload(cassette_path: Path):
    c = CassettePlayer(cassette_path, strict=True)
    key = c.compute_key("groq", "m", _MESSAGES)
    c.set(key, "groq", "m", _MESSAGES, {"content": "hi", "usage": {}})

    c2 = CassettePlayer(cassette_path, strict=True)
    assert c2.get(key) == {"content": "hi", "usage": {}}
    assert c2.get_request(key)["request_messages"] == _MESSAGES


def test_diff_against_nearest_finds_closest_entry(cassette_path: Path):
    c = CassettePlayer(cassette_path, strict=True)
    close_messages = [{"role": "user", "content": "The quick brown fox jumps over the lazy dog A"}]
    far_messages = [{"role": "user", "content": "completely unrelated text"}]
    c.set(c.compute_key("groq", "m", close_messages), "groq", "m", close_messages, {"content": "r1"})
    c.set(c.compute_key("groq", "m", far_messages), "groq", "m", far_messages, {"content": "r2"})

    target = [{"role": "user", "content": "The quick brown fox jumps over the lazy dog B"}]
    report = c.diff_against_nearest(target)
    assert "matches for the first" in report
    # Should identify the close entry (shared prefix much longer than the far one).
    assert "diverges" in report


def test_diff_against_nearest_reports_when_no_requests_stored(cassette_path: Path):
    c = CassettePlayer(cassette_path, strict=True)
    c._entries["old-key"] = {"content": "legacy"}
    c._save()
    c2 = CassettePlayer(cassette_path, strict=True)
    report = c2.diff_against_nearest(_MESSAGES)
    assert "pre-date the request-storage schema" in report
