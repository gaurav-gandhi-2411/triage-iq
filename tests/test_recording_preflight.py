"""scripts/recording_preflight.py: prompt inspection, summary and the no-LLM capture guard.

Hermetic: synthetic prompt strings; no model, cassette or network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "eval", ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import recording_preflight as pf  # noqa: E402


def _user(calibrated: bool) -> str:
    label = "80% prediction interval" + (" (coverage-calibrated)" if calibrated else "")
    return f"Repository: kubernetes/kubernetes\n\n--- ISSUE ---\nPoint estimate: 3.5 days\n{label}: [1.0d, 20.0d]\n"


def test_inspect_prompt_reads_point_interval_and_marker() -> None:
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": _user(True)}]
    got = pf.inspect_prompt(msgs)
    assert got["regex_match"] and got["calibrated"]
    assert got["point_days"] == 3.5 and got["interval"] == [1.0, 20.0]
    assert got["est_prompt_tokens"] > 0
    assert not pf.inspect_prompt([{"role": "user", "content": _user(False)}])["calibrated"]


def test_inspect_prompt_flags_a_prompt_the_regex_cannot_read() -> None:
    got = pf.inspect_prompt([{"role": "user", "content": "no interval here"}])
    assert got["regex_match"] is False and got["point_days"] is None


def test_summarize_counts_degraded_misses_and_uncalibrated() -> None:
    rows = [
        {
            "id": "a",
            "repo": "r1",
            "degraded": False,
            "regex_match": True,
            "calibrated": True,
            "point_days": 2.0,
            "point_source": "embedding",
            "est_prompt_tokens": 100,
        },
        {
            "id": "b",
            "repo": "r1",
            "degraded": True,
            "regex_match": True,
            "calibrated": False,
            "point_days": 4.0,
            "point_source": "model",
            "est_prompt_tokens": 50,
        },
        {"id": "c", "repo": "r2", "degraded": True, "regex_match": False, "point_source": None},
    ]
    s = pf.summarize(rows)
    assert s["total_est_prompt_tokens"] == 150
    assert s["degraded"] == ["b", "c"] and s["regex_misses"] == ["c"] and s["uncalibrated"] == ["b"]
    assert s["per_repo"]["r1"]["point_days_min_median_max"] == [2.0, 3.0, 4.0]
    assert s["per_repo"]["r1"]["point_sources"] == {"embedding": 1, "model": 1}


def test_capturing_cache_stops_the_call_with_the_final_messages() -> None:
    msgs = [{"role": "user", "content": "x"}]
    with pytest.raises(pf.PromptCaptured) as exc:
        pf.CapturingCache().compute_key("groq", "m", msgs, 0.0, 1234)
    assert exc.value.messages is msgs and exc.value.max_tokens == 1234
