"""LLM stage time budget: a per-minute rate limit degrades inside the budget, not after minutes.

Production 2026-10-11 (reports/latency_replay_2026-10-12.txt): 4 of 5 /triage calls took 53-135 s
(HTTP 200, not degraded) sleeping through Groq tokens-per-minute 429s. With a deadline the LLM
stage gives up at the budget and returns the signals-only plan with _degraded=true and a reason.
Network-free: a fake clock (sleep advances it) and a mocked Groq client raising real groq errors.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import groq
import httpx
import pytest
from tests.test_provider_error_degrade import (
    _TPM_MSG,
    _empty_cache,
    _make_assistant,
    _ok_response,
    _rate_limit,
    _signals,
)

from triage_iq.models.triage import (
    LLM_TIME_BUDGET_REASON,
    PROVIDER_DEGRADED_STATUS,
    LLMTimeBudgetExceeded,
    classify_provider_error,
)

_REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
_BUDGET_S = 15.0
# A schema-valid plan, built from the real fallback-plan constructor (no hand-typed fields).
_VALID_PLAN_JSON = _make_assistant()._make_fallback_plan(_signals()).model_dump_json()


class _Clock:
    """time stand-in: sleep() advances monotonic(), so budgets are testable without waiting."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def perf_counter(self) -> float:
        return self.now

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def _run(side_effect, budget_s: float | None = _BUDGET_S):
    clock = _Clock()
    asst = _make_assistant()
    asst.llm_time_budget_s = budget_s
    asst._cache = _empty_cache()
    client = MagicMock()
    client.chat.completions.create.side_effect = side_effect
    with (
        patch("groq.Groq", return_value=client) as groq_ctor,
        patch("triage_iq.models.triage.time", clock),
    ):
        out = asst._call_llm_verbose(_signals())
    return out, client, groq_ctor, clock


def test_rate_limit_degrades_within_budget():
    (plan, raw, usage, status, hit), client, _, clock = _run(_rate_limit(_TPM_MSG))
    assert status == PROVIDER_DEGRADED_STATUS
    assert usage["llm_status_reason"] == LLM_TIME_BUDGET_REASON == "llm_time_budget_exceeded"
    assert raw == "" and hit is False
    assert clock.now - 1000.0 <= _BUDGET_S  # never ran past the budget
    # backoff 2.5 s, 6.0 s fit; the 14.0 s sleep would cross 15 s, so it degrades instead.
    assert clock.sleeps == [2.5, 6.0]
    assert client.chat.completions.create.call_count == 3
    assert "manual review" in plan.triage_summary.lower()
    assert [s.number for s in plan.similar_issues] == [11, 12]


def test_long_retry_after_degrades_immediately():
    exc = _rate_limit(_TPM_MSG)
    exc.response.headers["retry-after"] = "40"
    (_, _, usage, status, _), client, _, clock = _run(exc)
    assert status == PROVIDER_DEGRADED_STATUS
    assert usage["llm_status_reason"] == LLM_TIME_BUDGET_REASON
    assert client.chat.completions.create.call_count == 1
    assert clock.sleeps == [] and clock.now == 1000.0


def test_normal_call_is_unaffected_by_the_budget():
    content = _VALID_PLAN_JSON
    (_, raw, usage, status, _), client, groq_ctor, clock = _run([_ok_response(content)])
    assert status != PROVIDER_DEGRADED_STATUS
    assert raw == content and "llm_status_reason" not in usage
    assert client.chat.completions.create.call_count == 1
    assert clock.sleeps == []
    # The SDK's own (invisible, ~30 s per attempt) retries are off and the attempt is time-boxed.
    assert groq_ctor.call_args.kwargs["max_retries"] == 0
    assert client.chat.completions.create.call_args.kwargs["timeout"] == pytest.approx(_BUDGET_S)


def test_recovery_inside_budget_still_succeeds():
    content = _VALID_PLAN_JSON
    (_, raw, _, status, _), client, _, clock = _run([_rate_limit(_TPM_MSG), _ok_response(content)])
    assert status != PROVIDER_DEGRADED_STATUS and raw == content
    assert client.chat.completions.create.call_count == 2
    assert clock.sleeps == [2.5]


def test_timeout_after_deadline_degrades_as_budget_not_provider_timeout():
    class _SlowTimeout(groq.APITimeoutError):
        pass

    clock_holder: dict = {}

    def slow(*_a, **_k):
        clock_holder["clock"].now += _BUDGET_S + 1  # the call itself consumed the budget
        raise _SlowTimeout(request=_REQ)

    clock = _Clock()
    clock_holder["clock"] = clock
    asst = _make_assistant()
    asst.llm_time_budget_s = _BUDGET_S
    asst._cache = _empty_cache()
    client = MagicMock()
    client.chat.completions.create.side_effect = slow
    with patch("groq.Groq", return_value=client), patch("triage_iq.models.triage.time", clock):
        _, _, usage, status, _ = asst._call_llm_verbose(_signals())
    assert status == PROVIDER_DEGRADED_STATUS
    assert usage["llm_status_reason"] == LLM_TIME_BUDGET_REASON
    assert client.chat.completions.create.call_count == 1


def test_no_budget_keeps_the_unbounded_retry_behaviour():
    (_, _, usage, status, _), client, groq_ctor, clock = _run(_rate_limit(_TPM_MSG), budget_s=None)
    assert status == PROVIDER_DEGRADED_STATUS
    assert usage["llm_status_reason"] == "rate_limited_tpm"  # the pre-existing reason, 6 attempts
    assert client.chat.completions.create.call_count == 6
    assert len(clock.sleeps) == 5
    assert "max_retries" not in groq_ctor.call_args.kwargs
    assert "timeout" not in client.chat.completions.create.call_args.kwargs


def test_budget_is_shared_with_the_parse_retry_call():
    """A parse-retry second call must not get a fresh budget: one deadline per request."""
    clock = _Clock()
    calls: list[int] = []

    def first_slow_then_rate_limited(*_a, **_k):
        calls.append(1)
        if len(calls) == 1:
            clock.now += 10.0  # the first call used 10 of the 15 s and returned non-JSON
            return _ok_response("not json at all")
        raise _rate_limit(_TPM_MSG)

    asst = _make_assistant()
    asst.llm_time_budget_s = _BUDGET_S
    asst._cache = _empty_cache()
    client = MagicMock()
    client.chat.completions.create.side_effect = first_slow_then_rate_limited
    with patch("groq.Groq", return_value=client), patch("triage_iq.models.triage.time", clock):
        _, _, usage, status, _ = asst._call_llm_verbose(_signals())
    assert status == PROVIDER_DEGRADED_STATUS
    assert usage["llm_status_reason"] == LLM_TIME_BUDGET_REASON
    # 5 s left for the retry: one 2.5 s backoff fits, the 6.0 s one does not (a fresh 15 s
    # budget would have allowed both).
    assert clock.sleeps == [2.5]
    assert clock.now - 1000.0 <= _BUDGET_S


def test_classify_maps_budget_exception():
    assert classify_provider_error(LLMTimeBudgetExceeded("x")) == LLM_TIME_BUDGET_REASON


def test_loader_default_budget_is_15s(monkeypatch):
    import inspect

    from triage_iq.api import loader

    src = inspect.getsource(loader)
    assert 'TRIAGE_LLM_TIME_BUDGET_S", "15"' in src
    assert "llm_time_budget_s=llm_time_budget_s" in src
