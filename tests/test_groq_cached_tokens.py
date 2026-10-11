"""A3: usage.prompt_tokens_details.cached_tokens is captured per call and stored with the records.

Network-free: a mocked Groq client returning real groq SDK usage types.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from eval import tpd_budget_gate
from groq.types.completion_usage import CompletionUsage, PromptTokensDetails
from tests.test_provider_error_degrade import _make_assistant, _signals

from triage_iq.models.triage import _cached_prompt_tokens


def _resp(content: str, usage: CompletionUsage | None) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")],
        usage=usage,
    )


def _usage(cached: int | None) -> CompletionUsage:
    details = None if cached is None else PromptTokensDetails(cached_tokens=cached)
    return CompletionUsage(
        prompt_tokens=4800,
        completion_tokens=1100,
        total_tokens=5900,
        prompt_tokens_details=details,
    )


@pytest.mark.parametrize(("cached", "expected"), [(4096, 4096), (0, 0), (None, None)])
def test_cached_tokens_captured_in_usage(cached, expected):
    asst = _make_assistant()
    client = MagicMock()
    client.chat.completions.create.return_value = _resp("{}", _usage(cached))
    with patch("groq.Groq", return_value=client):
        _, usage = asst._groq_completion([{"role": "user", "content": "x"}])
    assert usage["cached_tokens"] == expected  # 0 (measured) is distinct from None (not reported)
    assert usage["prompt_tokens"] == 4800


def test_helper_ignores_non_int_and_missing_details():
    assert _cached_prompt_tokens(None) is None
    assert _cached_prompt_tokens(MagicMock()) is None  # a MagicMock attribute is not a count
    assert _cached_prompt_tokens(SimpleNamespace(prompt_tokens_details=None)) is None
    assert _cached_prompt_tokens(_usage(123)) == 123


def test_cached_tokens_stored_in_cache_entry_and_request_metadata():
    asst = _make_assistant()
    cache = MagicMock()
    cache.compute_key.return_value = "k"
    cache.get.return_value = None
    asst._cache = cache
    plan_json = asst._make_fallback_plan(_signals()).model_dump_json()
    client = MagicMock()
    client.chat.completions.create.return_value = _resp(plan_json, _usage(4096))
    with patch("groq.Groq", return_value=client):
        _, _, usage, status, _ = asst._call_llm_verbose(_signals())
    assert status == "ok" and usage["cached_tokens"] == 4096
    # The cassette/cache entry carries it: cache.set(key, provider, model, request, response)
    stored = cache.set.call_args.args[4]
    assert stored["usage"]["cached_tokens"] == 4096
    assert json.dumps(stored)  # serialisable into the cassette as before


def test_ledger_stores_cached_tokens_without_changing_the_sum(tmp_path):
    p = tmp_path / "ledger.json"
    now = datetime(2026, 10, 12, 12, 0, tzinfo=timezone.utc)
    tpd_budget_gate.record_usage(5900, "a", now=now, path=p, cached_tokens=4096)
    tpd_budget_gate.record_usage(5900, "b", now=now, path=p)  # not reported -> key absent
    entries = json.loads(p.read_text(encoding="utf-8"))["entries"]
    assert entries[0]["cached_tokens"] == 4096
    assert "cached_tokens" not in entries[1]
    assert tpd_budget_gate.spent_last_24h(now=now, path=p) == 11800  # gate sum unchanged
