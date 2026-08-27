"""Unit tests for TruncatedCompletionError (2026-08-28).

A truncated completion (Groq finish_reason == "length") must be a distinct, loud failure
from a generic JSON-parse error -- that ambiguity is what hid the actual defect behind a
68.75% first-attempt parse-failure rate for an entire engagement. These tests exercise the
real _groq_completion method against a mocked Groq client (not a mocked _groq_completion,
which would bypass the finish_reason check entirely).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from triage_iq.models.triage import TriageAssistant, TruncatedCompletionError


def _make_assistant() -> TriageAssistant:
    asst = TriageAssistant.__new__(TriageAssistant)
    asst.repo = "microsoft/vscode"
    asst.model = "openai/gpt-oss-20b"
    asst.temperature = 0.0
    asst.max_tokens = 1024
    asst.seed = 42
    asst._groq_key = "test-key"
    asst.use_structured_output = False
    return asst


def _mock_groq_response(content: str, finish_reason: str, completion_tokens: int = 1024):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content), finish_reason=finish_reason)]
    resp.usage = MagicMock(prompt_tokens=500, completion_tokens=completion_tokens)
    return resp


def test_groq_completion_raises_on_length_finish_reason():
    asst = _make_assistant()
    truncated_content = '{"predicted_component": "kubectl", "similar_issues": ['
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        truncated_content, finish_reason="length", completion_tokens=1024
    )
    with patch("groq.Groq", return_value=mock_client):
        with pytest.raises(TruncatedCompletionError) as exc_info:
            asst._groq_completion([{"role": "user", "content": "triage this"}])
    assert exc_info.value.completion_tokens == 1024
    assert exc_info.value.max_tokens == 1024
    assert "length" in str(exc_info.value).lower() or "truncat" in str(exc_info.value).lower()


def test_groq_completion_does_not_raise_on_stop_finish_reason():
    asst = _make_assistant()
    complete_content = '{"predicted_component": "kubectl"}'
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        complete_content, finish_reason="stop", completion_tokens=200
    )
    with patch("groq.Groq", return_value=mock_client):
        content, usage = asst._groq_completion([{"role": "user", "content": "triage this"}])
    assert content == complete_content
    assert usage["finish_reason"] == "stop"
    assert usage["completion_tokens"] == 200


def test_truncated_completion_never_reaches_cache():
    """The defect this whole fix exists for: a truncated completion must never be
    cache.set() -- it must fail before the caller gets a (content, usage) tuple back."""
    asst = _make_assistant()
    asst._cache = MagicMock()
    asst._cache.compute_key.return_value = "some-key"
    asst._cache.get.return_value = None  # cache miss -> live call

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        '{"predicted_component": "kubectl", "similar_issues": [', finish_reason="length"
    )
    signals = {
        "prompt": "triage this",
        "classifier_top3": [{"label": "kubectl", "confidence": 0.5}],
        "lo_days": 1.0,
        "hi_days": 30.0,
        "resolution_bucket": "days",
        "resolution_conf_pct": 33.0,
    }
    with patch("groq.Groq", return_value=mock_client):
        with pytest.raises(TruncatedCompletionError):
            asst._call_llm_verbose(signals)
    asst._cache.set.assert_not_called()


# ---------------------------------------------------------------------------
# Structured output (A4)
# ---------------------------------------------------------------------------


def test_structured_output_sends_response_format_when_enabled():
    asst = _make_assistant()
    asst.use_structured_output = True
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(
        '{"predicted_component": "kubectl"}', finish_reason="stop", completion_tokens=50
    )
    with patch("groq.Groq", return_value=mock_client):
        content, usage = asst._groq_completion([{"role": "user", "content": "x"}])
    _, call_kwargs = mock_client.chat.completions.create.call_args
    assert "response_format" in call_kwargs
    assert call_kwargs["response_format"]["json_schema"]["strict"] is True
    assert usage["structured_output"] is True


def test_structured_output_falls_back_to_regex_extract_on_rejection():
    """Groq rejects response_format (400, schema issue) -> falls back to the classic
    unconstrained call for the rest of this assistant's lifetime, not a one-shot retry."""
    import groq

    asst = _make_assistant()
    asst.use_structured_output = True

    reject_response = MagicMock()
    reject_response.status_code = 400
    schema_error = groq.APIStatusError(
        message="invalid JSON schema for response_format: 'TriagePlan': bad schema",
        response=reject_response,
        body=None,
    )
    ok_response = _mock_groq_response(
        '{"predicted_component": "kubectl"}', finish_reason="stop", completion_tokens=50
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [schema_error, ok_response]

    with patch("groq.Groq", return_value=mock_client):
        content, usage = asst._groq_completion([{"role": "user", "content": "x"}])

    assert content == '{"predicted_component": "kubectl"}'
    assert asst.use_structured_output is False  # disabled for the rest of the session
    assert mock_client.chat.completions.create.call_count == 2
    second_call_kwargs = mock_client.chat.completions.create.call_args_list[1].kwargs
    assert "response_format" not in second_call_kwargs
