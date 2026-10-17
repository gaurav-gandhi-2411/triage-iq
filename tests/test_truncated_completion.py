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

from triage_iq.models.triage import SchemaValidationError, TriageAssistant, TruncatedCompletionError


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
    with patch("groq.Groq", return_value=mock_client), pytest.raises(TruncatedCompletionError) as exc_info:
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
    cache.set() -- it must fail before a (content, usage) tuple is ever cached.

    2026-08-28 (Part B3): _call_llm_verbose no longer lets TruncatedCompletionError
    propagate to its caller -- it catches it and degrades cleanly to a signals-only
    fallback plan (llm_status="degraded_truncated"), since an unhandled exception here
    would otherwise reach app.py as a raw 500 instead of a clean degrade. The original
    cache-safety property (never cache.set() a truncated completion) still holds and is
    still asserted here.
    """
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
        "_title": "Something broke",
        "_body": "It broke.",
        "_include_bucket": False,
        "_number": 1,
    }
    with patch("groq.Groq", return_value=mock_client):
        plan, raw, usage, llm_status, cache_hit = asst._call_llm_verbose(signals)
    asst._cache.set.assert_not_called()
    assert llm_status == "degraded_truncated"
    assert cache_hit is False
    assert usage["finish_reason"] == "length"
    assert usage["completion_tokens"] == 1024
    assert "manual review" in plan.triage_summary.lower()


# ---------------------------------------------------------------------------
# SchemaValidationError (2026-09-03, ADR-0055 Part P1a)
#
# Groq's json_validate_failed 400 -- a syntactically-complete completion its OWN
# post-hoc schema validator rejected (missing required field or malformed key) --
# matched no exception handling before this fix and propagated to app.py's /triage
# handler as a live HTTP 500. These tests exercise the real _groq_completion/
# _call_llm_verbose methods against a mocked Groq client, mirroring the
# TruncatedCompletionError tests above.
# ---------------------------------------------------------------------------


def _mock_json_validate_failed_error(detail: str = "missing properties: 'triage_summary'"):
    import groq

    reject_response = MagicMock()
    reject_response.status_code = 400
    return groq.APIStatusError(
        message=f"Generated JSON does not match the expected schema: {detail}",
        response=reject_response,
        body={
            "error": {
                "message": f"Generated JSON does not match the expected schema. {detail}",
                "type": "invalid_request_error",
                "code": "json_validate_failed",
                "failed_generation": '{"predicted_component": "kubectl"}',
            }
        },
    )


def test_groq_completion_raises_schema_validation_error_on_json_validate_failed():
    asst = _make_assistant()
    asst.use_structured_output = True
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _mock_json_validate_failed_error()
    with patch("groq.Groq", return_value=mock_client), pytest.raises(SchemaValidationError) as exc_info:
        asst._groq_completion([{"role": "user", "content": "triage this"}])
    assert exc_info.value.groq_error_code == "json_validate_failed"


def test_schema_validation_error_degrades_cleanly_not_unhandled():
    """The defect this fix exists for: before this, json_validate_failed propagated
    unhandled all the way to app.py's broad except-Exception, returning HTTP 500 for
    every real request that hit it -- an outage generator hidden behind the already-
    dead retired model. Must now degrade to a clean fallback plan instead, exactly
    like TruncatedCompletionError does, and never reach cache.set()."""
    asst = _make_assistant()
    asst.use_structured_output = True
    asst._cache = MagicMock()
    asst._cache.compute_key.return_value = "some-key"
    asst._cache.get.return_value = None  # cache miss -> live call

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = _mock_json_validate_failed_error()
    signals = {
        "prompt": "triage this",
        "classifier_top3": [{"label": "kubectl", "confidence": 0.5}],
        "lo_days": 1.0,
        "hi_days": 30.0,
        "resolution_bucket": "days",
        "resolution_conf_pct": 33.0,
        "_title": "Something broke",
        "_body": "It broke.",
        "_include_bucket": False,
        "_number": 1,
    }
    with patch("groq.Groq", return_value=mock_client):
        plan, raw, usage, llm_status, cache_hit = asst._call_llm_verbose(signals)
    asst._cache.set.assert_not_called()
    assert llm_status == "degraded_schema_invalid"
    assert cache_hit is False
    assert usage["groq_error_code"] == "json_validate_failed"
    assert "manual review" in plan.triage_summary.lower()


def test_schema_validation_error_on_parse_retry_call_also_degrades():
    """2026-09-03 (ADR-0055 Part P1c audit): the parse-retry call (triggered when the
    FIRST completion isn't valid JSON at all) has its own try/except for
    TruncatedCompletionError -- but SchemaValidationError on THAT retry call was
    missing entirely until this fix, meaning it would have propagated uncaught: the
    exact P1a defect, just on the second call instead of the first. Sequence: first
    call returns non-JSON text (triggers the parse-retry path), second (retry) call
    hits json_validate_failed."""
    asst = _make_assistant()
    asst.use_structured_output = True
    asst._cache = MagicMock()
    asst._cache.compute_key.return_value = "some-key"
    asst._cache.get.return_value = None

    not_json_response = _mock_groq_response(
        "I cannot produce that output.", finish_reason="stop", completion_tokens=10
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [
        not_json_response,
        _mock_json_validate_failed_error(),
    ]
    signals = {
        "prompt": "triage this",
        "classifier_top3": [{"label": "kubectl", "confidence": 0.5}],
        "lo_days": 1.0,
        "hi_days": 30.0,
        "resolution_bucket": "days",
        "resolution_conf_pct": 33.0,
        "_title": "Something broke",
        "_body": "It broke.",
        "_include_bucket": False,
        "_number": 1,
    }
    with patch("groq.Groq", return_value=mock_client):
        plan, raw, usage, llm_status, cache_hit = asst._call_llm_verbose(signals)
    assert llm_status == "degraded_schema_invalid"
    assert mock_client.chat.completions.create.call_count == 2
    assert "manual review" in plan.triage_summary.lower()


def test_connection_error_retries_with_backoff_then_succeeds():
    """2026-09-03 (ADR-0055 Part P1c/5): found by the error-shape audit --
    APIConnectionError/APITimeoutError are not subclasses of APIStatusError or
    RateLimitError, so before this fix neither except clause caught them: a transient
    network blip propagated with ZERO retries, unlike every other case in this method.
    Must now retry with the same backoff schedule as RateLimitError."""
    import groq
    import httpx

    asst = _make_assistant()
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    conn_error = groq.APIConnectionError(request=req)
    ok_response = _mock_groq_response(
        '{"predicted_component": "kubectl"}', finish_reason="stop", completion_tokens=50
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [conn_error, conn_error, ok_response]

    with patch("groq.Groq", return_value=mock_client), patch("time.sleep"):
        content, usage = asst._groq_completion([{"role": "user", "content": "x"}])

    assert content == '{"predicted_component": "kubectl"}'
    assert mock_client.chat.completions.create.call_count == 3


def test_connection_error_raises_after_six_attempts():
    import groq
    import httpx

    asst = _make_assistant()
    req = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    conn_error = groq.APIConnectionError(request=req)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = conn_error

    with patch("groq.Groq", return_value=mock_client), patch("time.sleep"), pytest.raises(
        groq.APIConnectionError
    ):
        asst._groq_completion([{"role": "user", "content": "x"}])
    assert mock_client.chat.completions.create.call_count == 6


def test_response_format_rejection_not_misclassified_as_schema_validation_error():
    """Regression: json_validate_failed detection (checked first, via structured
    e.body access) must not swallow the DIFFERENT response_format-rejection 400 (the
    entire response_format is invalid/unsupported, not a per-completion schema miss)
    -- these need different remedies (degrade one completion vs. disable structured
    output for the assistant's lifetime) and must stay mutually exclusive."""
    import groq

    asst = _make_assistant()
    asst.use_structured_output = True

    reject_response = MagicMock()
    reject_response.status_code = 400
    schema_error = groq.APIStatusError(
        message="invalid JSON schema for response_format: 'TriagePlan': bad schema",
        response=reject_response,
        body=None,  # this rejection shape carries no structured body in practice
    )
    ok_response = _mock_groq_response(
        '{"predicted_component": "kubectl"}', finish_reason="stop", completion_tokens=50
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [schema_error, ok_response]

    with patch("groq.Groq", return_value=mock_client):
        content, usage = asst._groq_completion([{"role": "user", "content": "x"}])

    # Falls back to regex-extract, exactly as before -- not raised as SchemaValidationError.
    assert content == '{"predicted_component": "kubectl"}'
    assert asst.use_structured_output is False


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
