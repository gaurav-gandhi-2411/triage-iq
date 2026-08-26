"""Unit tests for TriageAssistant's graceful-degradation wiring (triage.py:285-310).

Part A of the 2026-08-27 diagnostic session: a Groq-origin failure (auth/connection/
rate-limit-exhausted/5xx) must degrade triage_with_metadata() to a signals-only fallback
plan rather than raise; a programming/config error (bad request, 404 model-not-found, or
anything not Groq-shaped at all) must still raise. Isolates exactly the changed code path
by mocking _collect_signals and _call_llm_verbose -- the ML pipeline internals (classifier/
retrieval/resolution predictor) are exercised elsewhere and are irrelevant to this wiring.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from groq import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

from triage_iq.models.triage import TriageAssistant, _is_groq_unavailable


def _make_assistant() -> TriageAssistant:
    return TriageAssistant(
        repo="microsoft/vscode",
        classifier=MagicMock(),
        detector=MagicMock(),
        predictor=MagicMock(),
        train_df=pd.DataFrame(),
        groq_api_key="test-key-for-tests",
    )


def _fake_signals() -> dict:
    return {
        "prompt": "irrelevant",
        "classifier_top3": [{"label": "editor", "confidence": 0.8}],
        "similar_raw": [],
        "pred_days": 3.0,
        "lo_days": 1.0,
        "hi_days": 7.0,
        "resolution_bucket": "days",
        "resolution_conf_pct": 61.0,
        "_t_classify": 0.001,
        "_t_retrieve": 0.001,
        "_t_predict": 0.001,
    }


def _fake_response(status_code: int = 500):
    """Builds a mock httpx.Response sufficient for groq's exception constructors."""
    import httpx

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    return httpx.Response(status_code=status_code, request=request)


@pytest.mark.parametrize(
    "exc_factory",
    [
        lambda: AuthenticationError("401", response=_fake_response(status_code=401), body=None),
        lambda: APIConnectionError(request=httpx_request()),
        lambda: RateLimitError("429", response=_fake_response(status_code=429), body=None),
        lambda: InternalServerError("500", response=_fake_response(status_code=500), body=None),
    ],
    ids=["AuthenticationError", "APIConnectionError", "RateLimitError", "InternalServerError"],
)
def test_degrades_on_groq_unavailable_exceptions(exc_factory):
    assistant = _make_assistant()
    issue = pd.Series({"number": 42, "title": "t", "body_clean": "b"})
    with (
        patch.object(assistant, "_collect_signals", return_value=_fake_signals()),
        patch.object(assistant, "_call_llm_verbose", side_effect=exc_factory()),
    ):
        plan, meta = assistant.triage_with_metadata(issue)

    assert meta["llm_status"] == "unavailable"
    assert meta["llm_status_reason"] == type(exc_factory()).__name__
    # Signals-only content must still be real (from Systems 1-3, not placeholders)
    assert plan.predicted_component == "editor"
    assert plan.component_confidence == 0.8
    assert plan.expected_resolution_lower_days == 1.0
    assert plan.expected_resolution_upper_days == 7.0
    assert plan.resolution_bucket == "days"
    # No real LLM content
    assert plan.similar_issues == []
    assert plan.priority_guess == "medium"  # hardcoded fallback default, not a real judgment
    assert "Groq unavailable" in plan.triage_summary


@pytest.mark.parametrize(
    "exc",
    [
        ValueError("not a groq exception at all"),
        BadRequestError("400", response=_fake_response(status_code=400), body=None),
        NotFoundError("404 model not found", response=_fake_response(status_code=404), body=None),
    ],
    ids=["ValueError", "BadRequestError", "NotFoundError-model-deprecation"],
)
def test_does_not_degrade_on_non_groq_unavailable_exceptions(exc):
    """A programming/config error must still propagate -- including the exact 404
    model-not-found shape that caused the 2026-08-16 outage. Silently degrading that would
    hide a misconfigured model behind an always-200 response instead of failing loudly."""
    assistant = _make_assistant()
    issue = pd.Series({"number": 42, "title": "t", "body_clean": "b"})
    with (
        patch.object(assistant, "_collect_signals", return_value=_fake_signals()),
        patch.object(assistant, "_call_llm_verbose", side_effect=exc),
        pytest.raises(type(exc)),
    ):
        assistant.triage_with_metadata(issue)


def test_is_groq_unavailable_classification():
    assert _is_groq_unavailable(AuthenticationError("x", response=_fake_response(401), body=None))
    assert _is_groq_unavailable(RateLimitError("x", response=_fake_response(429), body=None))
    assert _is_groq_unavailable(InternalServerError("x", response=_fake_response(500), body=None))
    assert not _is_groq_unavailable(BadRequestError("x", response=_fake_response(400), body=None))
    assert not _is_groq_unavailable(NotFoundError("x", response=_fake_response(404), body=None))
    assert not _is_groq_unavailable(ValueError("unrelated"))


def test_is_tpd_error_classification():
    from triage_iq.models.triage import _is_tpd_error

    tpd_msg = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for ... on tokens "
        "per day (TPD): Limit 200000, Used 199500, Requested 900. ...'}}"
    )
    tpm_msg = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached for ... on tokens "
        "per minute (TPM): Limit 8000, Used 7950, Requested 900. ...'}}"
    )
    assert _is_tpd_error(RateLimitError(tpd_msg, response=_fake_response(429), body=None))
    assert not _is_tpd_error(RateLimitError(tpm_msg, response=_fake_response(429), body=None))


def test_groq_completion_does_not_retry_a_tpd_rate_limit():
    """Part B1: a daily-cap 429 must raise immediately (0 sleeps) -- burning the 6-attempt
    backoff (up to ~3.5 minutes) on a window that can't reopen until the day rolls over is
    pure added latency on every request until then."""
    assistant = _make_assistant()
    tpd_msg = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached ... on tokens per "
        "day (TPD): Limit 200000, Used 200000, Requested 500. ...'}}"
    )
    tpd_exc = RateLimitError(tpd_msg, response=_fake_response(429), body=None)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = tpd_exc
    with (
        patch("groq.Groq", return_value=mock_client),
        patch("time.sleep") as mock_sleep,
        pytest.raises(RateLimitError),
    ):
        assistant._groq_completion([{"role": "user", "content": "hi"}])
    mock_sleep.assert_not_called()
    assert mock_client.chat.completions.create.call_count == 1


def test_groq_completion_still_retries_a_tpm_rate_limit():
    """A per-minute 429 IS worth retrying -- the window reopens in under a minute."""
    assistant = _make_assistant()
    tpm_msg = (
        "Error code: 429 - {'error': {'message': 'Rate limit reached ... on tokens per "
        "minute (TPM): Limit 8000, Used 8000, Requested 500. ...'}}"
    )
    tpm_exc = RateLimitError(tpm_msg, response=_fake_response(429), body=None)

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = tpm_exc
    with (
        patch("groq.Groq", return_value=mock_client),
        patch("time.sleep") as mock_sleep,
        pytest.raises(RateLimitError),
    ):
        assistant._groq_completion([{"role": "user", "content": "hi"}])
    assert mock_sleep.call_count == 5  # attempts 0-4 sleep, attempt 5 raises
    assert mock_client.chat.completions.create.call_count == 6


def httpx_request():
    import httpx

    return httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
