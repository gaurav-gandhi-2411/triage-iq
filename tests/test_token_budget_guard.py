"""Unit tests for the per-request prompt-token budget guard (2026-08-28, Part A/B).

Live-measured against the full 64-issue eval set: with a FIXED max_tokens=2048, 13/64
issues (20.3%) would be rejected outright by Groq's TPM preflight (413) before ever
reaching the model. These tests exercise the fix -- a dynamically-sized max_tokens
computed from the actual estimated prompt size, input truncation when that still isn't
enough, and a clean degrade (never calling Groq at all) as the last resort -- against the
real _call_llm_verbose method, with only the Groq client itself mocked.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from triage_iq.models.triage import (
    _GROQ_TPM_LIMIT,
    _MIN_VIABLE_COMPLETION_TOKENS,
    _PROMPT_SIZE_SAFETY_MARGIN,
    TriageAssistant,
)


def _make_assistant(max_tokens: int = 2048) -> TriageAssistant:
    asst = TriageAssistant.__new__(TriageAssistant)
    asst.repo = "microsoft/vscode"
    asst.model = "openai/gpt-oss-20b"
    asst.temperature = 0.0
    asst.max_tokens = max_tokens
    asst.seed = 42
    asst._groq_key = "test-key"
    asst.use_structured_output = False
    asst._cache = None
    return asst


def _mock_groq_response(content: str, finish_reason: str = "stop", completion_tokens: int = 200,
                         prompt_tokens: int = 500):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content), finish_reason=finish_reason)]
    resp.usage = MagicMock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return resp


_VALID_PLAN_JSON = (
    '{"predicted_component": "editor", "component_confidence": 0.7, '
    '"expected_resolution_summary": "A few days.", "expected_resolution_lower_days": 1.0, '
    '"expected_resolution_upper_days": 5.0, "priority_guess": "medium", '
    '"priority_rationale": "Reasonable.", "suggested_assignee_class": "editor team", '
    '"suggested_next_steps": ["Reproduce."], "triage_summary": "A minimal valid plan."}'
)


def _base_signals(body: str = "Short body.") -> dict:
    return {
        "prompt": f"Repository: microsoft/vscode\n\n--- ISSUE ---\nTitle: X\nBody:\n{body}\n",
        "classifier_top3": [{"label": "editor", "confidence": 0.5}],
        "similar_raw": [],
        "pred_days": 2.0,
        "lo_days": 1.0,
        "hi_days": 5.0,
        "resolution_bucket": "days",
        "resolution_conf_pct": 40.0,
        "_title": "X",
        "_body": body,
        "_include_bucket": False,
        "_number": 42,
    }


def test_dynamic_max_tokens_shrinks_below_self_max_tokens_for_a_large_prompt():
    """A long-but-not-pathological issue body should push the real max_tokens sent to
    Groq below self.max_tokens, not send self.max_tokens unmodified and risk a 413."""
    asst = _make_assistant(max_tokens=2048)
    # ~100 repeats lands real estimated prompt tokens around 5,900-6,000 -- inside the
    # observed 64-issue eval-set range, enough to push dynamic_max_tokens below 2048 but
    # comfortably above the truncation floor. Not an extreme/unrealistic body.
    long_body = "Reproduction steps and stack trace. " * 100
    signals = _base_signals(body=long_body)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(_VALID_PLAN_JSON)
    with patch("groq.Groq", return_value=mock_client):
        asst._call_llm_verbose(signals)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] < 2048
    assert call_kwargs["max_tokens"] >= _MIN_VIABLE_COMPLETION_TOKENS


def test_dynamic_max_tokens_unchanged_for_a_short_prompt():
    """The common case: a normal-length prompt leaves plenty of room, so the dynamic
    budget should equal self.max_tokens exactly, not silently shrink for no reason."""
    asst = _make_assistant(max_tokens=2048)
    signals = _base_signals(body="A short, ordinary issue body.")

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(_VALID_PLAN_JSON)
    with patch("groq.Groq", return_value=mock_client):
        asst._call_llm_verbose(signals)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["max_tokens"] == 2048


def test_input_truncation_triggers_and_shrinks_body_when_prompt_alone_leaves_no_budget():
    """When even a small max_tokens ask can't fit, the guard must shrink the issue body
    (not the retrieved-issue payloads) before ever considering a full degrade."""
    asst = _make_assistant(max_tokens=2048)
    # Body long enough that dynamic_max_tokens < floor even before considering
    # max_tokens=2048 -- forces the truncation branch deterministically.
    huge_body = "Word " * 4000
    signals = _base_signals(body=huge_body)

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response(_VALID_PLAN_JSON)
    with patch("groq.Groq", return_value=mock_client):
        plan, raw, usage, llm_status, cache_hit = asst._call_llm_verbose(signals)

    assert llm_status == "ok"  # truncating input, not degrading -- Groq still answers normally
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    sent_user_content = call_kwargs["messages"][-1]["content"]
    assert len(sent_user_content) < len(huge_body)


def test_degrades_without_calling_groq_when_budget_exhausted_even_after_truncation():
    """Last-resort path: if even a fully truncated body still can't fit a viable
    completion budget, skip Groq entirely rather than send a request very likely to
    413 or truncate mid-completion."""
    asst = _make_assistant(max_tokens=2048)
    signals = _base_signals(body="Doesn't matter -- estimator is patched below.")

    mock_client = MagicMock()

    with (
        patch("groq.Groq", return_value=mock_client),
        patch(
            "triage_iq.models.triage._estimate_prompt_tokens",
            return_value=_GROQ_TPM_LIMIT - _PROMPT_SIZE_SAFETY_MARGIN,  # leaves 0 for completion
        ),
    ):
        plan, raw, usage, llm_status, cache_hit = asst._call_llm_verbose(signals)

    mock_client.chat.completions.create.assert_not_called()
    assert llm_status == "degraded_insufficient_budget"
    assert cache_hit is False
    assert raw == ""
    assert "manual review" in plan.triage_summary.lower()
