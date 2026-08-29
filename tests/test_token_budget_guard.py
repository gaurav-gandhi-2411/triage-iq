"""Unit tests for the per-request prompt-token budget guard (2026-08-28, Part A/B).

Live-measured against the full 64-issue eval set: with a FIXED max_tokens=2048, 37/64
issues (57.8%) would be rejected outright by Groq's TPM preflight (413) before ever
reaching the model. (Corrected 2026-08-29: originally recorded here as 13/64 -- that
figure applied the 8000 ceiling against prompt+max_tokens alone and dropped the 100-token
_PROMPT_SIZE_SAFETY_MARGIN from the comparison, understating the real rejection rate by
nearly 3x. test_documented_413_rate_matches_guard_formula below pins the figure against
the guard's own formula and constants -- not a hand-copied number -- so this docstring
cannot silently drift from the code again.) These tests exercise the fix -- a
dynamically-sized max_tokens computed from the actual estimated prompt size, input
truncation when that still isn't enough, and a clean degrade (never calling Groq at all)
as the last resort -- against the real _call_llm_verbose method, with only the Groq
client itself mocked.
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


# Frozen snapshot of _estimate_prompt_tokens(messages) for all 64 eval/eval_set.jsonl issues,
# measured 2026-08-29 against the real classifier/retriever/predictor pipeline (not mocked)
# with TRIAGE_PROMPT_INCLUDE_ATTRIBUTION=1 (current default prompt: SYSTEM_PROMPT_PROSE +
# build_few_shot_examples(), 3 examples). Re-freeze this list if SYSTEM_PROMPT_PROSE,
# _SCHEMA_BLOCK, build_few_shot_examples(), or build_triage_prompt()'s user-turn template
# change -- it is a snapshot of THIS prompt's token distribution, not a universal constant.
_EVAL_SET_PROMPT_TOKENS_2026_08_29 = [
    5902, 5662, 5795, 5918, 6183, 5750, 5897, 5739, 5969, 5948, 5961, 6024, 5970, 5942,
    5912, 5820, 5896, 5781, 5756, 5765, 5763, 5927, 6018, 5850, 5894, 5741, 5806, 5994,
    5917, 5708, 5835, 5894, 5820, 5821, 5938, 5873, 5799, 5920, 5809, 5811, 5754, 6091,
    5798, 6121, 5768, 5714, 5917, 5833, 5798, 5763, 5854, 5914, 5966, 6031, 5964, 5939,
    5860, 6075, 5887, 5905, 5939, 5875, 5728, 5872,
]


def test_documented_413_rate_matches_guard_formula():
    """Pin the "37/64 (57.8%)" figure quoted in this module's and triage.py's docstrings
    to the guard's ACTUAL rejection formula, applied to a frozen real-measurement snapshot
    -- not a hand-computed number that a docstring can silently drift away from.

    The bug this guards against: the number first recorded here (13/64) applied the 8000
    ceiling against prompt_tokens + max_tokens alone and dropped _PROMPT_SIZE_SAFETY_MARGIN
    from that comparison, even though the margin is a real, non-optional part of the
    guard's own dynamic_max_tokens formula (see _call_llm_verbose). This test uses that
    same formula, so a future edit to the margin or the ceiling automatically re-checks
    whether the documented percentage is still accurate.
    """
    assert len(_EVAL_SET_PROMPT_TOKENS_2026_08_29) == 64
    fixed_max_tokens = 2048
    would_413 = sum(
        1
        for prompt_tokens in _EVAL_SET_PROMPT_TOKENS_2026_08_29
        if prompt_tokens + fixed_max_tokens + _PROMPT_SIZE_SAFETY_MARGIN > _GROQ_TPM_LIMIT
    )
    assert would_413 == 37, (
        f"documented as 37/64 (57.8%) -- guard formula now gives {would_413}/64. "
        "Update the docstrings in triage.py and this file's module docstring to match, "
        "or re-freeze _EVAL_SET_PROMPT_TOKENS_2026_08_29 if the prompt itself changed."
    )
