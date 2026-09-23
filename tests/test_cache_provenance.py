"""Part D3 (2026-08-27 diagnostic session): cassette/cache entries previously carried no
record of which model, account, or when they were recorded, making it structurally
impossible to tell after the fact whether a recording mixed entries from different Groq
accounts/tiers. TriageAssistant._cache_response now attaches that provenance to every
live cache write (both the eval CassettePlayer and the production LLMCache store the
dict as-is, so no change was needed in either backend)."""

from __future__ import annotations

import contextlib
import os
from unittest.mock import MagicMock, patch

import pandas as pd

from triage_iq.models.triage import TriageAssistant


def _make_assistant() -> TriageAssistant:
    return TriageAssistant(
        repo="microsoft/vscode",
        classifier=MagicMock(),
        detector=MagicMock(),
        predictor=MagicMock(),
        train_df=pd.DataFrame(),
        groq_api_key="test-key-for-tests",
        model="openai/gpt-oss-20b",
    )


def test_cache_response_includes_model_and_timestamp():
    assistant = _make_assistant()
    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("GROQ_ACCOUNT_LABEL", None)
        entry = assistant._cache_response("raw content", {"prompt_tokens": 10, "completion_tokens": 5})

    assert entry["content"] == "raw content"
    assert entry["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    assert entry["_provenance"]["model"] == "openai/gpt-oss-20b"
    assert entry["_provenance"]["account_label"] == "unknown"
    # ISO 8601 with a timezone offset, not a bare unqualified timestamp
    assert "T" in entry["_provenance"]["recorded_at"]
    assert entry["_provenance"]["recorded_at"].endswith("+00:00")


def test_cache_response_uses_account_label_when_set():
    assistant = _make_assistant()
    with patch.dict(os.environ, {"GROQ_ACCOUNT_LABEL": "triageiq-production"}):
        entry = assistant._cache_response("x", {})
    assert entry["_provenance"]["account_label"] == "triageiq-production"


def test_live_call_writes_provenance_to_cache():
    """End-to-end: a cache miss followed by a live call must pass a provenance-bearing
    dict to cache.set(), not the old bare {content, usage}."""
    assistant = _make_assistant()
    fake_cache = MagicMock()
    fake_cache.compute_key.return_value = "key123"
    fake_cache.get.return_value = None  # miss
    assistant._cache = fake_cache

    # only the cache.set() call on the primary path is under test here
    with (
        patch.object(assistant, "_groq_completion", return_value=("{}", {"prompt_tokens": 1, "completion_tokens": 1})),
        patch.object(assistant, "_parse_plan", side_effect=ValueError("force a stable early return path")),
        contextlib.suppress(Exception),
    ):
        assistant._call_llm_verbose({"prompt": "p"})

    assert fake_cache.set.called
    stored = fake_cache.set.call_args[0][-1]
    assert "_provenance" in stored
    assert stored["_provenance"]["model"] == "openai/gpt-oss-20b"
