"""TriageJudge has no default model (2026-09-24).

Its old default was model_config.JUDGE_MODEL, a Groq model retired 2026-08-16. A caller that
forgot to pass a model would have silently targeted a dead model and failed at the first API
call. It must fail at construction instead, and model_config must no longer expose the
retired constant for anything to pick up.
"""

from __future__ import annotations

import pytest

import triage_iq.model_config as model_config
from triage_iq.evaluation.triage_eval import TriageJudge


def test_judge_without_model_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="explicit model"):
        TriageJudge(provider="ollama")


def test_judge_with_explicit_model_constructs() -> None:
    judge = TriageJudge(model="qwen3:8b", provider="ollama")
    assert judge.model == "qwen3:8b"


def test_retired_groq_judge_constant_is_gone() -> None:
    assert not hasattr(model_config, "JUDGE_MODEL")
