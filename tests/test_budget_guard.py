"""Unit tests for DailyBudgetGuard (Part B2, 2026-08-27 diagnostic session).

No real Firestore in these tests -- google-cloud-firestore isn't in requirements.txt yet
(deliberately: the guard fails open until the IAM grant + dependency are both in place, see
budget_guard.py's module docstring). Tests cover the fail-open contract itself, plus
NoOpBudgetGuard and DailyBudgetGuard against a fake in-memory client standing in for
Firestore's document/collection interface.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd

from triage_iq.models.budget_guard import DailyBudgetGuard, GroqBudgetExceeded, NoOpBudgetGuard
from triage_iq.models.triage import TriageAssistant


def _make_assistant() -> TriageAssistant:
    return TriageAssistant(
        repo="microsoft/vscode",
        classifier=MagicMock(),
        detector=MagicMock(),
        predictor=MagicMock(),
        train_df=pd.DataFrame(),
        groq_api_key="test-key-for-tests",
    )


def test_noop_guard_never_blocks():
    guard = NoOpBudgetGuard()
    assert guard.over_budget() is False
    guard.record(999_999)  # must not raise
    assert guard.over_budget() is False


def test_daily_budget_guard_fails_open_when_firestore_unavailable():
    """The exact scenario before the IAM grant exists: google-cloud-firestore not
    installed, or the client can't authenticate/reach the API. Must never block."""
    guard = DailyBudgetGuard(project="triageiq-prod-260812", limit_tokens=100)
    with patch.object(guard, "_get_client", return_value=None):
        assert guard.over_budget() is False
        guard.record(500)  # must not raise even though it can't actually write


def test_daily_budget_guard_over_budget_true_when_at_or_above_limit():
    fake_doc = MagicMock()
    fake_doc.exists = True
    fake_doc.get.return_value = 150

    fake_client = MagicMock()
    fake_client.collection.return_value.document.return_value.get.return_value = fake_doc

    guard = DailyBudgetGuard(project="p", limit_tokens=100)
    with patch.object(guard, "_get_client", return_value=fake_client):
        assert guard.over_budget() is True


def test_daily_budget_guard_under_budget_false_when_below_limit():
    fake_doc = MagicMock()
    fake_doc.exists = True
    fake_doc.get.return_value = 50

    fake_client = MagicMock()
    fake_client.collection.return_value.document.return_value.get.return_value = fake_doc

    guard = DailyBudgetGuard(project="p", limit_tokens=100)
    with patch.object(guard, "_get_client", return_value=fake_client):
        assert guard.over_budget() is False


def test_daily_budget_guard_no_doc_yet_means_under_budget():
    fake_doc = MagicMock()
    fake_doc.exists = False

    fake_client = MagicMock()
    fake_client.collection.return_value.document.return_value.get.return_value = fake_doc

    guard = DailyBudgetGuard(project="p", limit_tokens=100)
    with patch.object(guard, "_get_client", return_value=fake_client):
        assert guard.over_budget() is False


def test_daily_budget_guard_record_is_best_effort_on_write_failure():
    fake_client = MagicMock()
    fake_client.collection.return_value.document.return_value.set.side_effect = RuntimeError("boom")

    guard = DailyBudgetGuard(project="p", limit_tokens=100)
    with patch.object(guard, "_get_client", return_value=fake_client):
        guard.record(10)  # must not raise


def test_groq_completion_raises_budget_exceeded_without_calling_groq():
    """Part B2's whole point: skip the live call entirely when over budget."""
    assistant = _make_assistant()
    over_budget_guard = MagicMock()
    over_budget_guard.over_budget.return_value = True
    assistant._budget_guard = over_budget_guard

    mock_client = MagicMock()
    with patch("groq.Groq", return_value=mock_client):
        try:
            assistant._groq_completion([{"role": "user", "content": "hi"}])
            raise AssertionError("expected GroqBudgetExceeded")
        except GroqBudgetExceeded:
            pass
    mock_client.chat.completions.create.assert_not_called()
