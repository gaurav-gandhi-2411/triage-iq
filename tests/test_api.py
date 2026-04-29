"""Integration tests for the FastAPI triage API.

Uses TestClient with a mocked ModelStore so no real models or Groq calls needed.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from triage_iq.api.app import app
from triage_iq.api.loader import ModelStore, RepoBundle
from triage_iq.models.triage import SimilarIssue, TriagePlan


def _fake_plan() -> TriagePlan:
    return TriagePlan(
        predicted_component="editor",
        component_confidence=0.87,
        similar_issues=[SimilarIssue(number=1234, similarity=0.91, relevance_note="same crash")],
        expected_resolution_summary="Likely fixable in 2–5 days",
        expected_resolution_lower_days=2.0,
        expected_resolution_upper_days=5.0,
        priority_guess="medium",
        priority_rationale="No data loss; reproducible workaround exists",
        suggested_assignee_class="editor-core team",
        suggested_next_steps=["Reproduce locally", "Check recent editor diffs"],
        triage_summary="Editor crash on paste — medium priority",
    )


def _make_store() -> ModelStore:
    bundle = MagicMock()
    bundle.assistant.triage.return_value = _fake_plan()
    store = MagicMock()
    store.repos = ["microsoft/vscode", "kubernetes/kubernetes"]
    store.start_time = time.monotonic() - 5.0
    store.get.return_value = bundle
    return store


@pytest.fixture
def client():
    store = _make_store()
    with patch("triage_iq.api.app.ModelStore.load_all", return_value=store):
        with TestClient(app) as c:
            yield c


def test_health(client):
    with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}):
        r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "microsoft/vscode" in body["repos_loaded"]
    assert body["groq_key_present"] is True
    assert body["uptime_s"] >= 0


def test_triage_returns_plan(client):
    r = client.post("/triage", json={
        "repo": "microsoft/vscode",
        "title": "Editor crashes on paste",
        "body": "Reproduces every time I paste a 1 MB block of text.",
        "issue_number": 9999,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_component"] == "editor"
    assert 0.0 <= body["component_confidence"] <= 1.0
    assert isinstance(body["suggested_next_steps"], list)
    assert body["expected_resolution_lower_days"] <= body["expected_resolution_upper_days"]


def test_triage_invalid_repo(client):
    r = client.post("/triage", json={
        "repo": "unknown/repo",
        "title": "Bug",
    })
    assert r.status_code == 422


def test_triage_missing_title(client):
    r = client.post("/triage", json={
        "repo": "microsoft/vscode",
    })
    assert r.status_code == 422


def test_triage_propagates_assistant_error(client):
    app.state.store.get.return_value.assistant.triage.side_effect = RuntimeError("groq down")
    r = client.post("/triage", json={
        "repo": "microsoft/vscode",
        "title": "Some issue",
    })
    assert r.status_code == 500
