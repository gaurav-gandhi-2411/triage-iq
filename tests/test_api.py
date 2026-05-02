"""Integration tests for the FastAPI triage API.

Uses TestClient with a mocked ModelStore so no real models or Groq calls needed.
"""

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from triage_iq.api.app import app
from triage_iq.api.loader import ModelStore
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


def _fake_meta() -> dict:
    return {
        "system1_latency_ms": 5.0,
        "system2_latency_ms": 80.0,
        "system3_latency_ms": 2.0,
        "system4_latency_ms": 900.0,
        "total_latency_ms": 990.0,
        "groq_tokens_prompt": 500,
        "groq_tokens_completion": 200,
        "estimated_cost_usd": 0.0001,
        "duplicate_count": 1,
        "predicted_resolution_days_p50": 3.5,
    }


def _make_store() -> ModelStore:
    bundle = MagicMock()
    bundle.assistant.triage_with_metadata.return_value = (_fake_plan(), _fake_meta())
    store = MagicMock()
    store.repos = ["microsoft/vscode", "kubernetes/kubernetes"]
    store.start_time = time.monotonic() - 5.0
    store.get.return_value = bundle
    return store


@pytest.fixture
def client():
    store = _make_store()
    with patch("triage_iq.api.app.ModelStore.load_all", return_value=store), TestClient(app) as c:
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


def test_triage_includes_resolution_prediction(client):
    """All 4 systems must produce output — resolution predictor must not silently degrade."""
    r = client.post("/triage", json={
        "repo": "microsoft/vscode",
        "title": "Extension host crashes on startup",
        "body": "Every time I open VS Code the extension host process terminates immediately.",
    })
    assert r.status_code == 200
    body = r.json()
    # System 1: component classifier
    assert "predicted_component" in body
    assert body["predicted_component"] != ""
    # System 2: duplicate detector
    assert "similar_issues" in body
    assert isinstance(body["similar_issues"], list)
    # System 3 & 4: resolution predictor feeds LLM context, LLM returns these
    assert "expected_resolution_lower_days" in body
    assert "expected_resolution_upper_days" in body
    assert body["expected_resolution_lower_days"] >= 0
    assert body["expected_resolution_upper_days"] >= body["expected_resolution_lower_days"]
    # Request must include request_id (end-to-end plumbing)
    assert "_request_id" in body


def test_triage_accepts_created_at(client):
    """created_at is optional; when provided it must not cause a 500."""
    r = client.post("/triage", json={
        "repo": "kubernetes/kubernetes",
        "title": "Pod stuck in Terminating",
        "body": "kubectl delete pod hangs indefinitely.",
        "created_at": "2026-04-01T12:00:00Z",
    })
    assert r.status_code == 200


def test_triage_without_created_at(client):
    """created_at omitted — predictor must default to now() and not return 500."""
    r = client.post("/triage", json={
        "repo": "microsoft/vscode",
        "title": "Settings sync broken",
        "body": "Settings do not sync across machines after the latest update.",
    })
    assert r.status_code == 200


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
    app.state.store.get.return_value.assistant.triage_with_metadata.side_effect = RuntimeError("groq down")
    r = client.post("/triage", json={
        "repo": "microsoft/vscode",
        "title": "Some issue",
    })
    assert r.status_code == 500
