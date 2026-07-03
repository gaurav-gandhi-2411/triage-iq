"""Integration tests for the FastAPI triage API.

Uses TestClient with a mocked ModelStore so no real models or Groq calls needed.
"""

import json as _json
import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from triage_iq.api.app import app, limiter
from triage_iq.api.loader import ModelStore
from triage_iq.models.triage import SimilarIssue, TriageAssistant, TriagePlan


def _fake_plan() -> TriagePlan:
    return TriagePlan(
        predicted_component="editor",
        component_confidence=0.87,
        similar_issues=[SimilarIssue(number=1234, similarity=0.91, relevance_note="same crash")],
        expected_resolution_summary="Likely fixable in 2–5 days",
        expected_resolution_lower_days=2.0,
        expected_resolution_upper_days=5.0,
        resolution_bucket="days",
        resolution_confidence_pct=61.0,
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
        "resolution_bucket": "days",
        "resolution_confidence_pct": 61.0,
        "llm_status": "ok",
    }


def _make_store() -> ModelStore:
    bundle = MagicMock()
    bundle.assistant.triage_with_metadata.return_value = (_fake_plan(), _fake_meta())
    store = MagicMock()
    store.repos = ["microsoft/vscode", "kubernetes/kubernetes"]
    store.start_time = time.monotonic() - 5.0
    store.get.return_value = bundle
    return store


@pytest.fixture(autouse=True)
def _limiter_disabled():
    """Reset settings cache and disable rate limiting for all tests via env var."""
    from triage_iq.config import get_settings
    get_settings.cache_clear()
    with patch.dict("os.environ", {"RATE_LIMIT_ENABLED": "false"}):
        yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    store = _make_store()
    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key-for-tests"}),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        yield c


def test_service_info(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "TriageIQ"
    assert body["version"] == "0.1.0"
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"
    assert "gaurav-gandhi-2411/triage-iq" in body["repository"]
    assert "microsoft/vscode" in body["supported_repos"]
    assert "kubernetes/kubernetes" in body["supported_repos"]


def test_health(client):
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
    assert body["resolution_bucket"] in ("hours", "days", "weeks", "months", "long")


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
    # System 2: similar issue retriever
    assert "similar_issues" in body
    assert isinstance(body["similar_issues"], list)
    # System 3 & 4: resolution predictor feeds LLM context, LLM returns these
    assert "expected_resolution_lower_days" in body
    assert "expected_resolution_upper_days" in body
    assert body["expected_resolution_lower_days"] >= 0
    assert body["expected_resolution_upper_days"] >= body["expected_resolution_lower_days"]
    # Supplemental bucket field (does not replace float fields; see ADR-0009 T2.7)
    assert body["resolution_bucket"] in ("hours", "days", "weeks", "months", "long")
    assert 0.0 <= body["resolution_confidence_pct"] <= 100.0
    # Request must include request_id and llm_status (end-to-end plumbing)
    assert "_request_id" in body
    assert "_llm_status" in body
    assert body["_llm_status"] in ("ok", "parse_retry_succeeded", "parse_failure")


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


def test_triage_response_validates_against_schema(client):
    """/triage response must be valid against TriagePlan — locks the OpenAPI contract."""
    r = client.post("/triage", json={
        "repo": "microsoft/vscode",
        "title": "Editor crashes on paste",
        "body": "Reproducible every time I paste a 1 MB block of text.",
        "issue_number": 9999,
    })
    assert r.status_code == 200
    # model_validate must not raise — this is the contract assertion
    plan = TriagePlan.model_validate(r.json())
    assert plan.predicted_component == "editor"
    assert 0.0 <= plan.component_confidence <= 1.0
    assert plan.priority_guess in ("low", "medium", "high")
    assert len(plan.suggested_next_steps) >= 1


# ---------------------------------------------------------------------------
# TriageAssistant unit tests — LLM parse robustness
# ---------------------------------------------------------------------------

_VALID_PLAN_JSON = _json.dumps({
    "predicted_component": "editor",
    "component_confidence": 0.85,
    "similar_issues": [],
    "expected_resolution_summary": "3 days typical",
    "expected_resolution_lower_days": 1.0,
    "expected_resolution_upper_days": 5.0,
    "resolution_bucket": "days",
    "resolution_confidence_pct": 61.0,
    "priority_guess": "medium",
    "priority_rationale": "Standard bug",
    "suggested_assignee_class": "editor team",
    "suggested_next_steps": ["Reproduce locally"],
    "triage_summary": "Editor bug",
})

_MINIMAL_SIGNALS = {
    "prompt": "test",
    "classifier_top3": [{"label": "editor", "confidence": 0.85}],
    "similar_raw": [],
    "pred_days": 3.0,
    "lo_days": 1.0,
    "hi_days": 5.0,
    "resolution_bucket": "days",
    "resolution_conf_pct": 61.0,
    "_t_classify": 0.0,
    "_t_retrieve": 0.0,
    "_t_predict": 0.0,
}


def _make_assistant() -> TriageAssistant:
    asst = TriageAssistant.__new__(TriageAssistant)
    asst.repo = "microsoft/vscode"
    asst.model = "test"
    asst.temperature = 0.0
    asst.max_tokens = 1024
    asst._groq_key = "test-key"
    asst.classifier = MagicMock()
    asst.detector = MagicMock()
    asst.predictor = MagicMock()
    asst.train_df = MagicMock()
    return asst


def test_triage_handles_groq_preamble():
    """Groq response with prose before JSON: parsed on first attempt (llm_status=ok)."""
    asst = _make_assistant()
    raw = f"Here you go:\n{_VALID_PLAN_JSON}"
    with patch.object(asst, "_groq_completion", return_value=(raw, {})):
        plan, _, _, status, _ = asst._call_llm_verbose(_MINIMAL_SIGNALS)
    assert plan.predicted_component == "editor"
    assert status == "ok"


def test_triage_handles_groq_garbage():
    """Groq returns no JSON at all: retry also fails, fallback plan returned (llm_status=parse_failure)."""
    asst = _make_assistant()
    with patch.object(asst, "_groq_completion", return_value=("I cannot help with that.", {})):
        plan, _, _, status, _ = asst._call_llm_verbose(_MINIMAL_SIGNALS)
    assert status == "parse_failure"
    assert plan.predicted_component == "editor"   # from classifier_top3 fallback
    assert plan.priority_guess == "medium"
    assert len(plan.suggested_next_steps) >= 1


# ---------------------------------------------------------------------------
# PR 1 hardening tests
# ---------------------------------------------------------------------------

def test_rate_limiting():
    """11th request from the same IP within one hour must return 429."""
    from triage_iq.config import get_settings
    store = _make_store()
    payload = {
        "repo": "microsoft/vscode",
        "title": "Editor crashes on paste",
        "body": "Reproduces every time.",
        "issue_number": 1,
    }
    get_settings.cache_clear()
    limiter._storage.reset()
    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key-for-tests", "RATE_LIMIT_ENABLED": "true"}),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        for i in range(10):
            r = c.post("/triage", json=payload)
            assert r.status_code == 200, f"Request {i + 1} returned {r.status_code}"
        r = c.post("/triage", json=payload)
        assert r.status_code == 429
        assert r.json() == {"detail": "Too many requests"}


def test_missing_groq_key_raises_at_startup():
    """Settings must raise ValidationError when GROQ_API_KEY is absent or empty."""
    from pydantic import ValidationError

    from triage_iq.config import Settings, get_settings
    get_settings.cache_clear()
    with patch.dict("os.environ", {"GROQ_API_KEY": ""}), pytest.raises(ValidationError):
        Settings()


# ---------------------------------------------------------------------------
# PR 2 — Settings, JSON logging, prompt calibration
# ---------------------------------------------------------------------------

def test_settings_loads_from_env():
    """Settings correctly reads all fields from environment variables."""
    from triage_iq.config import Settings, get_settings
    get_settings.cache_clear()
    env = {
        "GROQ_API_KEY": "sk-test-key-123",
        "RATE_LIMIT_ENABLED": "false",
        "LOG_LEVEL": "DEBUG",
        "ENVIRONMENT": "dev",
        "PORT": "9090",
    }
    with patch.dict("os.environ", env):
        s = Settings()
    assert s.groq_api_key.get_secret_value() == "sk-test-key-123"
    assert s.rate_limit_enabled is False
    assert s.log_level == "DEBUG"
    assert s.environment == "dev"
    assert s.port == 9090


def test_settings_fails_without_groq_key():
    """Settings raises ValidationError when GROQ_API_KEY is empty."""
    from pydantic import ValidationError

    from triage_iq.config import Settings, get_settings
    get_settings.cache_clear()
    # Empty string overrides any .env file value (env vars take priority)
    with patch.dict("os.environ", {"GROQ_API_KEY": ""}), pytest.raises(ValidationError):
        Settings()


def test_log_request_emits_valid_json(capsys):
    """_log_request must emit a single parseable JSON line to stdout."""
    import json

    from triage_iq.api.app import _log_request
    _log_request(
        endpoint="/triage",
        status="success",
        repo="microsoft/vscode",
        total_latency_ms=123.4,
    )
    captured = capsys.readouterr()
    line = captured.out.strip()
    assert line, "Expected JSON output on stdout"
    data = json.loads(line)
    assert data["severity"] == "INFO"
    assert data["log_type"] == "access"
    assert data["endpoint"] == "/triage"
    assert data["status"] == "success"
    assert data["repo"] == "microsoft/vscode"
    assert "timestamp" in data
    assert "message" in data


def test_priority_calibration_in_prompt():
    """Prompt must contain the PRIORITY GUIDELINES block with all three calibration rules."""
    from triage_iq.prompts.triage_prompt import SYSTEM_PROMPT
    assert "PRIORITY GUIDELINES" in SYSTEM_PROMPT
    assert "low — cosmetic or non-blocking" in SYSTEM_PROMPT
    assert "medium — reproducible regression with a workaround" in SYSTEM_PROMPT
    assert "high — crash, data loss, auth failure" in SYSTEM_PROMPT
    assert "default to medium" in SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# PR 3 — Prometheus metrics
# ---------------------------------------------------------------------------

def test_metrics_endpoint_requires_token():
    """GET /metrics must return 401 without the correct Bearer token."""
    from triage_iq.config import get_settings
    store = _make_store()
    get_settings.cache_clear()
    with (
        patch.dict("os.environ", {
            "GROQ_API_KEY": "test-key-for-tests",
            "METRICS_TOKEN": "secret-scrape-token",
        }),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        assert c.get("/metrics").status_code == 401
        assert c.get("/metrics", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_metrics_endpoint_with_token():
    """GET /metrics with the correct Bearer token must return 200 and Prometheus text format."""
    from triage_iq.config import get_settings
    store = _make_store()
    get_settings.cache_clear()
    with (
        patch.dict("os.environ", {
            "GROQ_API_KEY": "test-key-for-tests",
            "METRICS_TOKEN": "secret-scrape-token",
        }),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        r = c.get("/metrics", headers={"Authorization": "Bearer secret-scrape-token"})
        assert r.status_code == 200
        assert "text/plain" in r.headers["content-type"]
        assert b"# HELP" in r.content


def test_triage_increments_counters(client):
    """Successful /triage call must increment triage_requests_total and triage_latency_seconds."""
    from prometheus_client import REGISTRY
    before = REGISTRY.get_sample_value(
        "triage_requests_total",
        {"repo": "microsoft/vscode", "status": "success"},
    ) or 0.0
    r = client.post("/triage", json={
        "repo": "microsoft/vscode",
        "title": "Editor crashes on paste",
        "body": "Reproducible every time.",
    })
    assert r.status_code == 200
    after = REGISTRY.get_sample_value(
        "triage_requests_total",
        {"repo": "microsoft/vscode", "status": "success"},
    ) or 0.0
    assert after == before + 1.0


def test_metrics_not_rate_limited():
    """GET /metrics must never return 429, even when rate limiting is enabled."""
    from triage_iq.config import get_settings
    store = _make_store()
    get_settings.cache_clear()
    limiter._storage.reset()
    with (
        patch.dict("os.environ", {
            "GROQ_API_KEY": "test-key-for-tests",
            "ENVIRONMENT": "dev",
            "RATE_LIMIT_ENABLED": "true",
        }),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        for i in range(20):
            r = c.get("/metrics")
            assert r.status_code != 429, f"Request {i + 1} was unexpectedly rate-limited"


def test_metrics_token_strips_whitespace():
    """METRICS_TOKEN with trailing \\r\\n must still accept a clean bearer token."""
    from triage_iq.config import get_settings
    store = _make_store()
    get_settings.cache_clear()
    with (
        patch.dict("os.environ", {
            "GROQ_API_KEY": "test-key-for-tests",
            "METRICS_TOKEN": "abc123\r\n",
        }),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        r = c.get("/metrics", headers={"Authorization": "Bearer abc123"})
        assert r.status_code == 200


def test_metrics_disabled_in_prod_without_token():
    """GET /metrics in prod with no METRICS_TOKEN must return 503 (fail-closed)."""
    from triage_iq.config import get_settings
    store = _make_store()
    get_settings.cache_clear()
    with (
        patch.dict("os.environ", {
            "GROQ_API_KEY": "test-key-for-tests",
            "ENVIRONMENT": "prod",
            "METRICS_TOKEN": "",  # explicitly absent
        }),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        r = c.get("/metrics")
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# PR 4 — parse retry + prompt unit tests
# ---------------------------------------------------------------------------

def test_triage_parse_retry_succeeded():
    """First Groq response is garbage; second is valid JSON: llm_status=parse_retry_succeeded."""
    asst = _make_assistant()
    responses = [
        ("not json at all %%", {}),
        (_VALID_PLAN_JSON, {"prompt_tokens": 100, "completion_tokens": 50}),
    ]
    with patch.object(asst, "_groq_completion", side_effect=responses):
        plan, _, _, status, _ = asst._call_llm_verbose(_MINIMAL_SIGNALS)
    assert status == "parse_retry_succeeded"
    assert plan.predicted_component == "editor"
    assert plan.priority_guess == "medium"


def test_triage_corrupted_primary_cache_uses_retry_cache_not_live_call():
    """Primary cache entry fails schema validation (e.g. malformed LLM JSON) but a
    valid retry-prompt entry is already cached from a prior recovery. The retry
    cache must be used — no live Groq call, since a replay-only cache (CI) has no
    real credentials to make one."""
    asst = _make_assistant()
    corrupted_content = _json.dumps({
        "predicted_component": "editor",
        "similar_issues": [{"number": None, "similarity": 0.5, "relevance_note": "x"}],
    })

    def fake_compute_key(provider, model, messages, temp, max_tok):
        is_retry = any(
            m.get("role") == "user" and "not valid JSON" in m.get("content", "")
            for m in messages
        )
        return "retry-key" if is_retry else "primary-key"

    def fake_get(key):
        if key == "primary-key":
            return {"content": corrupted_content, "usage": {}}
        if key == "retry-key":
            return {"content": _VALID_PLAN_JSON, "usage": {"prompt_tokens": 10, "completion_tokens": 5}}
        return None

    cache = MagicMock()
    cache.compute_key.side_effect = fake_compute_key
    cache.get.side_effect = fake_get
    asst._cache = cache

    with patch.object(asst, "_groq_completion") as mock_live:
        plan, _, _, status, cache_hit = asst._call_llm_verbose(_MINIMAL_SIGNALS)

    mock_live.assert_not_called()
    assert status == "parse_retry_succeeded"
    assert cache_hit is True
    assert plan.predicted_component == "editor"


def test_build_triage_prompt_contains_signals():
    """build_triage_prompt must embed all system signals in the returned string."""
    from triage_iq.prompts.triage_prompt import build_triage_prompt

    prompt = build_triage_prompt(
        issue_title="Editor crashes on paste",
        issue_body="Reproducible every time.",
        classifier_top3=[
            {"label": "editor", "confidence": 0.85},
            {"label": "workbench", "confidence": 0.10},
        ],
        similar_issues=[
            {"number": 1234, "score": 0.92, "text": "Same crash on paste with large text blocks."},
        ],
        resolution_point_days=3.0,
        resolution_lower_days=1.0,
        resolution_upper_days=7.0,
        repo="microsoft/vscode",
    )
    assert "microsoft/vscode" in prompt
    assert "Editor crashes on paste" in prompt
    assert "editor" in prompt
    assert "0.850" in prompt
    assert "#1234" in prompt
    assert "3.0 days" in prompt
    assert "[1.0d, 7.0d]" in prompt


# ---------------------------------------------------------------------------
# Conformal interval — ConformalIntervalResult + /triage injection
# ---------------------------------------------------------------------------

def test_conformal_interval_result_validates():
    """ConformalIntervalResult rejects out-of-range coverage values."""
    from pydantic import ValidationError

    from triage_iq.models.triage import ConformalIntervalResult

    # Valid construction
    r = ConformalIntervalResult(
        lower_days=1.0,
        upper_days=10.0,
        target_coverage=0.80,
        empirical_coverage=0.766,
        coverage_ci95_lower=0.740,
        coverage_ci95_upper=0.791,
    )
    assert r.lower_days == 1.0
    assert r.upper_days == 10.0
    assert r.target_coverage == 0.80

    # Coverage > 1.0 must fail
    with pytest.raises(ValidationError):
        ConformalIntervalResult(
            lower_days=0.0,
            upper_days=5.0,
            target_coverage=1.5,  # invalid
            empirical_coverage=0.8,
            coverage_ci95_lower=0.75,
            coverage_ci95_upper=0.85,
        )

    # lower_days < 0 must fail
    with pytest.raises(ValidationError):
        ConformalIntervalResult(
            lower_days=-1.0,  # invalid
            upper_days=5.0,
            target_coverage=0.80,
            empirical_coverage=0.8,
            coverage_ci95_lower=0.75,
            coverage_ci95_upper=0.85,
        )


def test_triage_plan_conformal_field_defaults_none():
    """resolution_interval_conformal defaults to None — existing plan construction unaffected."""
    plan = TriagePlan(
        predicted_component="editor",
        component_confidence=0.87,
        similar_issues=[],
        expected_resolution_summary="3 days",
        expected_resolution_lower_days=2.0,
        expected_resolution_upper_days=5.0,
        resolution_bucket="days",
        resolution_confidence_pct=61.0,
        priority_guess="medium",
        priority_rationale="Standard bug",
        suggested_assignee_class="editor team",
        suggested_next_steps=["Investigate"],
        triage_summary="Editor bug",
    )
    assert plan.resolution_interval_conformal is None


def test_triage_injects_conformal_when_adjustment_present():
    """When conformal_adjustments has an entry for the repo, /triage response includes the interval."""
    from triage_iq.models.triage import ConformalIntervalResult

    _adj = {
        "q_adjustment_hours": 0.2835,
        "target_coverage": 0.80,
        "empirical_coverage": 0.766,
        "coverage_ci95_lower": 0.740,
        "coverage_ci95_upper": 0.791,
    }

    store = _make_store()
    store.conformal_adjustments = {"microsoft/vscode": _adj}

    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key-for-tests"}),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        r = c.post("/triage", json={
            "repo": "microsoft/vscode",
            "title": "Editor crashes on paste",
            "body": "Reproducible every time I paste a 1 MB block.",
            "issue_number": 9999,
        })
    assert r.status_code == 200
    body = r.json()
    ci = body.get("resolution_interval_conformal")
    assert ci is not None, "Expected conformal interval to be populated"
    assert ci["target_coverage"] == 0.80
    assert ci["empirical_coverage"] == pytest.approx(0.766)
    assert ci["lower_days"] >= 0.0
    assert ci["upper_days"] >= ci["lower_days"]
    # Validate via Pydantic to lock the schema
    parsed = ConformalIntervalResult.model_validate(ci)
    assert 0.0 <= parsed.empirical_coverage <= 1.0


def test_triage_conformal_none_when_no_adjustment():
    """When conformal_adjustments is empty, resolution_interval_conformal is None."""
    store = _make_store()
    store.conformal_adjustments = {}  # no adjustments loaded

    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key-for-tests"}),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        TestClient(app) as c,
    ):
        r = c.post("/triage", json={
            "repo": "microsoft/vscode",
            "title": "Editor crash",
            "body": "Crash on startup.",
        })
    assert r.status_code == 200
    assert r.json().get("resolution_interval_conformal") is None


def test_model_store_conformal_adjustments_attribute():
    """ModelStore exposes conformal_adjustments dict on construction."""
    from triage_iq.api.loader import ModelStore, RepoBundle

    bundle = MagicMock(spec=RepoBundle)
    store_empty = ModelStore({"k8s": bundle}, start_time=0.0)
    assert store_empty.conformal_adjustments == {}

    adj = {"kubernetes/kubernetes": {"q_adjustment_hours": 0.28}}
    store_with = ModelStore({"k8s": bundle}, start_time=0.0, conformal_adjustments=adj)
    assert store_with.conformal_adjustments["kubernetes/kubernetes"]["q_adjustment_hours"] == 0.28


def test_load_conformal_adjustments_missing_file(tmp_path):
    """_load_conformal_adjustments returns empty dict and logs warning when file absent."""
    from triage_iq.api.loader import _load_conformal_adjustments

    result = _load_conformal_adjustments(tmp_path)
    assert result == {}


def test_load_conformal_adjustments_parses_json(tmp_path):
    """_load_conformal_adjustments correctly parses vscode (nested) and k8s (flat) entries."""
    import json

    from triage_iq.api.loader import _load_conformal_adjustments

    payload = {
        "target_coverage": 0.80,
        "repos": {
            "kubernetes/kubernetes": {
                "split": "30_70",
                "q_adjustment_hours": 0.2835,
                "empirical_test_coverage": 0.7664,
                "coverage_ci95_lower": 0.7399,
                "coverage_ci95_upper": 0.791,
            },
            "microsoft/vscode": {
                "40_60": {
                    "q_adjustment_hours": 1.2542,
                    "empirical_test_coverage": 0.7405,
                    "coverage_ci95_lower": 0.6936,
                    "coverage_ci95_upper": 0.7826,
                }
            },
        },
    }
    (tmp_path / "cqr_conformal_adjustments.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    result = _load_conformal_adjustments(tmp_path)

    assert set(result.keys()) == {"kubernetes/kubernetes", "microsoft/vscode"}
    k8s = result["kubernetes/kubernetes"]
    assert k8s["q_adjustment_hours"] == pytest.approx(0.2835)
    assert k8s["target_coverage"] == pytest.approx(0.80)
    assert k8s["empirical_coverage"] == pytest.approx(0.7664)

    vscode = result["microsoft/vscode"]
    assert vscode["q_adjustment_hours"] == pytest.approx(1.2542)
    assert vscode["empirical_coverage"] == pytest.approx(0.7405)
