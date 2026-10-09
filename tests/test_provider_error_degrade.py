"""Provider-side LLM failures degrade to a signals-only plan (HTTP 200) -- ADR-0063.

Network-free: every Groq call is a mocked client raising real groq exception classes, so the
real _groq_completion / _call_llm_verbose / triage_with_metadata / FastAPI handler run.
Production incident 2026-10-07 20:16-20:18 UTC: Groq 429 (TPD exhausted) -> POST /triage 500.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import groq
import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

from triage_iq.api.app import app
from triage_iq.models.triage import (
    PROVIDER_DEGRADED_STATUS,
    TriageAssistant,
    classify_provider_error,
)

_REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")

_TPD_MSG = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier "
    "`on_demand` on tokens per day (TPD): Limit 200000, Used 199990, Requested 5000. "
    "Please try again in 11m5s."
)
_TPM_MSG = (
    "Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute (TPM): "
    "Limit 8000, Used 7900, Requested 500. Please try again in 3.2s."
)


def _status_error(cls, status: int, message: str = "boom", code: str | None = None):
    body = {"error": {"message": message, "type": "t", "code": code}}
    return cls(message, response=httpx.Response(status, request=_REQ), body=body)


def _rate_limit(message: str = _TPD_MSG):
    return _status_error(groq.RateLimitError, 429, message, "rate_limit_exceeded")


def _make_assistant() -> TriageAssistant:
    asst = TriageAssistant.__new__(TriageAssistant)
    asst.repo = "microsoft/vscode"
    asst.model = "openai/gpt-oss-120b"
    asst.temperature = 0.0
    asst.max_tokens = 1024
    asst.seed = 42
    asst._groq_key = "test-key"
    asst.use_structured_output = False
    asst.enable_validated_override_rescue = False
    return asst


def _signals() -> dict:
    return {
        "prompt": "triage this",
        "classifier_top3": [
            {"label": "workbench", "confidence": 0.61},
            {"label": "editor", "confidence": 0.2},
        ],
        "similar_raw": [
            {"number": 11, "score": 0.9, "text": "a"},
            {"number": 12, "score": 0.8, "text": "b"},
        ],
        "pred_days": 4.0,
        "lo_days": 1.5,
        "hi_days": 20.0,
        "resolution_bucket": "days",
        "resolution_conf_pct": 55.0,
        # keys added by the train-median point estimate (ADR-0064); triage_with_metadata reads them
        "resolution_point_days": 4.0,
        "resolution_point_source": "model",
        "resolution_interval_basis": "model",
        "_t_classify": 0.01,
        "_t_retrieve": 0.01,
        "_t_predict": 0.01,
        "_title": "Something broke",
        "_body": "It broke.",
        "_include_bucket": False,
        "_number": 7,
    }


def _ok_response(content: str):
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content=content), finish_reason="stop")]
    resp.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
    return resp


def _run(asst: TriageAssistant, side_effect, cache=None):
    client = MagicMock()
    client.chat.completions.create.side_effect = side_effect
    if cache is not None:
        asst._cache = cache
    with patch("groq.Groq", return_value=client), patch("time.sleep") as sleep:
        out = asst._call_llm_verbose(_signals())
    return out, client, sleep


def _empty_cache() -> MagicMock:
    cache = MagicMock()
    cache.compute_key.return_value = "k"
    cache.get.return_value = None
    return cache


# ---------------------------------------------------------------------------
# Degradable classes -> degraded status + reason
# ---------------------------------------------------------------------------

DEGRADE_CASES = [
    pytest.param(lambda: _rate_limit(_TPD_MSG), "rate_limited_tpd", 1, id="429-tpd"),
    pytest.param(lambda: _rate_limit(_TPM_MSG), "rate_limited_tpm", 6, id="429-tpm"),
    pytest.param(lambda: _rate_limit("slow down"), "rate_limited", 6, id="429-unknown-shape"),
    pytest.param(lambda: groq.APIConnectionError(request=_REQ), "provider_unavailable", 6,
                 id="connection"),
    pytest.param(lambda: groq.APITimeoutError(request=_REQ), "provider_timeout", 6, id="timeout"),
    pytest.param(lambda: _status_error(groq.InternalServerError, 500), "provider_unavailable", 6,
                 id="500"),
    pytest.param(lambda: _status_error(groq.APIStatusError, 503), "provider_unavailable", 6,
                 id="503-generic-status"),
]


@pytest.mark.parametrize(("make_exc", "reason", "calls"), DEGRADE_CASES)
def test_provider_error_degrades_instead_of_raising(make_exc, reason, calls):
    cache = _empty_cache()
    (plan, raw, usage, llm_status, cache_hit), client, _ = _run(
        _make_assistant(), make_exc(), cache
    )
    assert llm_status == PROVIDER_DEGRADED_STATUS == "degraded_provider_error"
    assert usage["llm_status_reason"] == reason
    assert raw == "" and cache_hit is False
    assert client.chat.completions.create.call_count == calls
    cache.set.assert_not_called()  # a degraded result is never cached
    assert "manual review" in plan.triage_summary.lower()


def test_daily_quota_fails_fast_without_sleeping():
    """TPD cannot clear in-request: no 6-attempt ~108s backoff (the incident's 2 min)."""
    (_, _, _, status, _), client, sleep = _run(_make_assistant(), _rate_limit(_TPD_MSG))
    assert status == PROVIDER_DEGRADED_STATUS
    assert client.chat.completions.create.call_count == 1
    sleep.assert_not_called()


def test_degrade_path_itself_never_sleeps():
    # TPM: the pre-existing per-attempt backoff runs (5 sleeps) but the degrade adds none.
    (_, _, _, status, _), _, sleep = _run(_make_assistant(), _rate_limit(_TPM_MSG))
    assert status == PROVIDER_DEGRADED_STATUS
    assert sleep.call_count == 5


# ---------------------------------------------------------------------------
# Must keep raising: our bug / bad key
# ---------------------------------------------------------------------------

RAISE_CASES = [
    pytest.param(lambda: _status_error(groq.AuthenticationError, 401, "bad key"),
                 groq.AuthenticationError, id="401"),
    pytest.param(lambda: _status_error(groq.PermissionDeniedError, 403), groq.PermissionDeniedError,
                 id="403"),
    pytest.param(lambda: _status_error(groq.BadRequestError, 400, "invalid request"),
                 groq.BadRequestError, id="400"),
    pytest.param(lambda: _status_error(groq.UnprocessableEntityError, 422),
                 groq.UnprocessableEntityError, id="422"),
    pytest.param(lambda: _status_error(groq.NotFoundError, 404), groq.NotFoundError, id="404"),
]


@pytest.mark.parametrize(("make_exc", "exc_type"), RAISE_CASES)
def test_caller_or_config_errors_still_raise(make_exc, exc_type):
    cache = _empty_cache()
    asst = _make_assistant()
    asst._cache = cache
    client = MagicMock()
    client.chat.completions.create.side_effect = make_exc()
    with patch("groq.Groq", return_value=client), patch("time.sleep"), pytest.raises(exc_type):
        asst._call_llm_verbose(_signals())
    cache.set.assert_not_called()


def test_non_groq_exception_still_raises():
    with patch("groq.Groq", side_effect=ValueError("bug")), pytest.raises(ValueError):
        _make_assistant()._call_llm_verbose(_signals())


def test_classify_provider_error_split():
    assert classify_provider_error(_rate_limit(_TPD_MSG)) == "rate_limited_tpd"
    assert classify_provider_error(_status_error(groq.AuthenticationError, 401)) is None
    assert classify_provider_error(_status_error(groq.BadRequestError, 400)) is None
    assert classify_provider_error(RuntimeError("x")) is None


# ---------------------------------------------------------------------------
# Parse-retry branch
# ---------------------------------------------------------------------------


def test_parse_retry_call_provider_error_degrades():
    """First call returns non-JSON (-> retry), the retry call hits a 429."""
    cache = _empty_cache()
    (plan, _, usage, status, hit), client, _ = _run(
        _make_assistant(), [_ok_response("not json at all"), _rate_limit(_TPD_MSG)], cache
    )
    assert status == PROVIDER_DEGRADED_STATUS
    assert usage["llm_status_reason"] == "rate_limited_tpd"
    assert hit is False
    assert client.chat.completions.create.call_count == 2
    # Only the first (unparseable but real) completion was cached by pre-existing behaviour;
    # the degraded plan / retry never is.
    assert cache.set.call_count == 1
    assert cache.set.call_args.args[4]["content"] == "not json at all"


def test_parse_retry_call_auth_error_still_raises():
    asst = _make_assistant()
    client = MagicMock()
    client.chat.completions.create.side_effect = [
        _ok_response("not json"),
        _status_error(groq.AuthenticationError, 401),
    ]
    with patch("groq.Groq", return_value=client), patch("time.sleep"), pytest.raises(
        groq.AuthenticationError
    ):
        asst._call_llm_verbose(_signals())


# ---------------------------------------------------------------------------
# Fallback plan content
# ---------------------------------------------------------------------------


def test_fallback_plan_has_classifier_similar_issues_and_resolution():
    (plan, *_), _, _ = _run(_make_assistant(), _rate_limit(_TPD_MSG))
    assert plan.predicted_component == "workbench"
    assert plan.component_confidence == pytest.approx(0.61)
    assert [s.number for s in plan.similar_issues] == [11, 12]
    assert plan.similar_issues[0].similarity == pytest.approx(0.9)
    assert plan.expected_resolution_lower_days == 1.5
    assert plan.expected_resolution_upper_days == 20.0
    assert plan.resolution_bucket == "days"
    assert plan.priority_guess == "medium"
    assert "rate_limited_tpd" in plan.triage_summary
    # Provider text (could echo org ids) never reaches the plan.
    assert "org_x" not in plan.model_dump_json()


def test_older_degrade_paths_keep_empty_similar_issues():
    plan = _make_assistant()._make_fallback_plan(_signals(), reason="x")
    assert plan.similar_issues == []


def test_provider_text_truncated_in_logs(caplog):
    long_msg = _TPD_MSG + " " + "x" * 600
    with caplog.at_level("WARNING", logger="triage_iq.models.triage"):
        _run(_make_assistant(), _rate_limit(long_msg))
    rec = next(r for r in caplog.records if "degrading to signals-only" in r.getMessage())
    msg = rec.getMessage()
    assert "repo=microsoft/vscode" in msg and "status_code=429" in msg
    assert "error_code=rate_limit_exceeded" in msg and "reason=rate_limited_tpd" in msg
    assert "x" * 300 not in msg


# ---------------------------------------------------------------------------
# HTTP layer: 200, schema compatibility, metrics
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _env():
    from triage_iq.config import get_settings

    get_settings.cache_clear()
    with patch.dict("os.environ", {"RATE_LIMIT_ENABLED": "false"}):
        yield
    get_settings.cache_clear()


def _client_with_real_assistant(side_effect):
    asst = _make_assistant()
    asst._collect_signals = lambda issue: _signals()  # type: ignore[method-assign]
    groq_client = MagicMock()
    groq_client.chat.completions.create.side_effect = side_effect
    bundle = MagicMock()
    bundle.assistant = asst
    store = MagicMock()
    store.repos = ["microsoft/vscode"]
    store.start_time = time.monotonic()
    store.get.return_value = bundle
    store.conformal_adjustments = {}
    return store, groq_client


def _sample(name: str, labels: dict) -> float:
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_triage_endpoint_returns_200_degraded_and_labels_metrics():
    store, groq_client = _client_with_real_assistant(_rate_limit(_TPD_MSG))
    req_labels = {"repo": "microsoft/vscode", "status": "provider_degraded"}
    err_labels = {"reason": "rate_limited_tpd"}
    before = (
        _sample("triage_requests_total", req_labels),
        _sample("triage_llm_provider_errors_total", err_labels),
        _sample("triage_llm_fallback_total", {}),
    )
    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        patch("groq.Groq", return_value=groq_client),
        patch("time.sleep"),
        TestClient(app) as c,
    ):
        r = c.post("/triage", json={"repo": "microsoft/vscode", "title": "t", "body": "b"})
    assert r.status_code == 200
    d = r.json()
    assert d["_degraded"] is True
    assert d["_llm_status"] == "degraded_provider_error"
    assert d["_llm_status_reason"] == "rate_limited_tpd"
    assert d["_model"] == "openai/gpt-oss-120b"
    # Existing response contract intact.
    for key in (
        "predicted_component", "component_confidence", "similar_issues", "triage_summary",
        "expected_resolution_lower_days", "expected_resolution_upper_days", "resolution_bucket",
        "resolution_confidence_pct", "priority_guess", "grounding_status", "_request_id",
        "_llm_cache_hit", "classifier_top3", "resolution_model_beats_naive",
    ):
        assert key in d, key
    assert d["predicted_component"] == "workbench"
    assert [s["number"] for s in d["similar_issues"]] == [11, 12]
    assert _sample("triage_requests_total", req_labels) == before[0] + 1
    assert _sample("triage_llm_provider_errors_total", err_labels) == before[1] + 1
    assert _sample("triage_llm_fallback_total", {}) == before[2] + 1


def test_triage_endpoint_auth_error_is_still_500():
    store, groq_client = _client_with_real_assistant(
        _status_error(groq.AuthenticationError, 401, "invalid_api_key")
    )
    with (
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
        patch("groq.Groq", return_value=groq_client),
        patch("time.sleep"),
        TestClient(app, raise_server_exceptions=False) as c,
    ):
        r = c.post("/triage", json={"repo": "microsoft/vscode", "title": "t", "body": "b"})
    assert r.status_code == 500
