"""Per-request retrieval log record (G1 / D90): fields, privacy, and never breaking /triage."""

from __future__ import annotations

import json
import logging
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from test_api import _fake_meta, _fake_plan, _make_store
from tests.test_component_decision import _plan
from tests.test_vscode_naive_point import K8S, _assistant, _issue

from triage_iq.api.app import _access_logger, app
from triage_iq.models.triage import TriageMetadata

_REQ = {
    "repo": "microsoft/vscode",
    "title": "SECRET-TITLE-TEXT",
    "body": "SECRET-BODY-TEXT",
    "issue_number": 9999,
}
_TOP5 = [{"number": 7, "score": 0.9123}, {"number": 3, "score": 0.5}]


@pytest.fixture(autouse=True)
def _rate_limit_off():
    from triage_iq.config import get_settings

    get_settings.cache_clear()
    with patch.dict("os.environ", {"RATE_LIMIT_ENABLED": "false"}):
        yield
    get_settings.cache_clear()


def _patches(meta: dict, top5: list | None = None):
    wrapped = TriageMetadata(meta)
    wrapped.retrieval_top5 = top5
    store = _make_store()
    store.get.return_value.assistant.triage_with_metadata.return_value = (_fake_plan(), wrapped)
    return (
        patch.dict("os.environ", {"GROQ_API_KEY": "test-key-for-tests"}),
        patch("triage_iq.api.app.ModelStore.load_all", return_value=store),
    )


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def test_retrieval_record_fields_and_no_text() -> None:
    meta = {
        **_fake_meta(),
        "classifier_top3": [
            {"label": "editor", "confidence": 0.8},
            {"label": "git", "confidence": 0.1},
            {"label": "debug", "confidence": 0.05},
        ],
    }
    cap = _Capture()
    _access_logger.addHandler(cap)
    p1, p2 = _patches(meta, _TOP5)
    try:
        with p1, p2, TestClient(app) as c:
            r = c.post("/triage", json=_REQ)
    finally:
        _access_logger.removeHandler(cap)
    assert r.status_code == 200
    recs = [x for x in cap.records if getattr(x, "log_type", "") == "retrieval"]
    assert len(recs) == 1
    rec = recs[0]
    assert rec.repo == "microsoft/vscode"
    assert rec.issue_number == 9999
    assert rec.classifier_top3 == ["editor", "git", "debug"]
    assert rec.retrieval_top5 == _TOP5
    assert rec.body_cleaned is False
    assert rec.request_id == r.json()["_request_id"]
    dumped = json.dumps({k: v for k, v in vars(rec).items() if k != "args"}, default=str)
    assert "SECRET" not in dumped
    # the response must not gain the retrieval list
    assert "retrieval_top5" not in r.json()


def test_logging_failure_does_not_break_triage() -> None:
    p1, p2 = _patches(_fake_meta(), _TOP5)
    with p1, p2, TestClient(app) as c:
        baseline = c.post("/triage", json=_REQ).json()
    p1, p2 = _patches(_fake_meta(), _TOP5)
    real_info = _access_logger.info

    def _boom(msg, *a, **k):  # noqa: ANN001, ANN002, ANN003, ANN202
        if msg == "retrieval":
            raise RuntimeError("log sink down")
        return real_info(msg, *a, **k)

    with p1, p2, TestClient(app) as c, patch.object(_access_logger, "info", side_effect=_boom):
        r = c.post("/triage", json=_REQ)
    assert r.status_code == 200
    got = r.json()
    for d in (baseline, got):
        d.pop("_request_id")
    assert got == baseline


def test_assistant_attaches_top5_without_changing_metadata_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asst = _assistant(K8S)
    monkeypatch.setattr(
        asst, "_call_llm_verbose", lambda signals: (_plan("a"), "", {}, "ok", False)
    )
    _, meta = asst.triage_with_metadata(_issue())
    assert isinstance(meta, dict)
    assert "retrieval_top5" not in meta
    top5 = meta.retrieval_top5
    assert isinstance(top5, list)
    assert all(set(x) == {"number", "score"} for x in top5)
    assert all(x["score"] == round(x["score"], 4) for x in top5)
