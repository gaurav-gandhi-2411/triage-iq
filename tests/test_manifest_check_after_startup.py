"""The artifact drift check no longer sits in front of the models (2026-10-12).

Production startup_step logs: manifest_drift_check took 4.9 s serially before any model loaded. It only logs,
and the deploy job already fails a drifted image, so it now runs on a background thread after "Models ready".
An overlap attempt (drift check + torch-import prewarm running DURING loading) made the cold start slower
(load_all 35.5 s vs 25.4-28.4 s on three earlier revisions); this test pins the replacement: nothing runs
concurrently with the loaders.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import threading
from pathlib import Path

import pytest

from triage_iq.api import loader

# Not `from triage_iq.api import app`: that name is the FastAPI instance, not the module.
app_mod = importlib.import_module("triage_iq.api.app")


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    (tmp_path / "models").mkdir()
    (tmp_path / "processed").mkdir()

    class _Asst:
        def __init__(self, **kw) -> None:  # noqa: ANN003
            self.kw = kw

    import triage_iq.models.component_classifier as cc
    import triage_iq.models.triage as triage_mod

    monkeypatch.setattr(triage_mod, "TriageAssistant", _Asst)
    monkeypatch.setattr(cc, "load_classifier", lambda d, s: ("clf", s))
    monkeypatch.setattr(loader, "_load_detector", lambda d, s: ("det", s))
    monkeypatch.setattr(loader, "_load_predictor", lambda d, s: ("pred", s))
    monkeypatch.setattr(loader, "_load_train", lambda d, s: ("train", s))
    return tmp_path


def test_load_all_does_not_run_the_drift_check_or_start_threads(
    stubbed: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def must_not_run(d: Path) -> None:
        raise AssertionError("the drift check must not run inside load_all")

    monkeypatch.setattr(loader, "_check_manifest_drift", must_not_run)
    before = {t.name for t in threading.enumerate()}
    store = loader.ModelStore.load_all(data_dir=stubbed, groq_api_key="x")
    assert sorted(store.repos) == ["kubernetes/kubernetes", "microsoft/vscode"]
    assert {t.name for t in threading.enumerate()} - before == set()  # no background work during loading


def test_background_check_runs_once_off_the_calling_thread_and_logs_drift(
    stubbed: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (stubbed / "models" / "MANIFEST.sha256").write_text(
        "0" * 64 + "  data/models/missing_artifact.pkl\n", encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="triage_iq.api.loader"):
        t = loader.start_manifest_check_in_background(stubbed)
        assert t.name == "manifest-drift-check" and t is not threading.current_thread()
        t.join(timeout=10)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("ARTIFACT_DRIFT: missing data/models/missing_artifact.pkl" in m for m in msgs)
    assert any("1/1 artifact(s) do not match" in m for m in msgs)


def test_lifespan_starts_the_check_after_models_are_ready_and_before_serving() -> None:
    src = inspect.getsource(app_mod.lifespan)
    ready = src.index("Models ready")
    started = src.index("start_manifest_check_in_background(cfg.data_dir)")
    served = src.index("yield")
    assert ready < started < served
