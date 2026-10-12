"""Cold-start overlap in ModelStore.load_all (2026-10-12).

Production startup_step logs showed the first import of the retriever stack (~17.7 s) and the manifest
drift check (~4.9 s) running one after the other in front of the models. They are independent, so they now
run in background threads while the models load. Behaviour that must not change: the drift check still runs
exactly once, still finishes (and logs its ARTIFACT_DRIFT warnings) before load_all returns, a failing
prewarm never breaks startup, and the loaded bundles are the ones the loaders returned.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from pathlib import Path

import pytest

from triage_iq.api import loader


@pytest.fixture
def stubbed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """load_all with every heavy loader replaced by a recorder; returns (data_dir, calls)."""
    (tmp_path / "models").mkdir()
    (tmp_path / "processed").mkdir()
    calls: dict = {"threads": {}, "order": []}

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
    return tmp_path, calls


def test_drift_check_runs_once_in_the_background_and_finishes_before_load_all_returns(
    stubbed, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, calls = stubbed
    done = threading.Event()

    def slow_check(d: Path) -> None:
        calls["threads"]["drift"] = threading.current_thread().name
        calls["order"].append("drift-start")
        time.sleep(0.3)
        calls["order"].append("drift-end")
        done.set()

    monkeypatch.setattr(loader, "_check_manifest_drift", slow_check)
    monkeypatch.setattr(loader, "_prewarm_retriever_imports", lambda: calls["order"].append("prewarm"))

    store = loader.ModelStore.load_all(data_dir=data_dir, groq_api_key="x")

    assert done.is_set(), "load_all returned before the drift check finished"
    assert calls["order"].count("drift-start") == 1
    assert calls["threads"]["drift"] == "manifest-drift-check"  # not the main thread
    assert "prewarm" in calls["order"]
    assert sorted(store.repos) == ["kubernetes/kubernetes", "microsoft/vscode"]
    for b in (store.get("microsoft/vscode"), store.get("kubernetes/kubernetes")):
        assert b.classifier[0] == "clf" and b.detector[0] == "det"
        assert b.predictor[0] == "pred" and b.train_df[0] == "train"


def test_artifact_drift_warnings_still_reach_the_log_before_load_all_returns(
    stubbed, caplog: pytest.LogCaptureFixture
) -> None:
    data_dir, _ = stubbed
    (data_dir / "models" / "MANIFEST.sha256").write_text(
        "0" * 64 + "  data/models/missing_artifact.pkl\n", encoding="utf-8"
    )
    with caplog.at_level(logging.WARNING, logger="triage_iq.api.loader"):
        loader.ModelStore.load_all(data_dir=data_dir, groq_api_key="x")
    msgs = [r.getMessage() for r in caplog.records]
    assert any("ARTIFACT_DRIFT: missing data/models/missing_artifact.pkl" in m for m in msgs)
    assert any("1/1 artifact(s) do not match" in m for m in msgs)


def test_a_failing_prewarm_does_not_break_startup(
    stubbed, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir, _ = stubbed
    # A None entry in sys.modules makes `import triage_iq.models.similar_issues` raise ImportError.
    monkeypatch.setitem(sys.modules, "triage_iq.models.similar_issues", None)
    loader._prewarm_retriever_imports()  # must not raise
    store = loader.ModelStore.load_all(data_dir=data_dir, groq_api_key="x")
    assert len(store.repos) == 2
