"""RECORDING_CONFIG.json, the shared assistant builder and the config-aware artifact fingerprint.

Hermetic: tmp dirs, tiny files, loaders monkeypatched; no real model, no cassette, no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT / "src", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import artifact_fingerprint as af  # noqa: E402
import recording_config as rc  # noqa: E402

from triage_iq.api import loader  # noqa: E402
from triage_iq.models import component_classifier as cc  # noqa: E402
from triage_iq.models import resolution, triage  # noqa: E402

VSCODE, K8S = "microsoft/vscode", "kubernetes/kubernetes"
GOOD = {"vscode_classifier": "family_v2", "k8s_index": "v2", "resolution_evidence": True}


def _write(d: Path, cfg: Any) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    p = d / rc.CONFIG_FILENAME
    p.write_text(json.dumps(cfg), encoding="utf-8")
    return p


@pytest.fixture(autouse=True)
def _no_set_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TRIAGEIQ_RECORDING_SET_DIR", raising=False)


# --- config file ---------------------------------------------------------------------------


def test_absent_file_is_todays_behaviour(tmp_path: Path) -> None:
    assert rc.load_recording_config(tmp_path) == rc.RecordingConfig("served", "served", False)


def test_valid_file_in_cassette_dir_and_in_set_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(tmp_path / "eval" / "cassettes", GOOD)
    assert rc.load_recording_config(tmp_path) == rc.RecordingConfig("family_v2", "v2", True)
    set_dir = tmp_path / "set"
    monkeypatch.setenv("TRIAGEIQ_RECORDING_SET_DIR", str(set_dir))
    _write(set_dir, GOOD)
    assert (
        rc.load_recording_config(tmp_path, set_dir / "cassettes").vscode_classifier == "family_v2"
    )


def test_files_in_both_places_are_ambiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    set_dir = tmp_path / "set"
    monkeypatch.setenv("TRIAGEIQ_RECORDING_SET_DIR", str(set_dir))
    _write(set_dir, GOOD)
    _write(set_dir / "cassettes", GOOD)
    with pytest.raises(rc.RecordingConfigError, match="ambiguous"):
        rc.load_recording_config(tmp_path)


@pytest.mark.parametrize(
    "bad",
    [
        [],
        {},
        {**GOOD, "extra": 1},
        {"vscode_classifier": "family_v2", "k8s_index": "v2"},
        {**GOOD, "vscode_classifier": "family_v3"},
        {**GOOD, "k8s_index": "V2"},
        {**GOOD, "resolution_evidence": 1},
        {**GOOD, "resolution_evidence": "true"},
        {**GOOD, "vscode_classifier": None},
    ],
)
def test_invalid_config_raises(tmp_path: Path, bad: object) -> None:
    _write(tmp_path / "eval" / "cassettes", bad)
    with pytest.raises(rc.RecordingConfigError):
        rc.load_recording_config(tmp_path)


def test_unparseable_file_raises(tmp_path: Path) -> None:
    d = tmp_path / "eval" / "cassettes"
    d.mkdir(parents=True)
    (d / rc.CONFIG_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(rc.RecordingConfigError, match="unreadable"):
        rc.load_recording_config(tmp_path)


# --- rows ----------------------------------------------------------------------------------

_ISSUE = {"title": "t", "body": "cleaned", "number": 7, "created_at": "2025-01-02T03:04:05Z"}


def test_row_without_body_raw_is_unchanged() -> None:
    row = rc.issue_to_row(_ISSUE)
    assert list(row.index) == ["title", "body_clean", "number", "created_at"]
    assert row["body_clean"] == "cleaned"


def test_row_carries_body_raw_when_present() -> None:
    row = rc.issue_to_row({**_ISSUE, "body_raw": "RAW body"})
    assert row["body_raw"] == "RAW body" and row["body_clean"] == "cleaned"


# --- builder -------------------------------------------------------------------------------


class _Spy:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.assistant_kwargs: dict[str, Any] = {}


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _Spy:
    s = _Spy()

    def load_classifier(models_dir: Path, slug: str) -> str:
        s.calls.append(f"served:{slug}")
        return "served-clf"

    def load_family(models_dir: Path) -> str:
        s.calls.append("family_v2")
        return "family-clf"

    class Pred:
        @staticmethod
        def load(path: str) -> str:
            s.calls.append(f"predictor:{Path(path).name}")
            return "pred"

    class Asst:
        def __init__(self, **kw: Any) -> None:
            s.assistant_kwargs = kw

    def evidence(models_dir: Path, slug: str) -> Any:
        s.calls.append(f"evidence:{slug}")
        return "evidence"

    monkeypatch.setattr(cc, "load_classifier", load_classifier)
    monkeypatch.setattr(cc, "load_family_v2_classifier", load_family)
    monkeypatch.setattr(resolution, "ResolutionTimePredictor", Pred)
    monkeypatch.setattr(triage, "TriageAssistant", Asst)
    monkeypatch.setattr(loader, "_load_resolution_evidence", evidence)
    monkeypatch.setattr(pd, "read_parquet", lambda p: f"train:{Path(p).name}")
    return s


def _build(repo: str, config: rc.RecordingConfig) -> dict[str, Any]:
    slug = af.REPO_SLUGS[repo]
    return rc.build_assistant(
        repo,
        slug,
        models_dir=Path("m"),
        processed_dir=Path("p"),
        detector="frozen",
        config=config,
        groq_api_key="k",
        cache="cassette",
        max_tokens=5,
    )


@pytest.mark.parametrize("repo", [VSCODE, K8S])
def test_default_config_builds_exactly_what_the_old_code_built(spy: _Spy, repo: str) -> None:
    slug = af.REPO_SLUGS[repo]
    out = _build(repo, rc.RecordingConfig())
    assert spy.calls == [f"served:{slug}", f"predictor:resolution_predictor_{slug}.pkl"]
    assert spy.assistant_kwargs == {
        "repo": repo,
        "classifier": "served-clf",
        "detector": "frozen",
        "predictor": "pred",
        "train_df": f"train:{slug}_temporal_train.parquet",
        "groq_api_key": "k",
        "cache": "cassette",
        "max_tokens": 5,
    }
    assert out["classifier"] == "served-clf" and "resolution_evidence" not in spy.assistant_kwargs


def test_family_v2_applies_to_vscode_only(spy: _Spy) -> None:
    cfg = rc.RecordingConfig(vscode_classifier="family_v2")
    assert _build(VSCODE, cfg)["classifier"] == "family-clf"
    assert _build(K8S, cfg)["classifier"] == "served-clf"


def test_evidence_applies_to_k8s_only_and_fails_closed(
    spy: _Spy, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = rc.RecordingConfig(resolution_evidence=True)
    _build(K8S, cfg)
    assert spy.assistant_kwargs["resolution_evidence"] == "evidence"
    _build(VSCODE, cfg)
    assert "resolution_evidence" not in spy.assistant_kwargs
    monkeypatch.setattr(loader, "_load_resolution_evidence", lambda d, s: None)
    with pytest.raises(rc.RecordingConfigError, match="no usable evidence"):
        _build(K8S, cfg)


# --- fingerprint ---------------------------------------------------------------------------


def _tree(root: Path) -> None:
    paths = [p for r in af.REPO_SLUGS for p in af.artifact_paths_for_repo(r)]
    paths += [
        f"data/models/{cc.FAMILY_V2_FILENAME}",
        f"data/models/{loader._K8S_INDEX_V2_DIRNAME}/index.faiss",
        f"data/models/{loader._K8S_INDEX_V2_DIRNAME}/meta.pkl",
        "data/models/resolution_evidence_kubernetes_kubernetes.parquet",
    ]
    for rel in paths:
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(rel.encode())


def test_fingerprint_unchanged_without_config_and_with_default_config(tmp_path: Path) -> None:
    _tree(tmp_path)
    base = af.compute_artifact_hashes(tmp_path)
    assert af.compute_artifact_hashes(tmp_path, config=rc.RecordingConfig()) == base
    assert af.compute_artifact_hashes(tmp_path, repo=VSCODE, config=rc.RecordingConfig()) == (
        af.compute_artifact_hashes(tmp_path, repo=VSCODE)
    )
    assert set(base) == {p for r in af.REPO_SLUGS for p in af.artifact_paths_for_repo(r)}


def test_fingerprint_extends_per_selected_artifact(tmp_path: Path) -> None:
    _tree(tmp_path)
    base = af.compute_artifact_hashes(tmp_path)
    fam = f"data/models/{cc.FAMILY_V2_FILENAME}"
    idx = f"data/models/{loader._K8S_INDEX_V2_DIRNAME}/index.faiss"
    ev = "data/models/resolution_evidence_kubernetes_kubernetes.parquet"

    only_clf = af.compute_artifact_hashes(
        tmp_path, config=rc.RecordingConfig(vscode_classifier="family_v2")
    )
    assert set(only_clf) - set(base) == {fam}
    only_idx = af.compute_artifact_hashes(tmp_path, config=rc.RecordingConfig(k8s_index="v2"))
    assert set(only_idx) - set(base) == {idx, idx.replace("index.faiss", "meta.pkl")}
    only_ev = af.compute_artifact_hashes(
        tmp_path, config=rc.RecordingConfig(resolution_evidence=True)
    )
    assert set(only_ev) - set(base) == {ev}
    full = af.compute_artifact_hashes(tmp_path, config=rc.RecordingConfig("family_v2", "v2", True))
    assert af.combined_hash(full) != af.combined_hash(base)
    # per-repo tagging names only that repo's selected artifacts
    assert set(
        af.compute_artifact_hashes(
            tmp_path, repo=VSCODE, config=rc.RecordingConfig("family_v2", "v2", True)
        )
    ) - set(af.compute_artifact_hashes(tmp_path, repo=VSCODE)) == {fam}


def test_missing_selected_artifact_reads_missing(tmp_path: Path) -> None:
    _tree(tmp_path)
    (tmp_path / f"data/models/{cc.FAMILY_V2_FILENAME}").unlink()
    h = af.compute_artifact_hashes(
        tmp_path, config=rc.RecordingConfig(vscode_classifier="family_v2")
    )
    assert h[f"data/models/{cc.FAMILY_V2_FILENAME}"] == "MISSING"
