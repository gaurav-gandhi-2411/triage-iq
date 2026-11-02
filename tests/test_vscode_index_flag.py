"""TRIAGE_VSCODE_INDEX: index selection, fail-closed checks, k8s behaviour unchanged.

Network-free and hermetic: the indexes are tiny real faiss/meta files under tmp_path and the
SentenceTransformer constructor is stubbed out. No real model is loaded.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import faiss
import joblib
import numpy as np
import pytest

from triage_iq.api import loader
from triage_iq.api.loader import (
    K8sIndexArtifactError,
    ModelStore,
    VscodeIndexArtifactError,
    _load_detector,
    _vscode_index_dirname,
)
from triage_iq.models.similar_issues import SimilarIssueRetriever

FLAG = "TRIAGE_VSCODE_INDEX"
VSCODE = "microsoft_vscode"
K8S = "kubernetes_kubernetes"
SERVED = f"similar_issue_index_{VSCODE}_bge"
SERVED_DATED = f"similar_issue_index_{VSCODE}_served_dated"
V2_DATED = f"similar_issue_index_{VSCODE}_bge_v2_dated"
DIRS = {"served_dated": SERVED_DATED, "v2": V2_DATED}


@pytest.fixture(autouse=True)
def _no_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    def _init(self: Any, repo: str, model_key: str = "bge") -> None:
        self.repo, self.model_key = repo, model_key

    monkeypatch.setattr(SimilarIssueRetriever, "__init__", _init)


def _save_index(models_dir: Path, dirname: str, marker: int) -> None:
    """Tiny index in the SimilarIssueRetriever.save format; `marker` is the issue number."""
    d = models_dir / dirname
    d.mkdir(parents=True)
    idx = faiss.IndexFlatIP(4)
    idx.add(np.eye(1, 4, dtype="float32"))
    faiss.write_index(idx, str(d / "index.faiss"))
    joblib.dump({"repo": "r", "model_key": "bge", "issue_numbers": [marker], "texts": ["t"]},
                str(d / "meta.pkl"))  # fmt: skip


def _manifest(models_dir: Path, dirname: str, *, bad: str | None = None) -> None:
    lines = ["a" * 64 + f"  data/models/{SERVED}/index.faiss"]
    for name in ("index.faiss", "meta.pkl"):
        digest = hashlib.sha256((models_dir / dirname / name).read_bytes()).hexdigest()
        lines.append(f"{'0' * 64 if bad == name else digest}  data/models/{dirname}/{name}")
    (models_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _marker(det: SimilarIssueRetriever) -> int:
    return det.issue_numbers[0]


def test_flag_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    assert _vscode_index_dirname(VSCODE) is None
    monkeypatch.setenv(FLAG, "")
    assert _vscode_index_dirname(VSCODE) is None
    for value, dirname in DIRS.items():
        monkeypatch.setenv(FLAG, value)
        assert _vscode_index_dirname(VSCODE) == dirname
        assert _vscode_index_dirname(K8S) is None  # vscode only


@pytest.mark.parametrize("bad", ["v3", "V2", "served", "1", " v2"])
def test_unknown_value_raises(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    monkeypatch.setenv(FLAG, bad)  # a typo must not silently serve the default
    with pytest.raises(VscodeIndexArtifactError):
        _vscode_index_dirname(VSCODE)
    with pytest.raises(VscodeIndexArtifactError):
        _vscode_index_dirname(K8S)


def test_unset_loads_the_served_index_even_if_candidates_exist(tmp_path: Path) -> None:
    _save_index(tmp_path, SERVED, marker=1)
    _save_index(tmp_path, SERVED_DATED, marker=2)
    _save_index(tmp_path, V2_DATED, marker=3)
    assert _marker(_load_detector(tmp_path, VSCODE)) == 1
    assert _marker(_load_detector(tmp_path, VSCODE, vscode_dirname=None)) == 1


@pytest.mark.parametrize(("value", "marker"), [("served_dated", 2), ("v2", 3)])
def test_each_value_loads_its_directory(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, value: str, marker: int
) -> None:
    _save_index(tmp_path, SERVED, marker=1)
    _save_index(tmp_path, SERVED_DATED, marker=2)
    _save_index(tmp_path, V2_DATED, marker=3)
    _manifest(tmp_path, DIRS[value])
    with caplog.at_level("INFO", logger=loader.logger.name):
        det = _load_detector(tmp_path, VSCODE, vscode_dirname=DIRS[value])
    assert _marker(det) == marker
    assert DIRS[value] in caplog.text


@pytest.mark.parametrize("bad", ["index.faiss", "meta.pkl"])
def test_manifest_mismatch_fails_closed(tmp_path: Path, bad: str) -> None:
    _save_index(tmp_path, SERVED, marker=1)
    _save_index(tmp_path, V2_DATED, marker=3)
    _manifest(tmp_path, V2_DATED, bad=bad)
    with pytest.raises(VscodeIndexArtifactError, match="sha256"):
        _load_detector(tmp_path, VSCODE, vscode_dirname=V2_DATED)


def test_missing_manifest_or_line_fails_closed(tmp_path: Path) -> None:
    _save_index(tmp_path, SERVED, marker=1)
    _save_index(tmp_path, V2_DATED, marker=3)
    with pytest.raises(VscodeIndexArtifactError, match="TRIAGE_VSCODE_INDEX .*unreadable"):
        _load_detector(tmp_path, VSCODE, vscode_dirname=V2_DATED)
    _manifest(tmp_path, SERVED)  # only the served index is pinned
    with pytest.raises(VscodeIndexArtifactError, match="0 entries"):
        _load_detector(tmp_path, VSCODE, vscode_dirname=V2_DATED)


def test_missing_directory_fails_closed_without_falling_back(tmp_path: Path) -> None:
    _save_index(tmp_path, SERVED, marker=1)
    (tmp_path / "MANIFEST.sha256").write_text(
        f"{'a' * 64}  data/models/{V2_DATED}/index.faiss\n"
        f"{'b' * 64}  data/models/{V2_DATED}/meta.pkl\n",
        encoding="utf-8",
    )
    with pytest.raises(VscodeIndexArtifactError, match="cannot read"):
        _load_detector(tmp_path, VSCODE, vscode_dirname=V2_DATED)


def test_unloadable_files_fail_closed(tmp_path: Path) -> None:
    _save_index(tmp_path, SERVED, marker=1)
    _save_index(tmp_path, V2_DATED, marker=3)
    (tmp_path / V2_DATED / "meta.pkl").write_bytes(b"not a pickle")
    _manifest(tmp_path, V2_DATED)  # hashes match the corrupt file, so the load itself must fail
    with pytest.raises(VscodeIndexArtifactError, match="failed to load"):
        _load_detector(tmp_path, VSCODE, vscode_dirname=V2_DATED)


# --- load_all: errors must propagate, and k8s must be unaffected ---------------------------------


def _patch_store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[str]:
    """Stub everything but the detector step; returns the list of detector dirs loaded."""
    import pandas as pd

    from triage_iq.models import component_classifier
    from triage_iq.models import triage as triage_mod

    loaded: list[str] = []

    def _load(cls: Any, out_dir: str) -> str:
        loaded.append(Path(out_dir).name)
        return Path(out_dir).name

    monkeypatch.setattr(component_classifier, "load_classifier", lambda *a, **k: object())
    monkeypatch.setattr(SimilarIssueRetriever, "load", classmethod(_load))
    monkeypatch.setattr(loader, "_load_predictor", lambda *a, **k: object())
    monkeypatch.setattr(loader, "_load_train", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(loader, "_load_conformal_adjustments", lambda *a, **k: {})
    monkeypatch.setattr(triage_mod, "TriageAssistant", lambda **k: object())
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("TRIAGE_K8S_INDEX", raising=False)
    monkeypatch.delenv(FLAG, raising=False)
    (tmp_path / "models").mkdir()
    return loaded


def test_load_all_missing_dir_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store(monkeypatch, tmp_path)
    monkeypatch.setenv(FLAG, "v2")
    with pytest.raises(VscodeIndexArtifactError):  # not skipped, not "No repo models"
        ModelStore.load_all(data_dir=tmp_path)


def test_load_all_invalid_value_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store(monkeypatch, tmp_path)
    monkeypatch.setenv(FLAG, "bogus")
    with pytest.raises(VscodeIndexArtifactError):
        ModelStore.load_all(data_dir=tmp_path)


def test_load_all_selects_dir_for_vscode_only_and_k8s_flag_is_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _patch_store(monkeypatch, tmp_path)
    models = tmp_path / "models"
    k8s_served = f"similar_issue_index_{K8S}_bge"
    k8s_v2 = f"similar_issue_index_{K8S}_bge_v2"
    for d in (SERVED, SERVED_DATED, V2_DATED, k8s_served, k8s_v2):
        (models / d).mkdir()
    # sha-verified dirs need real files; here only the choice of directory is asserted
    monkeypatch.setattr(loader, "_verify_index_v2_manifest", lambda *a, **k: None)

    ModelStore.load_all(data_dir=tmp_path)  # flags unset: today's behaviour
    assert sorted(loaded) == sorted([SERVED, k8s_served])

    for value, dirname in DIRS.items():
        loaded.clear()
        monkeypatch.setenv(FLAG, value)
        ModelStore.load_all(data_dir=tmp_path)
        assert sorted(loaded) == sorted([dirname, k8s_served])  # k8s untouched

    loaded.clear()
    monkeypatch.delenv(FLAG)
    monkeypatch.setenv("TRIAGE_K8S_INDEX", "v2")
    ModelStore.load_all(data_dir=tmp_path)
    assert sorted(loaded) == sorted([SERVED, k8s_v2])  # k8s v2 does not move vscode


def test_k8s_v2_error_class_unchanged(tmp_path: Path) -> None:
    (tmp_path / "MANIFEST.sha256").write_text("", encoding="utf-8")
    with pytest.raises(K8sIndexArtifactError, match="TRIAGE_K8S_INDEX=v2 but"):
        _load_detector(tmp_path, K8S, v2=True)
