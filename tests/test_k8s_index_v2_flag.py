"""TRIAGE_K8S_INDEX=v2 (D92.4): index selection, fail-closed checks, vscode unaffected.

Network-free and hermetic: the indexes are tiny real faiss/meta files under tmp_path and the
SentenceTransformer constructor is stubbed out.
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
    _k8s_index_v2_requested,
    _load_detector,
)
from triage_iq.models.similar_issues import SimilarIssueRetriever

FLAG = "TRIAGE_K8S_INDEX"
K8S = "kubernetes_kubernetes"
VSCODE = "microsoft_vscode"
OLD = f"similar_issue_index_{K8S}_bge"
V2 = f"similar_issue_index_{K8S}_bge_v2"


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


def _manifest(models_dir: Path, dirname: str = V2, *, bad: str | None = None) -> None:
    lines = ["a" * 64 + f"  data/models/{OLD}/index.faiss"]
    for name in ("index.faiss", "meta.pkl"):
        digest = hashlib.sha256((models_dir / dirname / name).read_bytes()).hexdigest()
        lines.append(f"{'0' * 64 if bad == name else digest}  data/models/{dirname}/{name}")
    (models_dir / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _marker(det: SimilarIssueRetriever) -> int:
    return det.issue_numbers[0]


def test_flag_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    assert _k8s_index_v2_requested(K8S) is False
    monkeypatch.setenv(FLAG, "")
    assert _k8s_index_v2_requested(K8S) is False
    monkeypatch.setenv(FLAG, "v2")
    assert _k8s_index_v2_requested(K8S) is True
    assert _k8s_index_v2_requested(VSCODE) is False  # k8s only
    monkeypatch.setenv(FLAG, "v3")  # a typo must not silently serve the default
    with pytest.raises(K8sIndexArtifactError):
        _k8s_index_v2_requested(K8S)
    with pytest.raises(K8sIndexArtifactError):
        _k8s_index_v2_requested(VSCODE)


def test_flag_unset_loads_the_served_index_even_if_v2_exists(tmp_path: Path) -> None:
    _save_index(tmp_path, OLD, marker=1)
    _save_index(tmp_path, V2, marker=2)
    assert _marker(_load_detector(tmp_path, K8S)) == 1


def test_v2_loads_the_v2_directory(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _save_index(tmp_path, OLD, marker=1)
    _save_index(tmp_path, V2, marker=2)
    _manifest(tmp_path)
    with caplog.at_level("INFO", logger=loader.logger.name):
        det = _load_detector(tmp_path, K8S, v2=True)
    assert _marker(det) == 2
    assert V2 in caplog.text


@pytest.mark.parametrize("bad", ["index.faiss", "meta.pkl"])
def test_manifest_mismatch_fails_closed(tmp_path: Path, bad: str) -> None:
    _save_index(tmp_path, OLD, marker=1)
    _save_index(tmp_path, V2, marker=2)
    _manifest(tmp_path, bad=bad)
    with pytest.raises(K8sIndexArtifactError, match="sha256"):
        _load_detector(tmp_path, K8S, v2=True)


def test_missing_manifest_or_line_fails_closed(tmp_path: Path) -> None:
    _save_index(tmp_path, OLD, marker=1)
    _save_index(tmp_path, V2, marker=2)
    with pytest.raises(K8sIndexArtifactError, match="unreadable"):
        _load_detector(tmp_path, K8S, v2=True)
    _manifest(tmp_path, dirname=OLD)  # only the served index is pinned
    with pytest.raises(K8sIndexArtifactError, match="0 entries"):
        _load_detector(tmp_path, K8S, v2=True)


def test_missing_v2_directory_fails_closed_without_falling_back(tmp_path: Path) -> None:
    _save_index(tmp_path, OLD, marker=1)
    (tmp_path / "MANIFEST.sha256").write_text(
        f"{'a' * 64}  data/models/{V2}/index.faiss\n{'b' * 64}  data/models/{V2}/meta.pkl\n",
        encoding="utf-8",
    )
    with pytest.raises(K8sIndexArtifactError, match="cannot read"):
        _load_detector(tmp_path, K8S, v2=True)


def test_unloadable_v2_files_fail_closed(tmp_path: Path) -> None:
    _save_index(tmp_path, OLD, marker=1)
    _save_index(tmp_path, V2, marker=2)
    (tmp_path / V2 / "meta.pkl").write_bytes(b"not a pickle")
    _manifest(tmp_path)  # hashes match the corrupt file, so the load itself must fail
    with pytest.raises(K8sIndexArtifactError, match="failed to load"):
        _load_detector(tmp_path, K8S, v2=True)


# --- load_all: the error must propagate, and vscode must be unaffected ---------------------------


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
    (tmp_path / "models").mkdir()
    return loaded


def test_load_all_v2_missing_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store(monkeypatch, tmp_path)
    monkeypatch.setenv(FLAG, "v2")
    with pytest.raises(K8sIndexArtifactError):  # not skipped, not "No repo models"
        ModelStore.load_all(data_dir=tmp_path)


def test_load_all_invalid_value_propagates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_store(monkeypatch, tmp_path)
    monkeypatch.setenv(FLAG, "bogus")
    with pytest.raises(K8sIndexArtifactError):
        ModelStore.load_all(data_dir=tmp_path)


def test_load_all_vscode_unaffected_by_the_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = _patch_store(monkeypatch, tmp_path)
    models = tmp_path / "models"
    (models / f"similar_issue_index_{VSCODE}_bge").mkdir()
    (models / OLD).mkdir()
    (models / V2).mkdir()
    for flag in (None, "v2"):
        loaded.clear()
        if flag is None:
            monkeypatch.delenv(FLAG, raising=False)
        else:
            monkeypatch.setenv(FLAG, flag)
            # sha-verified v2 needs real files; here only vscode's choice of directory is asserted
            monkeypatch.setattr(loader, "_verify_index_v2_manifest", lambda *a, **k: None)
        ModelStore.load_all(data_dir=tmp_path)
        assert f"similar_issue_index_{VSCODE}_bge" in loaded
        assert (V2 in loaded) is (flag == "v2")
        assert (OLD in loaded) is (flag is None)
