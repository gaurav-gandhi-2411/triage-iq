"""TRIAGE_VSCODE_CLASSIFIER=family_v2 (D92.1): bundle loader, fail-closed checks, raw-text hook.

Network-free and hermetic: every artifact is a tiny synthetic bundle written under tmp_path.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tests.test_vscode_naive_point import VSCODE, _assistant

from triage_iq.api.loader import ModelStore, _family_v2_requested
from triage_iq.models.component_classifier import (
    FAMILY_V2_FILENAME,
    FamilyV2ArtifactError,
    FamilyV2Classifier,
    load_family_v2_classifier,
)

FLAG = "TRIAGE_VSCODE_CLASSIFIER"
CLASSES = [
    "editor",
    "git",
    "terminal",
    "other",
]  # `other` is listed but never trained (family-only)


def _bundle(kind: str = "tfidf_lr_family_only_v2") -> dict:
    texts = [
        "editor cursor",
        "editor tab",
        "git commit",
        "git push",
        "terminal shell",
        "terminal pty",
    ]
    y = ["editor", "editor", "git", "git", "terminal", "terminal"]
    vec = TfidfVectorizer(ngram_range=(1, 2))
    model = LogisticRegression(C=4, max_iter=300, random_state=42).fit(vec.fit_transform(texts), y)
    return {"model_kind": kind, "vectorizer": vec, "model": model, "classes": CLASSES, "tau": 0.2,
            "conformal": {"qhat": 1.0}}  # fmt: skip


def _write(tmp_path: Path, *, listed: bool = True, sha: str | None = None, **kw: str) -> Path:
    f = tmp_path / FAMILY_V2_FILENAME
    joblib.dump(_bundle(**kw), f)
    digest = sha or hashlib.sha256(f.read_bytes()).hexdigest()
    lines = ["a" * 64 + "  data/models/component_classifier_microsoft_vscode.pkl"]
    if listed:
        lines.append(f"{digest}  data/models/{FAMILY_V2_FILENAME}")
    (tmp_path / "MANIFEST.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def test_loads_and_returns_aligned_probabilities(tmp_path: Path) -> None:
    _write(tmp_path)
    clf = load_family_v2_classifier(tmp_path)
    assert list(clf.classes_()) == CLASSES
    p = clf.predict_proba_calibrated(pd.Series(["git commit push", "terminal shell"]))
    assert p.shape == (2, 4)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p[:, CLASSES.index("other")] == 0).all()  # unseen class: exactly 0, never top-3
    assert list(clf.classes_()[np.argsort(p[0])[::-1][:1]]) == ["git"]


def test_serving_text_is_the_training_recipe() -> None:
    long_body = "x" * 5000
    assert FamilyV2Classifier.serving_text("T", long_body) == "T\n" + "x" * 4000
    assert FamilyV2Classifier.serving_text("T", "") == "T\n"


def test_no_manifest_entry_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, listed=False)  # the state on main: the bundle exists but is not pinned
    with pytest.raises(FamilyV2ArtifactError, match="0 entries"):
        load_family_v2_classifier(tmp_path)


def test_sha_mismatch_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, sha="0" * 64)
    with pytest.raises(FamilyV2ArtifactError, match="sha256"):
        load_family_v2_classifier(tmp_path)


def test_missing_file_and_missing_manifest_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(FamilyV2ArtifactError, match="unreadable"):
        load_family_v2_classifier(tmp_path)  # no manifest at all
    f = _write(tmp_path)
    f.unlink()
    with pytest.raises(FamilyV2ArtifactError, match="cannot read"):
        load_family_v2_classifier(tmp_path)


def test_wrong_model_kind_fails_closed(tmp_path: Path) -> None:
    _write(tmp_path, kind="multilabel_ovr")
    with pytest.raises(FamilyV2ArtifactError, match="model_kind"):
        load_family_v2_classifier(tmp_path)


def test_flag_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(FLAG, raising=False)
    assert _family_v2_requested() is False
    monkeypatch.setenv(FLAG, "")
    assert _family_v2_requested() is False
    monkeypatch.setenv(FLAG, "family_v2")
    assert _family_v2_requested() is True
    monkeypatch.setenv(FLAG, "family_v3")  # a typo must not silently serve the default
    with pytest.raises(FamilyV2ArtifactError):
        _family_v2_requested()


def test_startup_with_flag_on_and_no_manifest_line_does_not_skip_or_fall_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_all skips a repo whose models fail to load; a family_v2 failure must propagate."""
    (tmp_path / "models").mkdir()
    _write(tmp_path / "models", listed=False)
    monkeypatch.setenv(FLAG, "family_v2")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with pytest.raises(FamilyV2ArtifactError):
        ModelStore.load_all(data_dir=tmp_path)


def test_startup_with_flag_off_ignores_the_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "models").mkdir()
    _write(tmp_path / "models", listed=False)
    monkeypatch.delenv(FLAG, raising=False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="No repo models"):  # default path, as on main
        ModelStore.load_all(data_dir=tmp_path)


# --- the raw-text hook in TriageAssistant._collect_signals ---------------------------------------


class _Spy:
    """Records the text the classifier is given."""

    def __init__(self, raw: bool) -> None:
        self.uses_raw_body_text = raw
        self.seen: list[str] = []

    @staticmethod
    def serving_text(title: str, raw_body: str) -> str:
        return FamilyV2Classifier.serving_text(title, raw_body)

    def predict_proba_calibrated(self, texts: pd.Series) -> np.ndarray:
        self.seen.append(texts.iloc[0])
        return np.array([[0.7, 0.2, 0.1]])

    def classes_(self) -> list[str]:
        return ["a", "b", "c"]


def _signals(spy: _Spy, **fields: object) -> dict:
    asst = _assistant(VSCODE)
    asst.classifier = spy
    issue = pd.Series({"number": 5, "title": "T", "created_at": pd.Timestamp("2026-01-01", tz="UTC"),
                       **fields})  # fmt: skip
    return asst._collect_signals(issue)


def test_hook_feeds_raw_body_to_family_v2_and_cleaned_body_elsewhere() -> None:
    spy = _Spy(raw=True)
    sig = _signals(spy, body_clean="CLEAN", body_raw="R" * 5000)
    assert spy.seen == ["T\n" + "R" * 4000]
    assert [e["label"] for e in sig["classifier_top3"]] == ["a", "b", "c"]
    assert set(sig["classifier_top3"][0]) == {"label", "confidence"}


def test_hook_without_body_raw_uses_body_clean_which_is_then_the_raw_body() -> None:
    spy = _Spy(raw=True)
    _signals(spy, body_clean="the body")  # TRIAGE_CLEAN_REQUEST_BODY off: no body_raw key
    assert spy.seen == ["T\nthe body"]


def test_other_classifiers_keep_the_cleaned_text() -> None:
    spy = _Spy(raw=False)
    _signals(spy, body_clean="CLEAN", body_raw="RAW")
    assert spy.seen == ["T. CLEAN"]  # byte-identical to main
