"""Phase 3b flag 1: packaged P4 priority model equals the original on a tiny synthetic fit.

Hermetic: fits the real scripts/ Featurizer on a handful of texts in tmp_path; no D: access.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from triage_iq.models.priority import (
    PRIORITY_CLASSES,
    PriorityModel,
    build_text,
    model_rationale,
    rationale_mentions_other_level,
)

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))  # qp1_tfidf_features lives in scripts/
TITLES = ["kubelet crash", "docs typo", "scheduler panic", "readme fix", "etcd data loss", "nit"]
BODIES = [
    "panic in kubelet pod",
    "typo in docs page",
    "scheduler panic nil",
    "fix readme",
    "etcd lost data",
    "",
]
LABELS = ["high", "low", "high", "low", "high", "medium"]
# Every doc is repeated so the Featurizer's min_df=2 keeps a vocabulary.
TITLES, BODIES, LABELS = TITLES * 2, BODIES * 2, LABELS * 2


def _load_packager():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location(
        "qp3b_package_priority_model", SCRIPTS / "qp3b_package_priority_model.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fitted(tmp_path: Path):  # noqa: ANN201
    import qp1_tfidf_features as tf

    feat = tf.Featurizer(True, False)
    texts = pd.Series([build_text(t, b) for t, b in zip(TITLES, BODIES, strict=True)])
    x = feat.fit_transform(texts)
    # min_df=2 on 6 texts leaves few features; fine for an equivalence test
    lr = LogisticRegression(max_iter=300, random_state=42).fit(x, LABELS)
    fp, mp = tmp_path / "featurizer.joblib", tmp_path / "model.joblib"
    joblib.dump(feat, fp)
    joblib.dump(lr, mp)
    return feat, lr, fp, mp


def test_packaged_predictor_equals_original(fitted, tmp_path: Path) -> None:  # noqa: ANN001
    feat, lr, fp, mp = fitted
    pk = _load_packager()
    out = tmp_path / "packaged.joblib"
    joblib.dump(pk.package(fp, mp), out)
    model = PriorityModel.load(out)

    titles = [*TITLES, "kubelet pod\nsecond line", ""]
    bodies = [*BODIES, "x" * 5000 + " panic", "docs"]
    texts = [build_text(t, b) for t, b in zip(titles, bodies, strict=True)]
    orig = np.zeros((len(texts), 3))
    orig[:, [PRIORITY_CLASSES.index(c) for c in lr.classes_]] = lr.predict_proba(
        feat.transform(pd.Series(texts))
    )
    assert np.abs(orig - model.proba_texts(texts)).max() < 1e-12
    p = model.predict(TITLES[0], BODIES[0])
    assert p.label == PRIORITY_CLASSES[int(orig[0].argmax())]
    assert sum(p.proba.values()) == pytest.approx(1.0)


def test_packaged_artifact_has_no_scripts_dependency(fitted, tmp_path: Path) -> None:  # noqa: ANN001
    _, _, fp, mp = fitted
    out = tmp_path / "packaged.joblib"
    joblib.dump(_load_packager().package(fp, mp), out)
    assert b"qp1_tfidf_features" not in out.read_bytes()  # global refs are stored as plain text


def test_load_rejects_foreign_artifact(tmp_path: Path) -> None:
    bad = tmp_path / "x.joblib"
    joblib.dump({"format_version": 99}, bad)
    with pytest.raises(ValueError, match="not a priority-model artifact"):
        PriorityModel.load(bad)


@pytest.mark.parametrize(
    ("text", "served", "expected"),
    [
        ("This is a high priority crash.", "high", False),
        ("Low impact, but high severity for users.", "high", True),
        ("Lower the risk; highlight docs", "medium", False),  # substrings are not levels
        ("area/low-level is unrelated", "high", False),
        ("Medium urgency.", "low", True),
    ],
)
def test_rationale_contradiction_detection(text: str, served: str, expected: bool) -> None:
    assert rationale_mentions_other_level(text, served) is expected


def test_model_rationale_cites_probability() -> None:
    proba = {"low": 0.2, "medium": 0.3, "high": 0.5}
    r = model_rationale("high", proba, "low")
    assert "high (p=0.50" in r and "the LLM proposed low" in r
    assert "proposed" not in model_rationale("high", proba, "high")
