"""ADR-0062: kubernetes/kubernetes computes the resolution predictor's emb_* features at serving
time from the retrieval query embedding; every other repo (microsoft/vscode) keeps the
zero-filled path unchanged.

Drives the real TriageAssistant._collect_signals() and the real engineer_features(), with
stubbed retriever / classifier / predictor so no model download or network is needed. The PCA is
a real sklearn PCA(64) fit on seeded random data.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

import triage_iq.models.resolution as resolution_mod
from triage_iq.models.resolution import engineer_features
from triage_iq.models.similar_issues import SimilarIssueRetriever
from triage_iq.models.triage import RESOLUTION_EMBEDDINGS_REPOS, TriageAssistant

K8S = "kubernetes/kubernetes"
VSCODE = "microsoft/vscode"
DIM = 768
N_COMP = 64
BASE_FEATURES = [
    "title_len_chars",
    "body_len_chars",
    "day_of_week",
    "days_since_repo_start",
    "author_prior_count",
]
EMB_FEATURES = [f"emb_{i}" for i in range(N_COMP)]
HITS = [{"number": 7, "score": 0.9, "text": "t"}]


def _unit(seed: int, dim: int = DIM) -> np.ndarray:
    v = np.random.default_rng(seed).normal(size=dim).astype(np.float32)
    return v / np.linalg.norm(v)


def _fit_pca(dim: int = DIM) -> PCA:
    X = np.random.default_rng(42).normal(size=(200, dim))
    return PCA(n_components=N_COMP, random_state=42).fit(X)


class _Classifier:
    def predict_proba_calibrated(self, texts):
        return np.array([[0.7, 0.3]])

    def classes_(self):
        return ["a", "b"]


class _Predictor:
    """Records the exact feature frame it is asked to predict on."""

    def __init__(self, pca: PCA | None) -> None:
        self.feature_names = BASE_FEATURES + EMB_FEATURES
        self.pca = pca
        self.seen: list[pd.DataFrame] = []

    def predict(self, X):
        self.seen.append(X.copy())
        return np.array([48.0])

    def predict_intervals(self, X):
        return np.array([24.0]), np.array([96.0])

    def predict_bucket(self, X):
        return ["days"], np.array([0.5])


class _EmbRetriever:
    """Has retrieve_with_embedding; retrieve() must not be needed on the k8s path."""

    def __init__(self, emb) -> None:
        self.emb = emb
        self.with_emb_calls = 0
        self.retrieve_calls = 0

    def retrieve(self, text, k=20, exclude_number=None):
        self.retrieve_calls += 1
        return list(HITS)

    def retrieve_with_embedding(self, text, k=20, exclude_number=None):
        self.with_emb_calls += 1
        return list(HITS), self.emb


class _PlainRetriever:
    """Only retrieve(): fails the test if the embedding API is touched."""

    def __init__(self) -> None:
        self.retrieve_calls = 0

    def retrieve(self, text, k=20, exclude_number=None):
        self.retrieve_calls += 1
        return list(HITS)

    def retrieve_with_embedding(self, *a, **kw):  # pragma: no cover - must not run
        raise AssertionError("retrieve_with_embedding must not be called for this repo")


class _RaisingRetriever:
    def retrieve(self, *a, **kw):
        raise RuntimeError("faiss down")

    def retrieve_with_embedding(self, *a, **kw):
        raise RuntimeError("faiss down")


def _train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": [1, 2, 3],
            "author": ["a", "b", "a"],
            "created_at": pd.to_datetime(
                ["2020-01-01", "2020-01-02", "2020-01-03"], utc=True
            ),
            "resolution_hours": [10.0, 20.0, 30.0],
            "state": ["closed"] * 3,
        }
    )


def _issue() -> pd.Series:
    return pd.Series(
        {
            "number": 99,
            "title": "Pod stuck in Pending",
            "body_clean": "kubelet never schedules the pod",
            "created_at": pd.Timestamp("2020-02-01", tz="UTC"),
        }
    )


def _assistant(repo: str, detector, predictor) -> TriageAssistant:
    return TriageAssistant(
        repo=repo,
        classifier=_Classifier(),
        detector=detector,
        predictor=predictor,
        train_df=_train_df(),
        groq_api_key="test-key",
    )


@pytest.fixture
def spy_engineer_features(monkeypatch):
    calls: list[dict] = []
    real = engineer_features

    def spy(df, train_df=None, embeddings=None, pca=None):
        calls.append({"embeddings": embeddings, "pca": pca})
        return real(df, train_df=train_df, embeddings=embeddings, pca=pca)

    monkeypatch.setattr(resolution_mod, "engineer_features", spy)
    return calls


def test_allowlist_is_k8s_only() -> None:
    assert frozenset({K8S}) == RESOLUTION_EMBEDDINGS_REPOS


def test_vscode_path_unchanged_zero_filled_and_no_embeddings_passed(spy_engineer_features) -> None:
    predictor = _Predictor(_fit_pca())
    detector = _PlainRetriever()
    sig = _assistant(VSCODE, detector, predictor)._collect_signals(_issue())

    assert detector.retrieve_calls == 1
    assert spy_engineer_features == [{"embeddings": None, "pca": None}]
    X = predictor.seen[0]
    assert (X[EMB_FEATURES].to_numpy() == 0.0).all()
    # Resolution stage ran normally (not the 7.0-day exception default).
    assert sig["pred_days"] == pytest.approx(2.0)

    # The frame the predictor saw equals what the pre-change code path builds.
    feats, _ = engineer_features(pd.DataFrame([_issue()]), train_df=_train_df())
    for col in predictor.feature_names:
        if col not in feats.columns:
            feats[col] = 0.0
    pd.testing.assert_frame_equal(X, feats[predictor.feature_names])


def test_k8s_emb_features_equal_pca_transform_of_query_embedding(spy_engineer_features) -> None:
    pca = _fit_pca()
    vec = _unit(1)
    predictor = _Predictor(pca)
    detector = _EmbRetriever(vec)
    _assistant(K8S, detector, predictor)._collect_signals(_issue())

    # One encode serves both System 2 and System 3.
    assert detector.with_emb_calls == 1
    assert detector.retrieve_calls == 0
    assert spy_engineer_features[0]["pca"] is pca
    assert spy_engineer_features[0]["embeddings"].shape == (1, DIM)

    expected = pca.transform(vec.reshape(1, -1))[0]
    got = predictor.seen[0][EMB_FEATURES].to_numpy()[0]
    assert np.array_equal(got, expected)
    assert np.abs(got).max() > 0.0
    # Non-embedding features are unaffected by the embeddings path.
    feats, _ = engineer_features(pd.DataFrame([_issue()]), train_df=_train_df())
    for col in BASE_FEATURES:
        assert predictor.seen[0][col].iloc[0] == feats[col].iloc[0]


@pytest.mark.parametrize(
    ("case", "predictor_pca", "detector", "reason"),
    [
        ("missing_pca", None, _EmbRetriever(_unit(2)), "predictor_has_no_pca"),
        ("wrong_dim_pca", _fit_pca(384), _EmbRetriever(_unit(3)), "embedding_dim_mismatch"),
        ("wrong_dim_vec", _fit_pca(), _EmbRetriever(_unit(4, 384)), "embedding_dim_mismatch"),
        ("no_embedding", _fit_pca(), _EmbRetriever(None), "no_query_embedding"),
        (
            "non_finite",
            _fit_pca(),
            _EmbRetriever(np.full(DIM, np.nan, dtype=np.float32)),
            "non_finite_embedding",
        ),
        ("retrieval_exception", _fit_pca(), _RaisingRetriever(), "no_query_embedding"),
    ],
)
def test_k8s_fallbacks_zero_fill_and_warn(
    case, predictor_pca, detector, reason, caplog, spy_engineer_features
) -> None:
    predictor = _Predictor(predictor_pca)
    with caplog.at_level(logging.WARNING, logger="triage_iq.models.triage"):
        sig = _assistant(K8S, detector, predictor)._collect_signals(_issue())

    assert spy_engineer_features == [{"embeddings": None, "pca": None}], case
    assert (predictor.seen[0][EMB_FEATURES].to_numpy() == 0.0).all(), case
    assert sig["pred_days"] == pytest.approx(2.0), case  # resolution stage did not fall to 7.0
    rec = [r for r in caplog.records if getattr(r, "reason", None) == reason]
    assert len(rec) == 1, (case, [r.getMessage() for r in caplog.records])
    assert rec[0].levelno == logging.WARNING
    assert rec[0].repo == K8S


def test_k8s_detector_without_embedding_api_falls_back(caplog, spy_engineer_features) -> None:
    class _OnlyRetrieve:
        def retrieve(self, text, k=20, exclude_number=None):
            return list(HITS)

    predictor = _Predictor(_fit_pca())
    with caplog.at_level(logging.WARNING, logger="triage_iq.models.triage"):
        _assistant(K8S, _OnlyRetrieve(), predictor)._collect_signals(_issue())
    assert spy_engineer_features == [{"embeddings": None, "pca": None}]
    assert any(getattr(r, "reason", None) == "no_query_embedding" for r in caplog.records)


# --- retrieve() backwards compatibility ------------------------------------------------------


class _StubModel:
    def __init__(self, vec: np.ndarray) -> None:
        self.vec = vec
        self.encoded: list[str] = []

    def encode(self, texts, **kwargs):
        self.encoded.extend(texts)
        return np.tile(self.vec, (len(texts), 1)).astype(np.float64)


def _retriever(vec: np.ndarray, repo: str = "kubernetes_kubernetes") -> SimilarIssueRetriever:
    r = SimilarIssueRetriever.__new__(SimilarIssueRetriever)
    r.repo = repo
    r.model_key = "bge"
    r.model = _StubModel(vec)
    docs = np.stack([vec, _unit(10, vec.shape[0]), _unit(11, vec.shape[0])]).astype(np.float32)
    r.index = faiss.IndexFlatIP(vec.shape[0])
    r.index.add(docs)
    r.issue_numbers = np.array([100, 200, 300], dtype=np.int64)
    r.texts = ["a", "b", "c"]
    return r


def test_retrieve_return_type_and_values_unchanged() -> None:
    vec = _unit(5, 16)
    r = _retriever(vec)
    hits = r.retrieve("q", k=2, exclude_number=100)
    assert isinstance(hits, list)
    assert [h["number"] for h in hits] == [h["number"] for h in r.retrieve_with_embedding(
        "q", k=2, exclude_number=100
    )[0]]
    assert 100 not in [h["number"] for h in hits]
    assert set(hits[0]) == {"number", "score", "text"}


def test_retrieve_with_embedding_returns_the_query_vector_with_instruction_applied() -> None:
    vec = _unit(6, 16)
    r = _retriever(vec)
    hits, emb = r.retrieve_with_embedding("hello", k=3)
    assert emb.dtype == np.float32
    assert emb.shape == (16,)
    assert np.array_equal(emb, vec)
    assert len(hits) == 3
    # k8s override applies the BGE query instruction, and only one encode happened.
    assert len(r.model.encoded) == 1
    assert r.model.encoded[0].startswith("Represent this sentence for searching")


def test_vscode_retrieval_has_no_instruction() -> None:
    r = _retriever(_unit(7, 16), repo="microsoft_vscode")
    r.retrieve("hello", k=1)
    assert r.model.encoded == ["hello"]


# --- eval FrozenRetriever --------------------------------------------------------------------


def _load_frozen_module():
    path = Path(__file__).resolve().parent.parent / "eval" / "frozen_retriever.py"
    spec = importlib.util.spec_from_file_location("frozen_retriever_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frozen_retriever_serves_frozen_embedding_and_none_when_absent() -> None:
    mod = _load_frozen_module()
    vec = _unit(8)
    fr = mod.FrozenRetriever({5: list(HITS)}, {5: vec})
    hits, emb = fr.retrieve_with_embedding("ignored", k=5, exclude_number=5)
    assert hits == HITS
    assert np.array_equal(emb, vec)
    assert fr.retrieve("ignored", k=5, exclude_number=5) == HITS
    assert fr.retrieve_with_embedding("ignored", k=5, exclude_number=6) == ([], None)
    assert mod.FrozenRetriever({5: list(HITS)}).retrieve_with_embedding("x", 5, 5)[1] is None
