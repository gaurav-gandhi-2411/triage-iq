"""Unit tests for the cross-encoder Reranker and DuplicateDetector integration.

CrossEncoder loading is mocked so tests run without downloading model weights.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from triage_iq.models.reranker import DEFAULT_RERANKER_MODEL, Reranker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_candidates(n: int = 10) -> list[dict]:
    """Return n fake FAISS hit dicts."""
    return [
        {"number": 100 + i, "score": float(n - i) / n, "text": f"issue body {i}"}
        for i in range(n)
    ]


@pytest.fixture
def mock_cross_encoder():
    """Patch CrossEncoder so no model weights are downloaded."""
    with patch("triage_iq.models.reranker.Reranker._load") as mock_load:
        yield mock_load


# ---------------------------------------------------------------------------
# DEFAULT_RERANKER_MODEL
# ---------------------------------------------------------------------------

def test_default_model_is_mxbai():
    assert DEFAULT_RERANKER_MODEL == "mixedbread-ai/mxbai-rerank-base-v1"


# ---------------------------------------------------------------------------
# Reranker.rerank()
# ---------------------------------------------------------------------------

def test_rerank_returns_top_k(mock_cross_encoder):
    reranker = Reranker()
    # Inject a fake _model that returns ascending scores (worst → best order)
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([float(i) for i in range(10)])
    reranker._model = fake_model

    candidates = _make_candidates(10)
    result = reranker.rerank("query text", candidates, top_k=5)

    assert len(result) == 5
    # Highest score was index 9 (score=9.0) — should be first
    assert result[0]["number"] == 109
    assert result[0]["reranker_score"] == pytest.approx(9.0)


def test_rerank_preserves_faiss_score(mock_cross_encoder):
    reranker = Reranker()
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([1.0, 0.5])
    reranker._model = fake_model

    candidates = [
        {"number": 1, "score": 0.9, "text": "a"},
        {"number": 2, "score": 0.8, "text": "b"},
    ]
    result = reranker.rerank("q", candidates, top_k=2)

    # Both entries should have faiss_score preserved and score set to reranker_score
    for r in result:
        assert "faiss_score" in r
        assert r["score"] == r["reranker_score"]


def test_rerank_empty_candidates_returns_empty(mock_cross_encoder):
    reranker = Reranker()
    assert reranker.rerank("query", [], top_k=5) == []


def test_rerank_fewer_candidates_than_top_k(mock_cross_encoder):
    reranker = Reranker()
    fake_model = MagicMock()
    fake_model.predict.return_value = np.array([0.8, 0.6])
    reranker._model = fake_model

    candidates = _make_candidates(2)
    result = reranker.rerank("query", candidates, top_k=10)

    assert len(result) == 2  # can't return more than available


def test_reranker_is_lazy_loaded():
    """_model should be None until rerank() is called."""
    reranker = Reranker()
    assert reranker._model is None


# ---------------------------------------------------------------------------
# DuplicateDetector with reranker
# ---------------------------------------------------------------------------

def test_detector_reranker_attribute_defaults_none():
    from triage_iq.models.duplicates import DuplicateDetector
    with patch("triage_iq.models.duplicates.SentenceTransformer"):
        det = DuplicateDetector(repo="test/repo", model_key="bge")
    assert det.reranker is None


def test_detector_accepts_reranker(mock_cross_encoder):
    from triage_iq.models.duplicates import DuplicateDetector
    with patch("triage_iq.models.duplicates.SentenceTransformer"):
        reranker = Reranker()
        det = DuplicateDetector(repo="test/repo", model_key="bge", reranker=reranker)
    assert det.reranker is reranker


def test_detector_retrieve_calls_reranker_when_set(mock_cross_encoder):
    """When reranker is attached, retrieve() should call reranker.rerank()."""
    from triage_iq.models.duplicates import DuplicateDetector, FAISS_RERANK_K

    with patch("triage_iq.models.duplicates.SentenceTransformer"):
        reranker = MagicMock(spec=Reranker)
        reranker.rerank.return_value = [{"number": 42, "score": 0.9, "text": "x"}]

        det = DuplicateDetector(repo="test/repo", model_key="bge", reranker=reranker)

    # Patch _faiss_retrieve to return fake hits without a real index
    det._faiss_retrieve = MagicMock(return_value=_make_candidates(FAISS_RERANK_K))

    result = det.retrieve("query", k=5)

    det._faiss_retrieve.assert_called_once()
    # _faiss_retrieve should have been called with FAISS_RERANK_K
    _, kwargs_or_args = det._faiss_retrieve.call_args
    reranker.rerank.assert_called_once()
    assert result == [{"number": 42, "score": 0.9, "text": "x"}]


def test_detector_retrieve_skips_reranker_when_none():
    """When no reranker, retrieve() delegates directly to _faiss_retrieve."""
    from triage_iq.models.duplicates import DuplicateDetector

    with patch("triage_iq.models.duplicates.SentenceTransformer"):
        det = DuplicateDetector(repo="test/repo", model_key="bge")

    det._faiss_retrieve = MagicMock(return_value=_make_candidates(5))

    result = det.retrieve("query", k=5)

    det._faiss_retrieve.assert_called_once_with("query", 5, None)
    assert len(result) == 5
