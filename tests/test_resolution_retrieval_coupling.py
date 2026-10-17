"""Coupling guard (ADR-0062 'Coupling'): k8s resolution features depend on retrieval's query recipe.

Since #150 the kubernetes resolution predictor's emb_* features are PCA(64) of the SAME query
embedding retrieval computes. The predictor was validated offline (and its cassette recorded)
against exactly one recipe. Changing ANY part of it -- query text assembly, the BGE query
instruction, the embedding model, normalisation, truncation, the PCA input dimension -- silently
moves the resolution numbers embedded in every k8s synthesis prompt without any test failing.
Every constant below is the value the model was validated against; a failure here means: stop,
re-validate the resolution metrics offline, re-record (ADR-0062 'Coupling' section), THEN update
the constant in the same PR. Network-free: SentenceTransformer.encode is replaced by a spy.

The live counterpart (real BGE weights vs eval/frozen_query_embeddings.npz) is
eval/test_resolution_query_coupling.py.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from triage_iq.models import similar_issues
from triage_iq.models.similar_issues import (
    QUERY_INSTRUCTION_REPO_OVERRIDE,
    QUERY_INSTRUCTIONS,
    SUPPORTED_MODELS,
    SimilarIssueRetriever,
)
from triage_iq.models.triage import RESOLUTION_EMBEDDINGS_REPOS, TriageAssistant

K8S = "kubernetes/kubernetes"
# The exact BGE-v1.5 query-side instruction the k8s resolution model's embeddings were built
# with (ADR-0040: ON for k8s, where it measured +6.67pp R@5). Dropping or rewording it shifts
# every k8s query vector and therefore every emb_* feature.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
EMBEDDING_DIM = 768  # BAAI/bge-base-en-v1.5 hidden size == PCA n_features_in_
PCA_COMPONENTS = 64


# --- 1. constants of the recipe ---------------------------------------------------------------


def test_embedding_model_is_pinned() -> None:
    # The PCA was fit on, and the index built with, this model's 768-d vectors.
    assert SUPPORTED_MODELS["bge"] == "BAAI/bge-base-en-v1.5"


def test_query_instruction_text_is_pinned() -> None:
    assert QUERY_INSTRUCTIONS["bge"] == BGE_QUERY_INSTRUCTION


def test_query_instruction_repo_override_is_pinned() -> None:
    # k8s: instruction ON (the recipe the resolution model saw). vscode: OFF, and vscode does
    # not consume the embedding at all, but flipping it would change retrieval hits in prompts.
    assert QUERY_INSTRUCTION_REPO_OVERRIDE == {
        "kubernetes_kubernetes": True,
        "microsoft_vscode": False,
    }


def test_resolution_embedding_repos_is_pinned() -> None:
    assert frozenset({K8S}) == RESOLUTION_EMBEDDINGS_REPOS


def test_query_side_does_not_override_truncation() -> None:
    """The query is truncated only by the model's own max_seq_length (512 for bge-base; asserted
    live in eval/test_resolution_query_coupling.py). Nothing in the retriever may change it or
    pass a max_length/truncate kwarg to encode()."""
    src = inspect.getsource(similar_issues.SimilarIssueRetriever)
    assert "max_seq_length =" not in src
    assert "truncate" not in inspect.getsource(SimilarIssueRetriever.retrieve_with_embedding)


# --- 2. query text assembly in triage.py --------------------------------------------------------


class _CaptureRetriever:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def retrieve_with_embedding(self, text, k=20, exclude_number=None):  # noqa: ANN001, ANN201
        self.texts.append(text)
        return [], None

    def retrieve(self, text, k=20, exclude_number=None):  # noqa: ANN001, ANN201
        self.texts.append(text)
        return []


class _Classifier:
    def predict_proba_calibrated(self, texts):  # noqa: ANN001, ANN201
        return np.array([[1.0]])

    def classes_(self):  # noqa: ANN201
        return ["a"]


def _query_text_for(title: str, body_clean: str) -> str:
    det = _CaptureRetriever()
    assistant = TriageAssistant(
        repo=K8S, classifier=_Classifier(), detector=det, predictor=object(),
        train_df=pd.DataFrame(), groq_api_key="test-key",
    )
    assistant._collect_signals(  # noqa: SLF001
        pd.Series({"title": title, "body_clean": body_clean, "number": 5})
    )
    assert len(det.texts) == 1
    return det.texts[0]


def test_query_text_is_title_dot_space_body_clean() -> None:
    assert _query_text_for("Pod stuck", "kubelet never schedules") == (
        "Pod stuck. kubelet never schedules"
    )


def test_query_text_has_no_character_truncation() -> None:
    body = "x" * 20_000  # far above any plausible char cap
    text = _query_text_for("T", body)
    assert text == "T. " + body
    assert len(text) == 20_003


# --- 3. what reaches SentenceTransformer.encode ---------------------------------------------


class _FakeIndex:
    def search(self, emb, k):  # noqa: ANN001, ANN201
        return np.array([[0.5]]), np.array([[0]])


class _SpyModel:
    """Stands in for SentenceTransformer: records encode() args, returns a non-unit float64
    vector so the float32 cast / shape contract of retrieve_with_embedding is exercised."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple, dict]] = []

    def encode(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN201
        self.calls.append((args, kwargs))
        return np.full((1, EMBEDDING_DIM), 0.25, dtype=np.float64)


def _retriever(repo: str) -> tuple[SimilarIssueRetriever, _SpyModel]:
    r = SimilarIssueRetriever.__new__(SimilarIssueRetriever)
    r.repo, r.model_key = repo, "bge"
    r.model = _SpyModel()  # type: ignore[assignment]
    r.index = _FakeIndex()  # type: ignore[assignment]
    r.issue_numbers = np.array([1], dtype=np.int64)
    r.texts = ["doc"]
    return r, r.model  # type: ignore[return-value]


def test_k8s_encode_call_is_instruction_prefixed_and_normalised() -> None:
    r, spy = _retriever("kubernetes_kubernetes")
    _, emb = r.retrieve_with_embedding("Pod stuck. body", k=5)
    assert len(spy.calls) == 1
    args, kwargs = spy.calls[0]
    # Exactly one text, prefixed; exactly these kwargs -- no batch_size / max_length / truncate.
    assert args == ([BGE_QUERY_INSTRUCTION + "Pod stuck. body"],)
    assert kwargs == {"normalize_embeddings": True, "convert_to_numpy": True}
    assert emb.dtype == np.float32 and emb.shape == (EMBEDDING_DIM,)


def test_vscode_encode_call_has_no_instruction() -> None:
    r, spy = _retriever("microsoft_vscode")
    r.retrieve_with_embedding("Crash on save. body", k=5)
    assert spy.calls[0][0] == (["Crash on save. body"],)
    assert spy.calls[0][1] == {"normalize_embeddings": True, "convert_to_numpy": True}


# --- 4. PCA input dimension -----------------------------------------------------------------


def _assistant_with_pca(pca) -> TriageAssistant:  # noqa: ANN001
    class _P:
        pass

    p = _P()
    p.pca = pca  # type: ignore[attr-defined]
    return TriageAssistant(
        repo=K8S, classifier=_Classifier(), detector=_CaptureRetriever(), predictor=p,
        train_df=pd.DataFrame(), groq_api_key="test-key",
    )


def test_pca_input_dim_must_match_embedding_dim() -> None:
    """A 768-d PCA is accepted; any other input dim silently falls back to zero-filled emb_*
    (fail-soft), which is exactly the drift this pin makes loud: the served PCA's
    n_features_in_ must equal the embedder's output dim."""
    rng = np.random.default_rng(42)
    ok = PCA(n_components=PCA_COMPONENTS, random_state=42).fit(rng.normal(size=(200, EMBEDDING_DIM)))
    bad = PCA(n_components=PCA_COMPONENTS, random_state=42).fit(rng.normal(size=(200, 384)))
    vec = rng.normal(size=EMBEDDING_DIM).astype(np.float32)
    assert ok.n_features_in_ == EMBEDDING_DIM
    assert "pca" in _assistant_with_pca(ok)._resolution_embedding_kwargs(vec)  # noqa: SLF001
    assert _assistant_with_pca(bad)._resolution_embedding_kwargs(vec) == {}  # noqa: SLF001


def test_served_predictor_pca_matches_recipe_when_artifact_present() -> None:
    from pathlib import Path

    from triage_iq.models.resolution import ResolutionTimePredictor

    path = Path(__file__).parent.parent / "data" / "models" / "resolution_predictor_kubernetes_kubernetes.pkl"
    if not path.exists():
        pytest.skip("served k8s predictor artifact not present (asserted in the eval job instead)")
    pca = ResolutionTimePredictor.load(str(path)).pca
    assert pca is not None
    assert pca.n_features_in_ == EMBEDDING_DIM
    assert pca.n_components_ == PCA_COMPONENTS
