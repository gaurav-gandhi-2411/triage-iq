"""Live coupling guard (ADR-0062 'Coupling'): the production query-encoding path must still
reproduce eval/frozen_query_embeddings.npz.

The k8s resolution model's emb_* features are PCA(64) of the retrieval query embedding, and the
eval replay uses the FROZEN vectors in that npz. If the embedding model, its weights, the query
instruction, normalisation, truncation or the query-text assembly changes, production would
feed the predictor vectors that no longer match what the cassette was recorded and the model
validated against -- while every offline test (which replays the frozen vectors) stays green.
This test encodes 3 fixed k8s eval issues LIVE through SimilarIssueRetriever (the production
class, loaded from the served FAISS index dir) and requires cosine >= 0.9999 against the frozen
vectors. Runs in the eval-gate quality job, where BAAI/bge-base-en-v1.5 is cached and the
production index is downloaded. Skips locally only if those artifacts are absent.

If this fails: do NOT just re-freeze. Re-validate the resolution metrics offline against the new
recipe, re-record the k8s cassette entries, then re-freeze (ADR-0062 'Coupling').
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
EVAL_SET = ROOT / "eval" / "eval_set.jsonl"
FROZEN = ROOT / "eval" / "frozen_query_embeddings.npz"
K8S = "kubernetes/kubernetes"
INDEX_DIR = ROOT / "data" / "models" / "similar_issue_index_kubernetes_kubernetes_bge"
# Fixed, spread across the eval set (short and long bodies); changing these is a test edit.
ISSUE_IDS = ("k8s-12122", "k8s-13508", "k8s-14935")
MIN_COSINE = 0.9999  # float32 CPU re-encode is ~1.0; 1e-4 slack for BLAS/library-version noise
EXPECTED_MAX_SEQ_LENGTH = 512  # query truncation is the model's own, never overridden


@pytest.fixture(scope="module")
def retriever():  # noqa: ANN201
    if not INDEX_DIR.exists():
        pytest.skip(f"served k8s index not present: {INDEX_DIR}")
    from triage_iq.models.similar_issues import SimilarIssueRetriever

    return SimilarIssueRetriever.load(str(INDEX_DIR))


def _frozen() -> dict[int, np.ndarray]:
    with np.load(FROZEN, allow_pickle=False) as z:
        return {
            int(n): np.asarray(e, dtype=np.float32)
            for r, n, e in zip(z["repos"].tolist(), z["numbers"].tolist(), z["embeddings"], strict=True)
            if r == K8S
        }


def _issues() -> list[dict]:
    rows = [json.loads(ln) for ln in EVAL_SET.read_text(encoding="utf-8").splitlines() if ln.strip()]
    by_id = {r["id"]: r for r in rows}
    return [by_id[i] for i in ISSUE_IDS]


def test_model_truncation_length_is_unchanged(retriever) -> None:  # noqa: ANN001
    assert retriever.model.max_seq_length == EXPECTED_MAX_SEQ_LENGTH


def test_frozen_vectors_have_the_pca_input_shape() -> None:
    vecs = _frozen()
    assert vecs, "no k8s vectors frozen"
    assert all(v.shape == (768,) and v.dtype == np.float32 for v in vecs.values())


@pytest.mark.parametrize("issue", _issues(), ids=ISSUE_IDS)
def test_live_encoding_matches_frozen_query_embedding(retriever, issue: dict) -> None:  # noqa: ANN001
    # Same text triage._collect_signals builds and same call it makes for k8s (ADR-0062).
    text = f"{issue['title']}. {issue['body']}"
    _, live = retriever.retrieve_with_embedding(text, k=5, exclude_number=int(issue["number"]))
    frozen = _frozen()[int(issue["number"])]
    cos = float(np.dot(live, frozen) / (np.linalg.norm(live) * np.linalg.norm(frozen)))
    assert cos >= MIN_COSINE, (
        f"{issue['id']}: live query embedding diverged from eval/frozen_query_embeddings.npz "
        f"(cosine {cos:.6f} < {MIN_COSINE}). The retrieval recipe changed: re-validate resolution "
        "metrics offline and re-record before updating the frozen vectors (ADR-0062 Coupling)."
    )


def test_served_k8s_predictor_pca_consumes_these_vectors() -> None:
    path = ROOT / "data" / "models" / "resolution_predictor_kubernetes_kubernetes.pkl"
    if not path.exists():
        pytest.skip("served k8s predictor not present")
    from triage_iq.models.resolution import ResolutionTimePredictor

    pca = ResolutionTimePredictor.load(str(path)).pca
    assert pca is not None and pca.n_features_in_ == 768 and pca.n_components_ == 64
