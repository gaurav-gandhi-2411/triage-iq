"""Sentence embedding + FAISS retrieval for duplicate issue detection.

One DuplicateDetector per repo. Supports BGE and MiniLM embeddings.
Index built over all issues; retrieval excludes the query issue itself.

Optional two-stage retrieval: if a Reranker is attached, retrieve
FAISS_RERANK_K candidates from FAISS, then rerank to the requested top-k.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

if TYPE_CHECKING:
    from triage_iq.models.reranker import Reranker

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {
    "bge": "BAAI/bge-base-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

# Number of FAISS candidates to retrieve before reranking.
# Must be >= the final k passed to retrieve().
FAISS_RERANK_K = 50


def _build_text(title: pd.Series, body: pd.Series, max_body: int = 512) -> list[str]:
    titles = title.fillna("").str.strip()
    bodies = body.fillna("").str.strip().str[:max_body]
    return (titles + ". " + bodies).tolist()


class DuplicateDetector:
    """Sentence embedding + FAISS IndexFlatIP retrieval for duplicate detection."""

    def __init__(
        self,
        repo: str,
        model_key: str = "bge",
        reranker: "Reranker | None" = None,
    ) -> None:
        self.repo = repo
        self.model_key = model_key
        self.reranker = reranker
        model_name = SUPPORTED_MODELS.get(model_key, model_key)
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.issue_numbers: np.ndarray | None = None
        self.texts: list[str] | None = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, df: pd.DataFrame) -> "DuplicateDetector":
        """Embed all issues and build inner-product (cosine) FAISS index.

        Embeddings are L2-normalised so inner product == cosine similarity.
        """
        self.texts = _build_text(df["title"], df["body_clean"])
        self.issue_numbers = df["number"].values.astype(np.int64)

        logger.info("[%s/%s] Encoding %d issues...", self.repo, self.model_key, len(self.texts))
        embs = self.model.encode(
            self.texts,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        dim = embs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embs)
        logger.info("[%s/%s] Index built: %d vectors, dim=%d", self.repo, self.model_key, len(self.texts), dim)
        return self

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def retrieve(self, query_text: str, k: int = 5, exclude_number: int | None = None) -> list[dict]:
        """Return top-k most similar issues (excluding query issue itself).

        When a reranker is attached: retrieves FAISS_RERANK_K candidates from
        FAISS first, then reranks to top-k via cross-encoder. When no reranker
        is set: returns top-k directly from FAISS (original behaviour).
        """
        if self.reranker is not None:
            # Two-stage: FAISS → cross-encoder rerank
            faiss_k = max(k, FAISS_RERANK_K)
            raw_hits = self._faiss_retrieve(query_text, faiss_k, exclude_number)
            return self.reranker.rerank(query_text, raw_hits, top_k=k)
        return self._faiss_retrieve(query_text, k, exclude_number)

    def _faiss_retrieve(
        self, query_text: str, k: int, exclude_number: int | None = None
    ) -> list[dict]:
        """Raw FAISS retrieval — no reranking."""
        assert self.index is not None, "Call build_index first"
        assert self.issue_numbers is not None
        assert self.texts is not None
        emb = self.model.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        # Retrieve k+1 to account for possible self-exclusion
        scores, indices = self.index.search(emb, k + 1)
        results = []
        for score, idx in zip(scores[0], indices[0], strict=False):
            if idx < 0:
                continue
            num = int(self.issue_numbers[idx])
            if exclude_number is not None and num == exclude_number:
                continue
            results.append({"number": num, "score": float(score), "text": self.texts[idx]})
            if len(results) >= k:
                break
        return results

    def retrieve_batch(self, query_texts: list[str], k: int = 20) -> list[list[dict]]:
        """Batch retrieval without self-exclusion."""
        assert self.index is not None
        assert self.issue_numbers is not None
        embs = self.model.encode(
            query_texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)
        scores_all, indices_all = self.index.search(embs, k)
        results = []
        for scores, indices in zip(scores_all, indices_all, strict=False):
            hits = []
            for score, idx in zip(scores, indices, strict=False):
                if idx >= 0:
                    hits.append({"number": int(self.issue_numbers[idx]), "score": float(score)})
            results.append(hits)
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, out_dir: str) -> None:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        joblib.dump({
            "repo": self.repo,
            "model_key": self.model_key,
            "issue_numbers": self.issue_numbers,
            "texts": self.texts,
        }, str(p / "meta.pkl"))
        logger.info("Saved index to %s", out_dir)

    @classmethod
    def load(cls, out_dir: str) -> "DuplicateDetector":
        p = Path(out_dir)
        meta = joblib.load(str(p / "meta.pkl"))
        obj = cls(repo=meta["repo"], model_key=meta["model_key"])
        obj.index = faiss.read_index(str(p / "index.faiss"))
        obj.issue_numbers = meta["issue_numbers"]
        obj.texts = meta["texts"]
        return obj
