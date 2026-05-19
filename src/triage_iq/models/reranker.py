"""Cross-encoder reranker for duplicate issue retrieval.

Wraps a sentence-transformers CrossEncoder to rerank FAISS candidates.
Used by DuplicateDetector as a second-stage filter: FAISS retrieves
top-RETRIEVAL_K, the reranker narrows to top-FINAL_K.

The model is lazy-loaded on first use so that import cost is zero when
the reranker is disabled (default behavior when no model_id is given).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Production default — chosen by W1.3 ADR-0006 benchmark.
# Smallest model within 1pp of the best R@5 on both repos.
DEFAULT_RERANKER_MODEL = "mixedbread-ai/mxbai-rerank-base-v1"


class Reranker:
    """Thin wrapper around a CrossEncoder for reranking candidate issues.

    Args:
        model_id: HuggingFace model ID. Defaults to DEFAULT_RERANKER_MODEL.
        max_length: Max token length passed to the cross-encoder. 512 covers
            most GitHub issue title+body snippets without truncation artifacts.
        trust_remote_code: Required by Jina rerankers; False for all others.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_RERANKER_MODEL,
        max_length: int = 512,
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self._model = None  # lazy-loaded

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # noqa: PLC0415
            logger.info("Loading reranker: %s", self.model_id)
            self._model = CrossEncoder(
                self.model_id,
                max_length=self.max_length,
                trust_remote_code=self.trust_remote_code,
            )
            logger.info("Reranker loaded: %s", self.model_id)

    def rerank(self, query: str, candidates: list[dict], top_k: int = 5) -> list[dict]:
        """Rerank candidates by relevance to query.

        Args:
            query: The query text (issue title + body snippet).
            candidates: List of dicts with at least {"number": int, "text": str, "score": float}.
                Typically the raw FAISS hits from DuplicateDetector.retrieve().
            top_k: Number of top candidates to return.

        Returns:
            Up to top_k candidates sorted by cross-encoder score (descending).
            Each dict gains a "reranker_score" key; the original "score" (FAISS
            cosine) is preserved as "faiss_score".
        """
        if not candidates:
            return []

        self._load()
        assert self._model is not None

        pairs = [(query, c["text"]) for c in candidates]
        scores: np.ndarray = self._model.predict(pairs, show_progress_bar=False)
        order = np.argsort(scores)[::-1]

        results = []
        for i in order[:top_k]:
            c = dict(candidates[i])
            c["reranker_score"] = float(scores[i])
            c["faiss_score"] = c.pop("score")
            c["score"] = c["reranker_score"]  # normalise key so callers are unaffected
            results.append(c)
        return results
