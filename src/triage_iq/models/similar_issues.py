"""Sentence embedding + FAISS retrieval for related-issue retrieval.

One SimilarIssueRetriever per repo. Supports BGE and MiniLM embeddings.
Index built over all issues; retrieval excludes the query issue itself.

Task context: retrieves the most semantically related historical issues given a
new issue's title + body. Relatedness is supervised by PR→issue references and
text-similarity pairs (see ADR-0008 for task framing).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol, cast

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Structural interface for the HF tokenizer _build_text() needs -- matches
    SentenceTransformer.tokenizer's encode()/decode() (transformers.PreTrainedTokenizerBase),
    without importing transformers just for a type hint."""

    def encode(
        self,
        text: str,
        add_special_tokens: bool = ...,
        truncation: bool = ...,
        max_length: int | None = ...,
    ) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...

SUPPORTED_MODELS = {
    "bge": "BAAI/bge-base-en-v1.5",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
}

# BGE-v1.5's documented query-side instruction prefix (asymmetric: passages/documents are
# NOT prefixed, only queries -- this is how BGE was trained/is intended to be used). Applied
# in retrieve()/retrieve_batch() only, never in build_index(), and via the same shared class
# both prod (loader.py) and eval (d1_baseline_eval.py etc.) call, so prod and eval can't drift
# out of the same embedding space. See LEVER 2, measured against LEVER 1's corpus-truncation fix.
QUERY_INSTRUCTIONS: dict[str, str] = {
    "bge": "Represent this sentence for searching relevant passages: ",
    "minilm": "",
}

# Per-repo override for whether to apply this model's query instruction (ADR-0040, LEVER 2).
# Measured on D1's frozen eval sets (paired bootstrap, reports/lever12_eval_results.json):
# k8s_related R@5 +6.67pp CI[+2.67,+10.67] with the instruction ON (excludes zero, real);
# vscode_duplicate R@5 the instruction ALONE (isolated from lever 1) moves -2.00pp
# CI[-5.0,+1.0] -- doesn't clear significance, but it's directionally negative and it erases
# lever 1's own positive trend on that repo (53.5% -> 51.5%). GG's call: encode the asymmetry
# rather than average over it -- shipping a change with a known-negative direction on one repo
# for the sake of a uniform config is accepting a real (if unproven) cost for code simplicity.
# Working hypothesis, NOT confirmed: vscode's task is near-duplicate matching, where BGE's
# "searching relevant passages" framing may dilute the exact-match lexical signal that task
# depends on; k8s's task is genuinely semantic relatedness, where the framing fits. If a
# future eval shows the instruction actually helps vscode, flip this -- don't leave it stale
# out of inertia. Repos not listed here fall back to the model's QUERY_INSTRUCTIONS default.
QUERY_INSTRUCTION_REPO_OVERRIDE: dict[str, bool] = {
    "kubernetes_kubernetes": True,
    "microsoft_vscode": False,
}


def _build_text(
    title: pd.Series,
    body: pd.Series,
    tokenizer: Tokenizer | None = None,
    max_tokens: int = 512,
    max_body: int = 512,
) -> list[str]:
    """Build "title. body" corpus text.

    With a tokenizer: truncates body by TOKEN count (reserving room for the title and the
    model's [CLS]/[SEP] special tokens) so the encoded text fills the model's actual sequence
    budget, instead of a fixed 512-CHARACTER cut that discards far more content than the model
    can use for any issue with a body longer than a couple hundred characters -- see LEVER 1,
    reports/lever1_truncation_measurement.json (the prior character cut fit as little as
    ~25-30% of a long vscode issue's true content into BGE's 512-token window).

    Without a tokenizer (legacy/back-compat path, e.g. callers without a loaded model handy):
    falls back to the old character-based cut at `max_body`.
    """
    titles = title.fillna("").str.strip()
    bodies = body.fillna("").str.strip()
    if tokenizer is None:
        return (titles + ". " + bodies.str[:max_body]).tolist()

    texts = []
    for t, b in zip(titles.tolist(), bodies.tolist(), strict=True):
        prefix = f"{t}. "
        prefix_n_tokens = len(tokenizer.encode(prefix, add_special_tokens=False))
        # -2 reserves room for the [CLS]/[SEP] special tokens BERT-family tokenizers add.
        body_budget = max(max_tokens - prefix_n_tokens - 2, 0)
        body_ids = tokenizer.encode(
            b, add_special_tokens=False, truncation=True, max_length=body_budget
        )
        truncated_body = tokenizer.decode(body_ids, skip_special_tokens=True)
        texts.append(prefix + truncated_body)
    return texts


class SimilarIssueRetriever:
    """Sentence embedding + FAISS IndexFlatIP retrieval for related-issue retrieval."""

    def __init__(self, repo: str, model_key: str = "bge") -> None:
        self.repo = repo
        self.model_key = model_key
        model_name = SUPPORTED_MODELS.get(model_key, model_key)
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.index: faiss.IndexFlatIP | None = None
        self.issue_numbers: np.ndarray | None = None
        self.texts: list[str] | None = None

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def build_index(self, df: pd.DataFrame) -> SimilarIssueRetriever:
        """Embed all issues and build inner-product (cosine) FAISS index.

        Embeddings are L2-normalised so inner product == cosine similarity.
        """
        self.texts = _build_text(
            df["title"],
            df["body_clean"],
            tokenizer=self.model.tokenizer,
            max_tokens=self.model.max_seq_length,
        )
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
        logger.info(
            "[%s/%s] Index built: %d vectors, dim=%d",
            self.repo,
            self.model_key,
            len(self.texts),
            dim,
        )
        return self

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def _apply_query_instruction(self, text: str, apply_query_instruction: bool | None) -> str:
        """Prefix `text` with this model's query-side instruction (BGE only; no-op for MiniLM).

        Resolution order for apply_query_instruction=None (all real callers, prod included):
          1. QUERY_INSTRUCTION_REPO_OVERRIDE[self.repo], if this repo has one (ADR-0040) --
             a per-repo decision beats the model-level default, since the same model can
             behave asymmetrically across genuinely different retrieval tasks (see ADR-0040).
          2. Otherwise, the model's own QUERY_INSTRUCTIONS default -- True for bge, False for
             minilm.
        The explicit True/False override exists only so eval scripts can A/B the instruction's
        effect in isolation, ignoring both the repo and model defaults; prod code should never
        pass it.
        """
        if apply_query_instruction is not None:
            use = apply_query_instruction
        elif self.repo in QUERY_INSTRUCTION_REPO_OVERRIDE:
            use = QUERY_INSTRUCTION_REPO_OVERRIDE[self.repo]
        else:
            use = bool(QUERY_INSTRUCTIONS.get(self.model_key, ""))
        return QUERY_INSTRUCTIONS.get(self.model_key, "") + text if use else text

    def retrieve(
        self,
        query_text: str,
        k: int = 20,
        exclude_number: int | None = None,
        apply_query_instruction: bool | None = None,
    ) -> list[dict]:
        """Return top-k most related issues (excluding query issue itself)."""
        assert self.index is not None, "Call build_index first"
        assert self.issue_numbers is not None
        assert self.texts is not None
        query_text = self._apply_query_instruction(query_text, apply_query_instruction)
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

    def retrieve_batch(
        self,
        query_texts: list[str],
        k: int = 20,
        apply_query_instruction: bool | None = None,
    ) -> list[list[dict]]:
        """Batch retrieval without self-exclusion."""
        assert self.index is not None
        assert self.issue_numbers is not None
        query_texts = [
            self._apply_query_instruction(t, apply_query_instruction) for t in query_texts
        ]
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
        assert self.index is not None, "Call build_index first"
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(p / "index.faiss"))
        joblib.dump(
            {
                "repo": self.repo,
                "model_key": self.model_key,
                "issue_numbers": self.issue_numbers,
                "texts": self.texts,
            },
            str(p / "meta.pkl"),
        )
        logger.info("Saved index to %s", out_dir)

    @classmethod
    def load(cls, out_dir: str) -> SimilarIssueRetriever:
        p = Path(out_dir)
        meta = joblib.load(str(p / "meta.pkl"))
        obj = cls(repo=meta["repo"], model_key=meta["model_key"])
        # faiss.read_index()'s return type is the generic Index base class, but save()
        # only ever writes an IndexFlatIP -- narrowing here matches the actual contract.
        obj.index = cast(faiss.IndexFlatIP, faiss.read_index(str(p / "index.faiss")))
        obj.issue_numbers = meta["issue_numbers"]
        obj.texts = meta["texts"]
        return obj
