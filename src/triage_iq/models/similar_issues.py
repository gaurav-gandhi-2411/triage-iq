"""Sentence embedding + FAISS retrieval for related-issue retrieval.

One SimilarIssueRetriever per repo. Supports BGE and MiniLM embeddings.
Index built over all issues; retrieval excludes the query issue itself.

Task context: retrieves the most semantically related historical issues given a
new issue's title + body. Relatedness is supervised by PR→issue references and
text-similarity pairs (see ADR-0008 for task framing).
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Protocol, cast

import faiss
import joblib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MissingNeighbourDateError(ValueError):
    """Strict created-before retrieval met a neighbour (or an index) with no created_at."""


def to_utc_datetime64(value: object) -> np.datetime64:
    """Normalise a datetime / pandas Timestamp / ISO string to a naive-UTC datetime64[ns].

    Naive inputs are taken as UTC. Raises ValueError on NaT/None/unparseable input.
    """
    ts = pd.Timestamp(value)  # type: ignore[arg-type]
    if ts is pd.NaT:
        raise ValueError(f"not a valid timestamp: {value!r}")
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return np.datetime64(ts.tz_localize(None).to_datetime64(), "ns")


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

    # Class-level defaults so instances built without __init__ (tests) still see "no dates".
    created_at: np.ndarray | None = None
    _warned_dateless: bool = False

    def __init__(self, repo: str, model_key: str = "bge") -> None:
        self.repo = repo
        self.model_key = model_key
        model_name = SUPPORTED_MODELS.get(model_key, model_key)
        logger.info("Loading embedding model: %s", model_name)
        t0 = time.perf_counter()
        self.model = SentenceTransformer(model_name)
        logger.info(
            "startup_step %s.sentence_transformer_load took %.1f ms",
            repo, (time.perf_counter() - t0) * 1000.0,
        )
        self.index: faiss.IndexFlatIP | None = None
        self.issue_numbers: np.ndarray | None = None
        self.texts: list[str] | None = None
        # Optional per-issue creation time (naive-UTC datetime64[ns], NaT = unknown), aligned
        # with issue_numbers. None for indexes built/saved before the created-before filter.
        self.created_at: np.ndarray | None = None
        self._warned_dateless = False

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
        if "created_at" in df.columns:
            ts = pd.to_datetime(df["created_at"], utc=True).dt.tz_localize(None)
            self.created_at = ts.to_numpy().astype("datetime64[ns]")

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

    def match_indexed_issue(self, title: str, body: str) -> int | None:
        """Number of the indexed issue whose stored corpus text is exactly this title+body.

        Users usually paste an issue that already exists, and without an issue_number the
        index returns the issue itself as its own top "similar issue" (seen live on vscode
        #286776, 2026-09-24). The corpus text is built from the CLEANED body
        (preprocess.clean_text) through _build_text with this model's tokenizer, so the
        query is canonicalized the same way before the lookup. Verified on #286776: the
        raw GitHub body cleaned this way reproduces the indexed text byte-for-byte; the
        raw body does not. The body is also tried as-is, for callers that already pass a
        cleaned body. Exact-text equality only, never a similarity threshold: a
        near-duplicate filed separately is a real result and must stay. If several indexed
        issues share the text, nothing is excluded (ambiguous).
        """
        from triage_iq.data.preprocess import clean_text

        if self.texts is None or self.issue_numbers is None:
            return None
        lookup = getattr(self, "_text_to_numbers", None)
        if lookup is None:
            lookup = {}
            for text, num in zip(self.texts, self.issue_numbers, strict=True):
                lookup.setdefault(text, []).append(int(num))
            self._text_to_numbers = lookup
        tokenizer = getattr(self.model, "tokenizer", None)
        max_tokens = getattr(self.model, "max_seq_length", 512)
        for candidate_body in dict.fromkeys((clean_text(body)[0], body)):
            built = _build_text(
                pd.Series([title]), pd.Series([candidate_body]),
                tokenizer=tokenizer, max_tokens=max_tokens,
            )[0]
            nums = lookup.get(built)
            if nums is not None and len(nums) == 1:
                return nums[0]
        return None

    def retrieve(
        self,
        query_text: str,
        k: int = 20,
        exclude_number: int | None = None,
        apply_query_instruction: bool | None = None,
        created_before: object | None = None,
        strict_dates: bool = False,
    ) -> list[dict]:
        """Return top-k most related issues (excluding query issue itself).

        created_before / strict_dates: see retrieve_with_embedding().
        """
        hits, _ = self.retrieve_with_embedding(
            query_text,
            k=k,
            exclude_number=exclude_number,
            apply_query_instruction=apply_query_instruction,
            created_before=created_before,
            strict_dates=strict_dates,
        )
        return hits

    def retrieve_with_embedding(
        self,
        query_text: str,
        k: int = 20,
        exclude_number: int | None = None,
        apply_query_instruction: bool | None = None,
        created_before: object | None = None,
        strict_dates: bool = False,
    ) -> tuple[list[dict], np.ndarray]:
        """Same as retrieve(), also returning the query embedding it computed.

        The embedding is the exact vector used for the FAISS search: float32, shape (dim,),
        L2-normalised, encoded from the query text WITH this repo's query instruction applied
        (see _apply_query_instruction). Exposed so callers that need the vector (serving's
        resolution stage, ADR-0062) reuse it instead of encoding the same text a second time.
        retrieve() delegates here, so its behaviour and return type are unchanged.

        created_before (datetime / Timestamp / ISO string; naive = UTC): keep only neighbours whose
        created_at is STRICTLY earlier (an equal timestamp is excluded). None = no filter, and the
        search is identical to the pre-filter code (k+1 candidates). When filtering, the candidate
        pool is over-fetched -- first max(4*(k+1), 32), then doubled until k hits survive or the
        whole index has been ranked -- so k results come back whenever k qualifying issues exist.
        Worst case (almost nothing qualifies) is one exhaustive ranking of the flat index, which
        costs about the same as the base search (IndexFlatIP scores every vector either way).
        A neighbour with no date is kept when strict_dates=False (serving) and raises
        MissingNeighbourDateError when strict_dates=True (recording/evaluation, fail closed). An
        index with no dates at all: serving applies no filter (logged once); strict raises.
        """
        assert self.index is not None, "Call build_index first"
        assert self.issue_numbers is not None
        assert self.texts is not None
        query_text = self._apply_query_instruction(query_text, apply_query_instruction)
        emb = self.model.encode(
            [query_text], normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)

        cutoff: np.datetime64 | None = None
        if created_before is not None:
            cutoff = to_utc_datetime64(created_before)
            if self.created_at is None:
                if strict_dates:
                    raise MissingNeighbourDateError(
                        f"{self.repo}: created_before given but the index carries no created_at"
                    )
                if not self._warned_dateless:
                    logger.warning(
                        "%s: index has no created_at; created_before filter not applied", self.repo
                    )
                    self._warned_dateless = True
                cutoff = None

        # Retrieve k+1 to account for possible self-exclusion; over-fetch when filtering.
        n_total = 0 if cutoff is None else int(self.index.ntotal)
        n_fetch = k + 1 if cutoff is None else max(4 * (k + 1), 32)
        while True:
            if cutoff is not None:
                n_fetch = min(n_fetch, n_total)
            scores, indices = self.index.search(emb, n_fetch)
            results = []
            for score, idx in zip(scores[0], indices[0], strict=False):
                if idx < 0:
                    continue
                num = int(self.issue_numbers[idx])
                if exclude_number is not None and num == exclude_number:
                    continue
                if cutoff is not None:
                    assert self.created_at is not None
                    ts = self.created_at[idx]
                    if np.isnat(ts):
                        if strict_dates:
                            raise MissingNeighbourDateError(
                                f"{self.repo}: neighbour #{num} has no created_at"
                            )
                    elif not ts < cutoff:
                        continue
                results.append({"number": num, "score": float(score), "text": self.texts[idx]})
                if len(results) >= k:
                    break
            if cutoff is None or len(results) >= k or n_fetch >= n_total:
                return results, emb[0]
            n_fetch *= 2

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
        meta: dict = {
            "repo": self.repo,
            "model_key": self.model_key,
            "issue_numbers": self.issue_numbers,
            "texts": self.texts,
        }
        if self.created_at is not None:  # key absent (not None) so undated metas stay byte-stable
            meta["created_at"] = self.created_at
        joblib.dump(meta, str(p / "meta.pkl"))
        logger.info("Saved index to %s", out_dir)

    @classmethod
    def load(cls, out_dir: str) -> SimilarIssueRetriever:
        p = Path(out_dir)
        t0 = time.perf_counter()
        meta = joblib.load(str(p / "meta.pkl"))
        logger.info(
            "startup_step %s.meta_pkl_load took %.1f ms",
            meta["repo"], (time.perf_counter() - t0) * 1000.0,
        )
        obj = cls(repo=meta["repo"], model_key=meta["model_key"])
        t0 = time.perf_counter()
        # faiss.read_index()'s return type is the generic Index base class, but save()
        # only ever writes an IndexFlatIP -- narrowing here matches the actual contract.
        obj.index = cast(faiss.IndexFlatIP, faiss.read_index(str(p / "index.faiss")))
        logger.info(
            "startup_step %s.faiss_read took %.1f ms",
            meta["repo"], (time.perf_counter() - t0) * 1000.0,
        )
        obj.issue_numbers = meta["issue_numbers"]
        obj.texts = meta["texts"]
        obj.created_at = meta.get("created_at")  # absent in metas saved before the filter
        if obj.created_at is not None and len(obj.created_at) != len(obj.issue_numbers):
            raise ValueError(f"{out_dir}: created_at length != issue_numbers length")
        return obj
